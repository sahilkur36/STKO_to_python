"""Multi-case section cut — one :class:`SectionCutSpec`, many cases.

Parallel to how :class:`MPCOResults` aggregates :class:`NodalResults`
across ground-motion ensembles, :class:`MultiCutResult` aggregates
:class:`SectionCut` instances across cases (ground motions, scenarios,
parameter studies). The canonical state is ``dict[case_name -> SectionCut]``;
factories cover the two natural entry points:

- :meth:`MultiCutResult.from_datasets` — run the kernel against a dict
  of live :class:`MPCODataSet` instances. Convenient for small studies
  done in one Python session.
- :meth:`MultiCutResult.from_cuts` — assemble from pre-computed
  :class:`SectionCut` instances (probably loaded from pickle). The
  workflow for large ground-motion ensembles: compute each cut in its
  own batch job, pickle it, then aggregate.

The aggregator validates that all cuts share the same spec — otherwise
cross-case comparisons aren't comparing the same thing.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping

import numpy as np
import pandas as pd

from .section_cut import SectionCut
from .specs import SectionCutSpec

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class MultiCutResult:
    """Same :class:`SectionCutSpec` evaluated across many cases.

    Attributes
    ----------
    cuts : dict[str, SectionCut]
        Per-case results, keyed by an arbitrary case name (e.g. ground
        motion id, scenario name).
    spec : SectionCutSpec
        Shared spec all cases were computed from. Carried explicitly
        rather than re-derived so it survives empty containers too.
    """

    cuts: dict[str, SectionCut] = field(default_factory=dict)
    spec: SectionCutSpec | None = None

    def __post_init__(self) -> None:
        if self.spec is None and self.cuts:
            # Pick the first cut's spec as canonical; verify the rest match.
            spec0 = next(iter(self.cuts.values())).spec
            object.__setattr__(self, "spec", spec0)
        # Strict spec-equality across cuts is enforced in the factories
        # (where it's a setup error), not here (where the user may have
        # built MultiCutResult manually for a deliberate reason).

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_datasets(
        cls,
        datasets: Mapping[str, "MPCODataSet"],
        spec: SectionCutSpec,
        *,
        model_stage: str,
    ) -> "MultiCutResult":
        """Run the same spec against every dataset in ``datasets``."""
        cuts = {
            name: SectionCut.compute(spec, ds, model_stage=model_stage)
            for name, ds in datasets.items()
        }
        return cls(cuts=cuts, spec=spec)

    @classmethod
    def from_cuts(
        cls,
        cuts: Mapping[str, SectionCut],
        *,
        require_matching_spec: bool = True,
    ) -> "MultiCutResult":
        """Assemble from pre-computed cuts.

        With ``require_matching_spec=True`` (default) every cut must
        share the same :class:`SectionCutSpec`. Setting it ``False`` is
        an escape hatch for advanced workflows (e.g. cuts that differ
        only in ``label`` or ``name`` cosmetic fields) — use sparingly.
        """
        cuts_dict = dict(cuts)
        if not cuts_dict:
            return cls(cuts={}, spec=None)
        spec0 = next(iter(cuts_dict.values())).spec
        if require_matching_spec:
            for name, cut in cuts_dict.items():
                if cut.spec != spec0:
                    raise ValueError(
                        f"Case {name!r} has a different spec than the first "
                        f"case. Set require_matching_spec=False to allow."
                    )
        return cls(cuts=cuts_dict, spec=spec0)

    # ------------------------------------------------------------------ #
    # Container protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.cuts)

    def __getitem__(self, case_name: str) -> SectionCut:
        return self.cuts[case_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.cuts)

    def __contains__(self, case_name: object) -> bool:
        return case_name in self.cuts

    def __repr__(self) -> str:
        spec_label = ""
        if self.spec is not None and self.spec.label:
            spec_label = f", label={self.spec.label!r}"
        return f"MultiCutResult(n_cases={len(self.cuts)}{spec_label})"

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def case_names(self) -> tuple[str, ...]:
        return tuple(self.cuts)

    @property
    def n_cases(self) -> int:
        return len(self.cuts)

    @property
    def is_empty(self) -> bool:
        return not self.cuts

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #
    def envelope_per_case(self) -> pd.DataFrame:
        """One row per case, 18 peak-stat columns (6 components × {max, min, peak_abs}).

        Index is ``case_name``.
        """
        rows: list[dict] = []
        for name, cut in self.cuts.items():
            env = cut.envelope()
            row: dict = {"case": name}
            for comp in _COMPONENTS:
                if comp in env.index:
                    row[f"{comp}_max"] = float(env.loc[comp, "max"])
                    row[f"{comp}_min"] = float(env.loc[comp, "min"])
                    row[f"{comp}_peak_abs"] = float(env.loc[comp, "peak_abs"])
                else:
                    row[f"{comp}_max"] = np.nan
                    row[f"{comp}_min"] = np.nan
                    row[f"{comp}_peak_abs"] = np.nan
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("case")

    def peak_over_cases(
        self,
        component: str = "Fx",
        agg: str = "peak_abs",
    ) -> pd.Series:
        """One Series indexed by case_name with a single aggregated value.

        ``agg`` must be one of ``"max"``, ``"min"``, ``"peak_abs"`` —
        same as :meth:`SectionCut.envelope` columns.
        """
        if component not in _COMPONENTS:
            raise ValueError(
                f"Unknown component {component!r}. Expected one of {_COMPONENTS}."
            )
        if agg not in ("max", "min", "peak_abs"):
            raise ValueError(f"agg must be 'max', 'min', or 'peak_abs'; got {agg!r}.")
        if self.is_empty:
            return pd.Series(dtype=float, name=f"{component}_{agg}")
        env = self.envelope_per_case()
        return env[f"{component}_{agg}"].rename(f"{component}_{agg}")

    def to_dataframe(
        self,
        component: str = "Fx",
    ) -> pd.DataFrame:
        """Wide DataFrame for one component: rows = time, cols = case name.

        Cases with different time axes are aligned by step index (column
        per case), and the index becomes the first non-empty case's time
        axis. Mismatched step counts raise — multi-case overlay is only
        meaningful when the cases share an analysis step axis.
        """
        if component not in _COMPONENTS:
            raise ValueError(
                f"Unknown component {component!r}. Expected one of {_COMPONENTS}."
            )
        if self.is_empty:
            return pd.DataFrame()
        comp_idx = _COMPONENTS.index(component)
        # Pick a reference time axis from the first non-empty cut.
        ref_time = None
        ref_n_steps = -1
        for cut in self.cuts.values():
            if not cut.is_empty:
                ref_time = cut.time
                ref_n_steps = cut.n_steps
                break
        if ref_time is None:
            return pd.DataFrame(
                index=pd.Index([], name="time"),
                columns=list(self.cuts.keys()),
            )
        # Validate step counts.
        for name, cut in self.cuts.items():
            if not cut.is_empty and cut.n_steps != ref_n_steps:
                raise ValueError(
                    f"Case {name!r} has {cut.n_steps} steps but reference "
                    f"has {ref_n_steps}. Overlay requires matching step counts."
                )
        data = np.full((ref_n_steps, len(self.cuts)), np.nan, dtype=float)
        for j, (_, cut) in enumerate(self.cuts.items()):
            if cut.is_empty:
                continue
            stacked = np.concatenate([cut.F, cut.M], axis=1)
            data[:, j] = stacked[:, comp_idx]
        return pd.DataFrame(
            data,
            index=pd.Index(ref_time, name="time"),
            columns=list(self.cuts.keys()),
        )

    # ------------------------------------------------------------------ #
    # Plotting (lazy)
    # ------------------------------------------------------------------ #
    @property
    def plot(self):
        from .plotting.multi_plotter import MultiCutPlotter
        return MultiCutPlotter(self)

    # ------------------------------------------------------------------ #
    # Pickle
    # ------------------------------------------------------------------ #
    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        opener = gzip.open if compress else open
        with opener(p, "wb") as f:
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(cls, path: str | Path) -> "MultiCutResult":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj
