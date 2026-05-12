"""Section sweep — a grid of :class:`SectionCut` instances.

A :class:`SectionSweep` is the natural data structure for
story-shear-vs-elevation profiles, soil-column shear depth plots,
through-thickness force distributions, and anything else that wants
"the same cut at many parallel positions." Pure composition over the
existing beam kernel; no new kernel math.

API shape:

- :meth:`SectionSweep.compute` — many planes + one shared filter +
  one model stage → many cuts.
- :meth:`SectionSweep.from_specs` — escape hatch when each plane needs
  a different filter (spec list in, sweep out).
- :meth:`envelope` — per-plane peak statistics as a DataFrame indexed
  by plane index.
- :meth:`to_dataframe` — wide DataFrame for one component, rows =
  time, columns = plane index (handy for heat maps).
- :meth:`plane_locators` — natural coordinate per plane along an axis,
  inferred from the planes' shared normal when not specified.
- :attr:`plot` — lazy :class:`SectionSweepPlotter` with ``profile`` and
  ``heatmap`` matplotlib methods.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .plane import Plane
from .section_cut import SectionCut
from .specs import SectionCutSpec

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class SectionSweep:
    """A sequence of :class:`SectionCut`s sharing one model stage.

    All cuts in the sweep are computed against the same dataset and
    model stage, so they share a time axis. The per-cut planes and
    optionally the per-cut filters can differ (the former is the
    typical case; use :meth:`from_specs` for the latter).

    Attributes
    ----------
    cuts : tuple[SectionCut, ...]
        One entry per plane, in the order the planes were given.
    model_stage : str
        The model stage all cuts in this sweep were computed from.
    """

    cuts: tuple[SectionCut, ...]
    model_stage: str

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def compute(
        cls,
        planes: Sequence[Plane],
        dataset: "MPCODataSet",
        *,
        model_stage: str,
        selection_set_name: str | Sequence[str] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        element_ids: Sequence[int] | None = None,
        side: str = "positive",
        label: str | None = None,
    ) -> "SectionSweep":
        """Compute a sweep with the **same filter** at each plane.

        Build one :class:`SectionCutSpec` per plane by varying ``plane``
        and holding everything else constant. The most common case for
        story-shear profiles, depth scans, etc.
        """
        if not planes:
            return cls(cuts=(), model_stage=model_stage)
        cuts: list[SectionCut] = []
        for plane in planes:
            spec = SectionCutSpec(
                plane=plane,
                selection_set_name=selection_set_name,
                selection_set_id=selection_set_id,
                element_ids=element_ids,
                side=side,
                label=label,
            )
            cuts.append(SectionCut.compute(spec, dataset, model_stage=model_stage))
        return cls(cuts=tuple(cuts), model_stage=model_stage)

    @classmethod
    def from_specs(
        cls,
        specs: Sequence[SectionCutSpec],
        dataset: "MPCODataSet",
        *,
        model_stage: str,
    ) -> "SectionSweep":
        """Compute a sweep from a list of pre-built specs.

        Use this when each plane needs its own filter (e.g. different
        column subsets per story). For a uniform filter across all
        planes, :meth:`compute` is more concise.
        """
        cuts = tuple(
            SectionCut.compute(s, dataset, model_stage=model_stage) for s in specs
        )
        return cls(cuts=cuts, model_stage=model_stage)

    # ------------------------------------------------------------------ #
    # Container protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.cuts)

    def __getitem__(self, i: int) -> SectionCut:
        return self.cuts[i]

    def __iter__(self) -> Iterator[SectionCut]:
        return iter(self.cuts)

    def __repr__(self) -> str:
        return (
            f"SectionSweep(n_planes={len(self.cuts)}, "
            f"model_stage={self.model_stage!r})"
        )

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def n_planes(self) -> int:
        return len(self.cuts)

    @property
    def n_steps(self) -> int:
        # All cuts share model_stage and therefore the time axis. Use
        # the first non-empty cut's step count; fall back to 0.
        for cut in self.cuts:
            if not cut.is_empty:
                return cut.n_steps
        return 0

    @property
    def time(self) -> np.ndarray:
        for cut in self.cuts:
            if not cut.is_empty:
                return cut.time
        return np.zeros((0,))

    @property
    def is_empty(self) -> bool:
        return not self.cuts or all(c.is_empty for c in self.cuts)

    # ------------------------------------------------------------------ #
    # Aggregations
    # ------------------------------------------------------------------ #
    def envelope(self) -> pd.DataFrame:
        """Peak statistics per plane, one row per cut.

        Columns are the 6 components × {max, min, peak_abs}: 18 columns
        total, named ``<Component>_<stat>`` (e.g. ``Fx_max``,
        ``Mz_peak_abs``). Index is the plane index.
        """
        rows: list[dict] = []
        for i, cut in enumerate(self.cuts):
            env = cut.envelope()
            row: dict = {"plane_index": i}
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
        return pd.DataFrame(rows).set_index("plane_index")

    def to_dataframe(self, component: str = "Fx") -> pd.DataFrame:
        """Wide DataFrame for one component: rows = time, cols = plane index.

        Empty cuts contribute a column of NaN aligned to the sweep's
        time axis. Useful as a source for heatmaps and overlay plots.
        """
        if component not in _COMPONENTS:
            raise ValueError(
                f"Unknown component {component!r}. Expected one of {_COMPONENTS}."
            )
        if self.is_empty:
            return pd.DataFrame()
        comp_idx = _COMPONENTS.index(component)
        time = self.time
        n_steps = time.shape[0]
        data = np.full((n_steps, len(self.cuts)), np.nan, dtype=float)
        for j, cut in enumerate(self.cuts):
            if cut.is_empty:
                continue
            stacked = np.concatenate([cut.F, cut.M], axis=1)
            data[:, j] = stacked[:, comp_idx]
        return pd.DataFrame(
            data,
            index=pd.Index(time, name="time"),
            columns=pd.Index(range(len(self.cuts)), name="plane_index"),
        )

    def plane_locators(self, axis: str | None = None) -> np.ndarray:
        """Return the natural-coordinate locator of each plane.

        For a sweep of axis-aligned planes (the common case — horizontal
        grid, vertical grid), all planes share a normal. Picking the
        coordinate of ``plane.point`` along that axis gives a clean
        numeric locator: ``z`` for horizontal cuts, ``x`` or ``y`` for
        vertical cuts.

        When ``axis`` is ``None`` and the cuts' first plane normal is
        nearly aligned with a global axis (tolerance 1e-6 on the
        component magnitude), the axis is inferred. Otherwise the
        caller must pass it explicitly.
        """
        if not self.cuts:
            return np.zeros((0,))
        if axis is None:
            n0 = self.cuts[0].spec.plane.normal_arr
            if abs(abs(n0[2]) - 1) < 1e-6:
                axis = "z"
            elif abs(abs(n0[0]) - 1) < 1e-6:
                axis = "x"
            elif abs(abs(n0[1]) - 1) < 1e-6:
                axis = "y"
            else:
                raise ValueError(
                    "Cannot infer axis from oblique plane normal "
                    f"{n0.tolist()}; pass axis='x'|'y'|'z' explicitly."
                )
        axis_key = axis.strip().lower()
        if axis_key not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}.")
        col = {"x": 0, "y": 1, "z": 2}[axis_key]
        return np.array(
            [cut.spec.plane.point_arr[col] for cut in self.cuts], dtype=float
        )

    # ------------------------------------------------------------------ #
    # Plotting (lazy)
    # ------------------------------------------------------------------ #
    @property
    def plot(self):
        from .plotting.sweep_plotter import SectionSweepPlotter
        return SectionSweepPlotter(self)

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
    def load_pickle(cls, path: str | Path) -> "SectionSweep":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj
