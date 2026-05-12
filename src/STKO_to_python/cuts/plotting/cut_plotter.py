"""Matplotlib plotter bound to a :class:`SectionCut`.

Five plot kinds in v1:

- :meth:`time_history` — F/M component vs time.
- :meth:`orbit` — one component vs another (e.g. Fx vs Fy lateral orbit).
- :meth:`envelope_bars` — peak (max, min, |peak|) per component as bars.
- :meth:`hysteresis` — force-vs-drift (capacity / hysteresis curve),
  pairs the cut with a :class:`DriftSpec`.
- :meth:`consistency_residual` — visualizes the Newton-3rd residual
  over time. Diagnostic.

Each method returns ``(ax, meta)`` matching the
``NodalResultsPlotter`` / ``Plot`` facade idiom in the rest of the
library. ``meta`` is a dict with the parameters that drove the plot —
useful for downstream consumers that want to label or compose further.

Geometry view (3D model + plane + contributing elements, matplotlib /
pyvista) is intentionally out-of-scope for this module; it lands in
:mod:`cuts.plotting.geometry_view`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..section_cut import SectionCut
    from ..specs import DriftSpec
    from ...core.dataset import MPCODataSet


_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_FORCE_LABEL = {"Fx": "$F_x$", "Fy": "$F_y$", "Fz": "$F_z$"}
_MOMENT_LABEL = {"Mx": "$M_x$", "My": "$M_y$", "Mz": "$M_z$"}


def _component_index(name: str) -> int:
    """Map a component name to its column index in the stacked ``(F, M)`` array."""
    try:
        return _COMPONENTS.index(name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown component {name!r}. Expected one of {_COMPONENTS}."
        ) from exc


def _component_label(name: str) -> str:
    """Pretty label for a component (TeX if applicable)."""
    return _FORCE_LABEL.get(name, _MOMENT_LABEL.get(name, name))


class SectionCutPlotter:
    """Matplotlib plotter bound to one :class:`SectionCut`.

    Held lazily on :attr:`SectionCut.plot` — instantiated per access so
    the cut itself can stay a plain frozen dataclass. The plotter holds
    only a reference to the cut, so creation is cheap.
    """

    def __init__(self, cut: "SectionCut") -> None:
        self._cut = cut

    def __repr__(self) -> str:
        return f"<SectionCutPlotter bound to {self._cut!r}>"

    # ------------------------------------------------------------------ #
    # Time history of one component
    # ------------------------------------------------------------------ #
    def time_history(
        self,
        component: str = "Fx",
        *,
        ax: "Axes | None" = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Plot one component (``"Fx"`` .. ``"Mz"``) versus time."""
        idx = _component_index(component)
        data = self._stacked()
        if ax is None:
            _, ax = plt.subplots()
        line_label = self._resolve_label(label, suffix=component)
        ax.plot(self._cut.time, data[:, idx], label=line_label, **plot_kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel(_component_label(component))
        ax.set_title(self._title_with_label(default=f"Section cut: {component}"))
        if line_label:
            ax.legend()
        meta = {
            "kind": "time_history",
            "component": component,
            "n_steps": self._cut.n_steps,
            "label": line_label,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Orbit (one component vs another)
    # ------------------------------------------------------------------ #
    def orbit(
        self,
        x: str = "Fx",
        y: str = "Fy",
        *,
        ax: "Axes | None" = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Trajectory of two components against one another."""
        ix = _component_index(x)
        iy = _component_index(y)
        data = self._stacked()
        if ax is None:
            _, ax = plt.subplots()
        line_label = self._resolve_label(label, suffix=f"{y} vs {x}")
        ax.plot(data[:, ix], data[:, iy], label=line_label, **plot_kwargs)
        ax.set_xlabel(_component_label(x))
        ax.set_ylabel(_component_label(y))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(self._title_with_label(default=f"Section cut orbit: {y} vs {x}"))
        if line_label:
            ax.legend()
        meta = {
            "kind": "orbit",
            "x_component": x,
            "y_component": y,
            "n_steps": self._cut.n_steps,
            "label": line_label,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Envelope bars
    # ------------------------------------------------------------------ #
    def envelope_bars(
        self,
        *,
        ax: "Axes | None" = None,
        show_minmax: bool = True,
        **bar_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Peak per component as a bar chart.

        With ``show_minmax=True`` (default), draws stacked min/max bars
        per component so the asymmetry between positive and negative
        peaks is visible. With ``show_minmax=False``, plots only the
        absolute peak.
        """
        env = self._cut.envelope()
        if ax is None:
            _, ax = plt.subplots()
        components = list(env.index)
        positions = np.arange(len(components))
        if show_minmax:
            max_vals = env["max"].to_numpy()
            min_vals = env["min"].to_numpy()
            ax.bar(positions, max_vals, label="max", **bar_kwargs)
            ax.bar(positions, min_vals, label="min", **bar_kwargs)
            ax.axhline(0.0, color="black", linewidth=0.5)
            ax.legend()
        else:
            ax.bar(positions, env["peak_abs"].to_numpy(), **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels([_component_label(c) for c in components])
        ax.set_ylabel("resultant")
        ax.set_title(self._title_with_label(default="Section cut envelope"))
        meta = {
            "kind": "envelope_bars",
            "components": components,
            "envelope": env,
            "show_minmax": show_minmax,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Hysteresis (force vs drift)
    # ------------------------------------------------------------------ #
    def hysteresis(
        self,
        force: str,
        drift: "DriftSpec",
        dataset: "MPCODataSet",
        *,
        ax: "Axes | None" = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Capacity / hysteresis: cut force component vs node-pair drift.

        ``drift`` is applied against ``dataset`` to recover the drift
        time series; it is paired step-for-step with the cut's
        ``force`` component. The two must share a model stage and step
        count — mismatches raise.
        """
        idx = _component_index(force)
        force_series = self._stacked()[:, idx]
        drift_series = drift.apply(dataset, model_stage=self._cut.model_stage)
        drift_values = drift_series.to_numpy()
        if drift_values.shape[0] != force_series.shape[0]:
            raise ValueError(
                f"Drift series has {drift_values.shape[0]} steps but cut has "
                f"{force_series.shape[0]}. Drift and cut must come from the "
                f"same model stage."
            )
        if ax is None:
            _, ax = plt.subplots()
        line_label = self._resolve_label(
            label, suffix=f"{force} vs {drift.label or 'drift'}",
        )
        ax.plot(drift_values, force_series, label=line_label, **plot_kwargs)
        ax.set_xlabel(drift.label or "drift")
        ax.set_ylabel(_component_label(force))
        ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_title(self._title_with_label(default=f"Hysteresis: {force} vs drift"))
        if line_label:
            ax.legend()
        meta = {
            "kind": "hysteresis",
            "force": force,
            "drift_spec": drift,
            "n_steps": force_series.shape[0],
            "label": line_label,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Consistency-check residual
    # ------------------------------------------------------------------ #
    def consistency_residual(
        self,
        dataset: "MPCODataSet",
        *,
        ax: "Axes | None" = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Per-step Newton-3rd residual over time, one line per component.

        Calls :meth:`SectionCut.consistency_check` under the hood and
        plots ``|residual|`` per component as a sanity-check diagnostic.
        Should sit at machine epsilon for a correct kernel.
        """
        _, residual = self._cut.consistency_check(dataset)
        if ax is None:
            _, ax = plt.subplots()
        for i, comp in enumerate(_COMPONENTS):
            ax.plot(
                self._cut.time, np.abs(residual[:, i]),
                label=_component_label(comp), **plot_kwargs,
            )
        ax.set_xlabel("time")
        ax.set_ylabel("|residual|")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.legend()
        ax.set_title(
            self._title_with_label(default="Newton-3rd residual (per component)")
        )
        meta = {
            "kind": "consistency_residual",
            "max_abs_residual": float(np.abs(residual).max(initial=0.0)),
            "n_steps": self._cut.n_steps,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _stacked(self) -> np.ndarray:
        """Stack ``(F, M)`` into a ``(n_steps, 6)`` array — column order Fx..Mz."""
        return np.concatenate([self._cut.F, self._cut.M], axis=1)

    def _resolve_label(self, explicit: str | None, *, suffix: str | None = None) -> str | None:
        if explicit is not None:
            return explicit
        if self._cut.spec.label:
            return f"{self._cut.spec.label} — {suffix}" if suffix else self._cut.spec.label
        return suffix

    def _title_with_label(self, *, default: str) -> str:
        if self._cut.spec.label:
            return self._cut.spec.label
        if self._cut.spec.name:
            return self._cut.spec.name
        return default
