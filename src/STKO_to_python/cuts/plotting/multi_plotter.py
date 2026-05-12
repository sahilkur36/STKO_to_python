"""Matplotlib plotter bound to a :class:`MultiCutResult`.

Three plot kinds in v1:

- :meth:`overlay_time_history` — one trace per case on a single time
  axis. Visualizes variability of the cut force across an ensemble.
- :meth:`case_envelope_bars` — one bar per case, peak/max/min for the
  chosen component. Compact comparison of case-level demand.
- :meth:`case_scatter` — peak of one component vs peak of another,
  one point per case. Useful for biaxial demand plots.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..multi_cut import MultiCutResult


_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_AGGS = ("max", "min", "peak_abs")


class MultiCutPlotter:
    """Plotter bound to one :class:`MultiCutResult`."""

    def __init__(self, multi: "MultiCutResult") -> None:
        self._multi = multi

    def __repr__(self) -> str:
        return f"<MultiCutPlotter bound to {self._multi!r}>"

    # ------------------------------------------------------------------ #
    # Overlay time history
    # ------------------------------------------------------------------ #
    def overlay_time_history(
        self,
        component: str = "Fx",
        *,
        cases: Iterable[str] | None = None,
        ax: "Axes | None" = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Plot every case's time history of one component on one axes.

        Parameters
        ----------
        component : str
            ``Fx`` .. ``Mz``.
        cases : iterable of str, optional
            Subset of case names to plot. ``None`` plots all cases in
            insertion order.
        """
        df = self._multi.to_dataframe(component=component)
        if ax is None:
            _, ax = plt.subplots()
        case_subset = list(df.columns) if cases is None else list(cases)
        for case in case_subset:
            if case not in df.columns:
                raise KeyError(f"Unknown case {case!r}. Known: {list(df.columns)}")
            ax.plot(df.index.to_numpy(), df[case].to_numpy(), label=case, **plot_kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel(component)
        ax.set_title(f"Section cut overlay: {component}")
        if len(case_subset) <= 12:
            ax.legend()
        meta = {
            "kind": "overlay_time_history",
            "component": component,
            "cases": case_subset,
            "n_cases": len(case_subset),
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Case envelope bars
    # ------------------------------------------------------------------ #
    def case_envelope_bars(
        self,
        component: str = "Fx",
        *,
        agg: Literal["max", "min", "peak_abs"] = "peak_abs",
        cases: Iterable[str] | None = None,
        ax: "Axes | None" = None,
        **bar_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """One bar per case, peak (max/min/|peak|) of the chosen component."""
        if agg not in _AGGS:
            raise ValueError(f"agg must be one of {_AGGS}; got {agg!r}.")
        series = self._multi.peak_over_cases(component=component, agg=agg)
        if cases is not None:
            cases_list = list(cases)
            series = series.reindex(cases_list)
        if ax is None:
            _, ax = plt.subplots()
        positions = np.arange(len(series))
        ax.bar(positions, series.to_numpy(), **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(list(series.index), rotation=45, ha="right")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel(f"{component} ({agg})")
        ax.set_title(f"Per-case envelope: {component} ({agg})")
        meta = {
            "kind": "case_envelope_bars",
            "component": component,
            "agg": agg,
            "cases": list(series.index),
            "values": series.to_numpy(),
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Case scatter (peak vs peak)
    # ------------------------------------------------------------------ #
    def case_scatter(
        self,
        x_component: str = "Fx",
        y_component: str = "Fy",
        *,
        agg: Literal["max", "min", "peak_abs"] = "peak_abs",
        ax: "Axes | None" = None,
        annotate: bool = True,
        **scatter_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Scatter of per-case peaks: one point per case.

        Useful for biaxial demand plots (peak |Fx| vs peak |Fy| across
        a ground-motion ensemble). With ``annotate=True`` the points are
        labeled with their case names.
        """
        if agg not in _AGGS:
            raise ValueError(f"agg must be one of {_AGGS}; got {agg!r}.")
        x_series = self._multi.peak_over_cases(component=x_component, agg=agg)
        y_series = self._multi.peak_over_cases(component=y_component, agg=agg)
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(x_series.to_numpy(), y_series.to_numpy(), **scatter_kwargs)
        if annotate:
            for case, x, y in zip(x_series.index, x_series.to_numpy(), y_series.to_numpy()):
                ax.annotate(case, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"{x_component} ({agg})")
        ax.set_ylabel(f"{y_component} ({agg})")
        ax.set_title(f"Per-case: {y_component} vs {x_component} ({agg})")
        ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0.0, color="grey", linewidth=0.5, linestyle=":")
        meta = {
            "kind": "case_scatter",
            "x_component": x_component,
            "y_component": y_component,
            "agg": agg,
            "cases": list(x_series.index),
        }
        return ax, meta
