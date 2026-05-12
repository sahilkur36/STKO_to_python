"""Matplotlib plotter bound to a :class:`SectionSweep`.

Two plot kinds:

- :meth:`profile` — one aggregate (max / min / |peak|) per plane,
  plotted against the plane locator. The typical "story shear vs
  elevation" plot in earthquake engineering: vertical orientation has
  the locator on the y-axis (elevation) and the force on the x-axis.
- :meth:`heatmap` — time × plane grid of one component as an image.
  Useful for visualizing wave propagation through tall buildings or
  through soil columns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..sweep import SectionSweep


_AGGS = ("max", "min", "peak_abs")


class SectionSweepPlotter:
    """Plotter bound to one :class:`SectionSweep`."""

    def __init__(self, sweep: "SectionSweep") -> None:
        self._sweep = sweep

    def __repr__(self) -> str:
        return f"<SectionSweepPlotter bound to {self._sweep!r}>"

    # ------------------------------------------------------------------ #
    # Profile (peak per plane)
    # ------------------------------------------------------------------ #
    def profile(
        self,
        component: str = "Fx",
        *,
        agg: Literal["max", "min", "peak_abs"] = "peak_abs",
        axis: str | None = None,
        vertical: bool = True,
        ax: "Axes | None" = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Plot one aggregate per plane against the plane locator.

        Parameters
        ----------
        component : str
            ``Fx``, ``Fy``, ``Fz``, ``Mx``, ``My``, or ``Mz``.
        agg : str
            Which aggregate to plot: ``"max"``, ``"min"``, or
            ``"peak_abs"`` (default).
        axis : str | None
            Axis to use for the locator. ``None`` infers from the planes'
            shared normal (``z`` for horizontal cuts, ``x`` / ``y`` for
            vertical cuts).
        vertical : bool
            When ``True`` (default) the locator goes on the y-axis and
            the force/moment on the x-axis — the classic "story shear vs
            elevation" orientation. ``False`` swaps the axes.
        """
        if agg not in _AGGS:
            raise ValueError(f"agg must be one of {_AGGS}; got {agg!r}.")
        env = self._sweep.envelope()
        col = f"{component}_{agg}"
        if col not in env.columns:
            raise ValueError(
                f"Column {col!r} not found in envelope. Components: "
                f"Fx..Mz; aggregates: {_AGGS}."
            )
        locators = self._sweep.plane_locators(axis=axis)
        values = env[col].to_numpy()

        if ax is None:
            _, ax = plt.subplots()

        plot_label = label if label is not None else f"{component} ({agg})"
        if vertical:
            ax.plot(values, locators, marker="o", label=plot_label, **plot_kwargs)
            ax.set_xlabel(f"{component} ({agg})")
            ax.set_ylabel(f"plane locator [{axis or 'inferred'}]")
        else:
            ax.plot(locators, values, marker="o", label=plot_label, **plot_kwargs)
            ax.set_xlabel(f"plane locator [{axis or 'inferred'}]")
            ax.set_ylabel(f"{component} ({agg})")
        ax.axvline(0.0, color="grey", linewidth=0.5, linestyle=":") if vertical else \
            ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")
        ax.legend()
        ax.set_title(f"Section sweep profile: {component} ({agg})")
        meta = {
            "kind": "profile",
            "component": component,
            "agg": agg,
            "locators": locators,
            "values": values,
            "vertical": vertical,
        }
        return ax, meta

    # ------------------------------------------------------------------ #
    # Heatmap (time × plane)
    # ------------------------------------------------------------------ #
    def heatmap(
        self,
        component: str = "Fx",
        *,
        axis: str | None = None,
        ax: "Axes | None" = None,
        cmap: str = "RdBu_r",
        **imshow_kwargs: Any,
    ) -> tuple["Axes", dict]:
        """Time × plane heatmap of one component as an imshow.

        Y-axis is the plane locator (low to high), x-axis is time, color
        encodes the chosen component. ``cmap`` defaults to ``RdBu_r`` so
        the sign is immediately visible (red = positive, blue = negative)
        — diverging colormap centered at zero via ``TwoSlopeNorm`` when
        the data straddles zero.
        """
        df = self._sweep.to_dataframe(component=component)
        if df.empty:
            if ax is None:
                _, ax = plt.subplots()
            ax.set_title(f"Section sweep heatmap: {component} (empty)")
            return ax, {"kind": "heatmap", "component": component, "empty": True}

        locators = self._sweep.plane_locators(axis=axis)
        time = df.index.to_numpy()
        values = df.to_numpy().T  # rows = planes, cols = time

        if ax is None:
            _, ax = plt.subplots()

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        norm = None
        if vmin < 0 < vmax:
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        im = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=[float(time[0]), float(time[-1]), float(locators.min()), float(locators.max())],
            cmap=cmap,
            norm=norm,
            **imshow_kwargs,
        )
        ax.set_xlabel("time")
        ax.set_ylabel(f"plane locator [{axis or 'inferred'}]")
        ax.set_title(f"Section sweep heatmap: {component}")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(component)
        meta = {
            "kind": "heatmap",
            "component": component,
            "image": im,
            "values": values,
            "locators": locators,
            "time": time,
        }
        return ax, meta
