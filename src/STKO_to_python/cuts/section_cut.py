"""User-facing :class:`SectionCut` dataclass.

Wraps a kernel result with broker-style accessors (resultant,
envelope, at_step, at_time, to_dataframe) and structural validators
(:meth:`consistency_check`, :meth:`compare_to`). Mirrors the
:class:`NodalResults` / :class:`ElementResults` shape so users feel
the same idiom across the library.

Validator philosophy
--------------------
Equilibrium against support reactions is too narrow a reference —
DRM, PML / absorbing boundaries, and explicit dynamics frequently
have no classical reactions. Two universal checks ship instead:

- :meth:`consistency_check` — flip ``spec.side`` and verify the two
  cuts sum to zero (Newton's 3rd law). Free, works for any model,
  catches sign and rotation bugs.
- :meth:`compare_to` — take a second cut at a parallel plane in a
  no-external-load region; the resultants (transferred to a common
  reference point) must match. Position invariance / free-body band.

What this *intentionally does not* try to do: tell the user whether
their cut "balances" against an external loading they haven't
specified. That depends on the model — gravity static, DRM effective
forces, PML radiation, distributed seismic input, inertia — and is
the user's responsibility to compose.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .kernels.beam import BeamIntersection
from .kernels.beam_resultant import compute_beam_cut
from .kernels.shell import ShellIntersection, compute_shell_cut
from .specs import SectionCutSpec

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class SectionCut:
    """Result of cutting a model with a plane and integrating tractions.

    Created via :meth:`SectionCut.compute` or :meth:`MPCODataSet.section_cut`.

    Attributes
    ----------
    spec : SectionCutSpec
        The cut definition this result was computed from.
    model_stage : str
        Model stage the data was read from.
    F : np.ndarray
        ``(n_steps, 3)`` — total force the discarded side exerts on the
        kept side, in global frame.
    M : np.ndarray
        ``(n_steps, 3)`` — total moment about :attr:`centroid`, global.
    time : np.ndarray
        ``(n_steps,)``.
    centroid : np.ndarray
        ``(3,)`` — reference point for the moment summation.
    intersections : tuple[BeamIntersection, ...]
        Per-beam intersection records.
    per_beam_F, per_beam_M_at_intersection : dict[int, np.ndarray]
        Per-beam contributions (force; moment evaluated at the
        intersection point, before the centroid arm transfer).
    """

    spec: SectionCutSpec
    model_stage: str
    F: np.ndarray
    M: np.ndarray
    time: np.ndarray
    centroid: np.ndarray
    intersections: tuple[BeamIntersection, ...]
    per_beam_F: dict[int, np.ndarray] = field(default_factory=dict)
    per_beam_M_at_intersection: dict[int, np.ndarray] = field(default_factory=dict)
    shell_intersections: tuple[ShellIntersection, ...] = ()
    per_shell_F: dict[int, np.ndarray] = field(default_factory=dict)
    per_shell_M_at_midpoint: dict[int, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def compute(
        cls,
        spec: SectionCutSpec,
        dataset: "MPCODataSet",
        *,
        model_stage: str,
    ) -> "SectionCut":
        """Compute the cut against ``dataset`` at ``model_stage``.

        Composes the beam and shell kernels and aggregates ``(F, M)``
        about a shared centroid (the mean of all reference points —
        beam intersection points and shell chord midpoints). The solid
        kernel will plug into the same composition in v2.0.
        """
        # Share one PolygonClipper between the two kernels so the plane
        # basis isn't recomputed twice when bounding_polygon is set.
        clipper = None
        if spec.bounding_polygon is not None:
            from .geometry import prepare_clipper
            clipper = prepare_clipper(spec.plane, spec.bounding_polygon)

        beam = compute_beam_cut(
            dataset, spec, model_stage=model_stage, clipper=clipper,
        )
        shell = compute_shell_cut(
            dataset, spec, model_stage=model_stage, clipper=clipper,
        )

        # Pick a time axis from whichever kernel found something. Beam
        # and shell readers route through the same broker, so when both
        # have intersections their time arrays are identical by
        # construction.
        if not beam.is_empty:
            time = beam.time
        elif not shell.is_empty:
            time = shell.time
        else:
            return cls(
                spec=spec,
                model_stage=model_stage,
                F=np.zeros((0, 3)),
                M=np.zeros((0, 3)),
                time=np.zeros((0,)),
                centroid=np.zeros(3),
                intersections=(),
                per_beam_F={},
                per_beam_M_at_intersection={},
                shell_intersections=(),
                per_shell_F={},
                per_shell_M_at_midpoint={},
            )

        # Aggregate F + M about the shared centroid of every reference
        # point in both kernels.
        ref_points = []
        for b in beam.intersections:
            ref_points.append(b.point_arr)
        for s in shell.intersections:
            ref_points.append(s.chord_midpoint)
        centroid = np.mean(np.stack(ref_points, axis=0), axis=0)

        F_total = np.zeros((time.shape[0], 3), dtype=float)
        M_total = np.zeros_like(F_total)
        for ix in beam.intersections:
            Fi = beam.per_beam_F[ix.element_id]
            Mi = beam.per_beam_M_at_intersection[ix.element_id]
            arm = ix.point_arr - centroid
            F_total += Fi
            M_total += Mi + np.cross(arm, Fi)
        for ix in shell.intersections:
            Fi = shell.per_shell_F[ix.element_id]
            Mi = shell.per_shell_M_at_midpoint[ix.element_id]
            arm = ix.chord_midpoint - centroid
            F_total += Fi
            M_total += Mi + np.cross(arm, Fi)

        return cls(
            spec=spec,
            model_stage=model_stage,
            F=F_total,
            M=M_total,
            time=time,
            centroid=centroid,
            intersections=beam.intersections,
            per_beam_F=dict(beam.per_beam_F),
            per_beam_M_at_intersection=dict(beam.per_beam_M_at_intersection),
            shell_intersections=shell.intersections,
            per_shell_F=dict(shell.per_shell_F),
            per_shell_M_at_midpoint=dict(shell.per_shell_M_at_midpoint),
        )

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def plot(self):
        """Lazy matplotlib plotter bound to this cut.

        Each access returns a fresh ``SectionCutPlotter``. That's fine —
        the plotter is just a thin wrapper holding a reference to this
        cut; construction cost is negligible. Kept as a property (not a
        cached attribute) because :class:`SectionCut` is a frozen
        dataclass.
        """
        from .plotting.cut_plotter import SectionCutPlotter
        return SectionCutPlotter(self)

    @property
    def is_empty(self) -> bool:
        return (
            len(self.intersections) == 0
            and len(self.shell_intersections) == 0
        )

    @property
    def contributing_element_ids(self) -> tuple[int, ...]:
        """All element ids whose contribution the cut sums up — beams
        first, then shells, each block sorted by element_id."""
        beam_ids = [ix.element_id for ix in self.intersections]
        shell_ids = [ix.element_id for ix in self.shell_intersections]
        return tuple(beam_ids + shell_ids)

    def __repr__(self) -> str:
        label = f", label={self.spec.label!r}" if self.spec.label else ""
        n_total = len(self.intersections) + len(self.shell_intersections)
        return (
            f"SectionCut(stage={self.model_stage!r}, n_steps={self.n_steps}, "
            f"n_intersections={n_total}, "
            f"side={self.spec.side!r}{label})"
        )

    # ------------------------------------------------------------------ #
    # Resultant accessors
    # ------------------------------------------------------------------ #
    def resultant(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(F, M)`` as copies — same shape as :attr:`F` / :attr:`M`."""
        return self.F.copy(), self.M.copy()

    def at_step(self, k: int) -> pd.Series:
        """Snapshot of the resultant at step ``k`` as a 6-element Series.

        Indexed by ``("Fx", "Fy", "Fz", "Mx", "My", "Mz")``.
        """
        if not (0 <= k < self.n_steps):
            raise IndexError(
                f"step index {k} out of range [0, {self.n_steps})."
            )
        values = np.concatenate([self.F[k], self.M[k]])
        return pd.Series(
            values, index=list(_COMPONENTS),
            name=f"step={k}, time={float(self.time[k]):.6g}",
        )

    def at_time(self, t: float, *, tol: float | None = None) -> pd.Series:
        """Snapshot at the step closest to time ``t``.

        ``tol`` (optional) raises if no step lies within ``tol`` of ``t``.
        """
        if self.n_steps == 0:
            raise ValueError("Cannot snapshot an empty cut.")
        k = int(np.argmin(np.abs(self.time - t)))
        if tol is not None and abs(float(self.time[k]) - t) > tol:
            raise ValueError(
                f"No step within tol={tol} of t={t}. Closest is "
                f"step {k} at t={float(self.time[k])}."
            )
        return self.at_step(k)

    def envelope(self) -> pd.DataFrame:
        """Peak statistics per component.

        Returns a DataFrame indexed by component name with columns
        ``max``, ``min``, ``peak_abs``, ``peak_abs_step``,
        ``peak_abs_time``.
        """
        data = np.concatenate([self.F, self.M], axis=1)  # (n_steps, 6)
        if data.size == 0:
            return pd.DataFrame(
                columns=["max", "min", "peak_abs", "peak_abs_step", "peak_abs_time"],
                index=pd.Index(list(_COMPONENTS), name="component"),
            )
        rows = []
        for i, name in enumerate(_COMPONENTS):
            col = data[:, i]
            k_peak = int(np.argmax(np.abs(col)))
            rows.append({
                "component": name,
                "max": float(col.max()),
                "min": float(col.min()),
                "peak_abs": float(np.abs(col).max()),
                "peak_abs_step": k_peak,
                "peak_abs_time": float(self.time[k_peak]),
            })
        return pd.DataFrame(rows).set_index("component")

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame: one row per step, columns Fx..Mz.

        Indexed by ``time`` for easy plotting.
        """
        data = np.concatenate([self.F, self.M], axis=1)
        return pd.DataFrame(
            data,
            columns=list(_COMPONENTS),
            index=pd.Index(self.time, name="time"),
        )

    # ------------------------------------------------------------------ #
    # Moment-arm transfer
    # ------------------------------------------------------------------ #
    def moment_about(self, reference_point: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Transfer the moment to a different reference point.

        ``M_about_P = M_about_centroid + (centroid - P) × F``.

        Used by :meth:`compare_to` to compare two cuts whose centroids
        differ. Also useful for users that want the moment about a
        specific structural point (e.g. a column base).
        """
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != 3:
            raise ValueError(
                f"reference_point must be length-3, got shape {ref.shape}."
            )
        arm = self.centroid - ref
        # np.cross broadcasts (3,) against (n_steps, 3) row-wise.
        return self.M + np.cross(arm, self.F)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    def consistency_check(
        self,
        dataset: "MPCODataSet",
        *,
        atol: float = 1e-6,
        rtol: float = 1e-9,
    ) -> tuple[bool, np.ndarray]:
        """Newton-3rd-law internal consistency check.

        Recomputes the same cut with the opposite ``side`` and verifies
        ``F_positive + F_negative ≈ 0`` and ``M_positive + M_negative ≈ 0``.
        Any honest kernel must satisfy this by construction; failures
        indicate a sign-convention bug, a rotation mistake, or a dedup
        miss between the two side calls.

        Independent of boundary conditions and analysis type — works
        for DRM, PML, explicit dynamics, anything.

        Returns
        -------
        (ok, residual) : tuple[bool, np.ndarray]
            ``residual`` is shape ``(n_steps, 6)`` — the per-step sum
            ``[F + F_neg, M + M_neg]``. ``ok`` is True when every
            entry satisfies ``|x| <= atol + rtol * |reference|``.
        """
        flipped = replace(
            self.spec,
            side="negative" if self.spec.side == "positive" else "positive",
        )
        other = SectionCut.compute(flipped, dataset, model_stage=self.model_stage)
        # other.centroid should equal self.centroid (same plane, same
        # intersections), but to be defensive transfer to a common ref
        # before comparing moments.
        other_M_at_self_centroid = other.moment_about(self.centroid)
        sum_F = self.F + other.F
        sum_M = self.M + other_M_at_self_centroid
        residual = np.concatenate([sum_F, sum_M], axis=1)
        scale = atol + rtol * (
            np.maximum(np.abs(self.F).max(initial=0.0), np.abs(self.M).max(initial=0.0))
            + np.maximum(np.abs(other.F).max(initial=0.0), np.abs(other.M).max(initial=0.0))
        )
        ok = bool(np.all(np.abs(residual) <= scale))
        return ok, residual

    def compare_to(
        self,
        other: "SectionCut",
        *,
        atol: float = 1e-6,
        rtol: float = 1e-9,
    ) -> tuple[bool, np.ndarray]:
        """Compare two parallel cuts (no external load between them).

        For a static analysis with no eleLoads between the two cut
        planes, the resultants — both transferred to a common reference
        point — must be equal. For dynamic analysis the difference
        equals the impulse from inertial forces in the band between
        the cuts.

        Both cuts must share the same time axis; their centroids may
        differ. Moments are transferred to ``self.centroid`` before the
        comparison.

        Returns
        -------
        (ok, residual) : tuple[bool, np.ndarray]
            ``residual`` is shape ``(n_steps, 6)`` — the per-step
            ``[F_other - F_self, M_other_at_self_centroid - M_self]``.
        """
        if other.n_steps != self.n_steps:
            raise ValueError(
                f"Cuts have different step counts: self={self.n_steps}, "
                f"other={other.n_steps}."
            )
        delta_F = other.F - self.F
        delta_M = other.moment_about(self.centroid) - self.M
        residual = np.concatenate([delta_F, delta_M], axis=1)
        scale = atol + rtol * (
            np.abs(self.F).max(initial=0.0)
            + np.abs(self.M).max(initial=0.0)
            + np.abs(other.F).max(initial=0.0)
            + np.abs(other.M).max(initial=0.0)
        )
        ok = bool(np.all(np.abs(residual) <= scale))
        return ok, residual

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
        """Pickle the cut to ``path``.

        Matches the rest of the library: a ``.gz`` suffix triggers
        gzip compression unless explicitly overridden. The pickled
        payload contains only the spec, the result arrays, and the
        intersection records — no live dataset reference.
        """
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"
        opener = gzip.open if compress else open
        with opener(p, "wb") as f:
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(cls, path: str | Path) -> "SectionCut":
        p = Path(path)
        opener = gzip.open if p.suffix.lower() == ".gz" else open
        with opener(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj
