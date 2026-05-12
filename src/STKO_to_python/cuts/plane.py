"""Plane primitive for section cuts.

A :class:`Plane` is the universal cut object: a point + outward unit
normal. Picklable, hashable, immutable. Construction helpers cover the
common cases (horizontal, vertical, three-point, horizontal grid) and
the class exposes the geometric primitives that every cut kernel needs:

- :meth:`Plane.signed_distance` — scalar field d(x) = (x - p0) · n
- :meth:`Plane.side`            — -1/0/+1 classification with tolerance
- :meth:`Plane.intersect_segment`  — line-element cuts (beam kernel)
- :meth:`Plane.intersect_polygon`  — convex face cuts (shell kernel,
  later the polygon-clip step for the solid kernel)

Normals are auto-normalized on construction; a zero-length normal or a
degenerate three-point spec raises ``ValueError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Vec3 = tuple[float, float, float]


_AXIS_TO_NORMAL: dict[str, Vec3] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _to_vec3(v: Sequence[float] | np.ndarray, *, label: str) -> Vec3:
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"{label} must be length-3, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite, got {arr.tolist()}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass(frozen=True)
class Plane:
    """Oriented plane in 3D: point on the plane + unit normal.

    Parameters
    ----------
    point:
        Any point on the plane, length-3.
    normal:
        Outward normal, length-3. Need not be unit; auto-normalized.

    The signed distance from a point ``x`` to the plane is
    ``d(x) = (x - point) · normal``. Positive ``d`` means ``x`` is on
    the side the normal points to — call this the ``"positive"`` side.
    """

    point: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        p = _to_vec3(self.point, label="point")
        n = np.asarray(_to_vec3(self.normal, label="normal"), dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-300:
            raise ValueError(f"Plane normal must be nonzero, got {n.tolist()}.")
        n_unit = tuple((n / n_norm).tolist())
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "normal", n_unit)

    # ------------------------------------------------------------------ #
    # Numpy views
    # ------------------------------------------------------------------ #
    @property
    def point_arr(self) -> np.ndarray:
        return np.asarray(self.point, dtype=float)

    @property
    def normal_arr(self) -> np.ndarray:
        return np.asarray(self.normal, dtype=float)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def horizontal(cls, z: float) -> "Plane":
        """Plane perpendicular to global Z at the given elevation."""
        return cls(point=(0.0, 0.0, float(z)), normal=(0.0, 0.0, 1.0))

    @classmethod
    def vertical(cls, *, axis: str, at: float) -> "Plane":
        """Plane perpendicular to global X or Y at the given offset."""
        key = axis.strip().lower()
        if key not in ("x", "y"):
            raise ValueError(
                f"vertical(axis=...) must be 'x' or 'y', got {axis!r}."
            )
        normal = _AXIS_TO_NORMAL[key]
        point = tuple(float(at) * c for c in normal)
        return cls(point=point, normal=normal)

    @classmethod
    def from_three_points(
        cls,
        p1: Sequence[float] | np.ndarray,
        p2: Sequence[float] | np.ndarray,
        p3: Sequence[float] | np.ndarray,
        *,
        normal_hint: Sequence[float] | np.ndarray | None = None,
    ) -> "Plane":
        """Build a plane from three non-collinear points.

        Three points define an unoriented plane; ``normal_hint`` resolves
        which side counts as ``"positive"``. The returned plane's normal
        is flipped if its dot product with the hint is negative.
        """
        a = np.asarray(_to_vec3(p1, label="p1"), dtype=float)
        b = np.asarray(_to_vec3(p2, label="p2"), dtype=float)
        c = np.asarray(_to_vec3(p3, label="p3"), dtype=float)
        n = np.cross(b - a, c - a)
        n_norm = float(np.linalg.norm(n))
        # 1e-12 is small enough that any honest mesh won't trip it but
        # numerically degenerate (collinear) triples will.
        if n_norm < 1e-12:
            raise ValueError(
                "Three points are collinear or coincident — cannot define a plane."
            )
        n = n / n_norm
        if normal_hint is not None:
            hint = np.asarray(_to_vec3(normal_hint, label="normal_hint"), dtype=float)
            if float(np.dot(n, hint)) < 0.0:
                n = -n
        return cls(point=tuple(a.tolist()), normal=tuple(n.tolist()))

    @classmethod
    def horizontal_grid(cls, z: Iterable[float]) -> list["Plane"]:
        """List of horizontal planes at each elevation in ``z``."""
        return [cls.horizontal(float(zi)) for zi in z]

    # ------------------------------------------------------------------ #
    # Geometric primitives
    # ------------------------------------------------------------------ #
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance d(x) = (x - point) · normal.

        Accepts a single ``(3,)`` point or an ``(N, 3)`` batch and
        returns a scalar or ``(N,)`` array respectively.
        """
        pts = np.asarray(points, dtype=float)
        scalar = pts.ndim == 1
        if scalar:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"points must be shape (3,) or (N, 3), got {pts.shape}."
            )
        d = (pts - self.point_arr) @ self.normal_arr
        return float(d[0]) if scalar else d

    def side(self, points: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
        """Classify points relative to the plane.

        Returns ``+1`` on the positive side (along the normal), ``-1``
        on the negative side, and ``0`` within ``tol`` of the plane.
        """
        d = self.signed_distance(points)
        out = np.sign(d) if np.ndim(d) else float(np.sign(d))
        mask = np.abs(d) <= tol
        if np.ndim(out):
            out = np.where(mask, 0.0, out)
            return out.astype(np.int8)
        return np.int8(0) if mask else np.int8(out)

    def intersect_segment(
        self,
        p0: Sequence[float] | np.ndarray,
        p1: Sequence[float] | np.ndarray,
        *,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, float] | None:
        """Intersect a line segment ``p0 → p1`` with the plane.

        Returns ``(point, t)`` where ``point = p0 + t * (p1 - p0)`` and
        ``t ∈ [0, 1]``, or ``None`` if the segment does not cross the
        plane. A segment lying exactly in the plane returns ``None``
        (the caller must handle that edge case explicitly).
        """
        a = np.asarray(_to_vec3(p0, label="p0"), dtype=float)
        b = np.asarray(_to_vec3(p1, label="p1"), dtype=float)
        d0 = float(np.dot(a - self.point_arr, self.normal_arr))
        d1 = float(np.dot(b - self.point_arr, self.normal_arr))
        denom = d1 - d0
        if abs(denom) <= tol:
            return None
        t = -d0 / denom
        if t < -tol or t > 1.0 + tol:
            return None
        t = float(np.clip(t, 0.0, 1.0))
        return a + t * (b - a), t

    def intersect_polygon(
        self,
        vertices: np.ndarray | Sequence[Sequence[float]],
        *,
        tol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Intersect a convex polygon (in 3D) with the plane.

        ``vertices`` is shape ``(M, 3)``, ordered around the polygon.
        Returns the chord ``(q0, q1)`` along which the plane cuts the
        polygon, or ``None`` if the polygon does not cross the plane.
        Vertices lying within ``tol`` of the plane are treated as on it.

        For a convex polygon the intersection is at most a single chord
        (two points); if more than two crossings are found — which can
        happen on a non-convex face or under numerical noise — only the
        first and last along the edge walk are returned.
        """
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
            raise ValueError(
                f"vertices must be shape (M>=3, 3), got {verts.shape}."
            )
        d = self.signed_distance(verts)
        sgn = np.where(np.abs(d) <= tol, 0, np.sign(d).astype(int))

        # Polygon strictly on one side -> no crossing.
        if np.all(sgn > 0) or np.all(sgn < 0):
            return None

        m = verts.shape[0]
        crossings: list[np.ndarray] = []
        for i in range(m):
            j = (i + 1) % m
            si, sj = sgn[i], sgn[j]
            if si == 0:
                crossings.append(verts[i])
                continue
            if si * sj < 0:
                t = d[i] / (d[i] - d[j])
                crossings.append(verts[i] + t * (verts[j] - verts[i]))

        # Deduplicate near-coincident hits (e.g. when an entire vertex
        # is on the plane and both adjacent edges report it).
        if not crossings:
            return None
        uniq: list[np.ndarray] = [crossings[0]]
        for q in crossings[1:]:
            if not any(np.linalg.norm(q - u) <= tol for u in uniq):
                uniq.append(q)
        if len(uniq) < 2:
            return None
        return uniq[0], uniq[-1]
