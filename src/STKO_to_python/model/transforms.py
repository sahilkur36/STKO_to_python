"""Coordinate-frame transforms backed by ``.cdata`` ``*LOCAL_AXES``.

STKO emits one orientation quaternion per element in the
``*LOCAL_AXES`` section of the ``.cdata`` sidecar. The quaternion
describes the element-local frame relative to global; applying the
corresponding rotation matrix transforms a vector from element-local
coordinates to global coordinates.

This module provides:

- :func:`quaternion_to_rotation_matrix` — pure numpy, vectorized,
  ``(qw, qx, qy, qz)`` → ``(3, 3)`` (or batched).
- :meth:`STKO_to_python.model.cdata_reader.CDataReader.rotation_matrix`
  and
  :meth:`STKO_to_python.model.cdata_reader.CDataReader.rotation_matrices` —
  reader-side conveniences that fold the quaternion lookup and the
  conversion into one call.

Convention
----------
A quaternion ``q = (qw, qx, qy, qz)`` from the ``*LOCAL_AXES`` section
describes the orientation of the element-local frame relative to
global. The returned matrix ``R`` rotates a vector from the
*element-local* frame to *global* coordinates::

    v_global = R @ v_local

Inverse: ``v_local = R.T @ v_global``. Verified against the
``elasticFrame_mesh_results`` fixture (columns parallel to global Z,
beams parallel to global X) and the ``force`` / ``localForce``
recorders on the same elements.
"""
from __future__ import annotations

import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert one or more unit quaternions to ``(3, 3)`` rotation matrices.

    Args:
        q: Either a single quaternion as shape ``(4,)`` or a batch as
            ``(N, 4)``. Component order is ``(qw, qx, qy, qz)`` — the
            same order STKO writes in ``*LOCAL_AXES``.

    Returns:
        Shape ``(3, 3)`` for a single quaternion, ``(N, 3, 3)`` for a
        batch. The returned matrix transforms a vector from the
        element-local frame to global coordinates::

            v_global = R @ v_local

    Notes:
        The quaternion is normalized internally so the returned ``R``
        is exactly orthogonal even when the input has limited precision.
        STKO emits ``*LOCAL_AXES`` quaternions at six significant
        figures; without normalization the resulting matrix is off
        orthogonality by ~1e-6, which compounds into errors of a few
        units in 1e6-magnitude force outputs.

    Raises:
        ValueError: if the last dimension is not 4, or if any input
        quaternion has zero magnitude.
    """
    arr = np.asarray(q, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] != 4:
        raise ValueError(
            f"Expected last dimension of size 4 (qw, qx, qy, qz); got {arr.shape!r}."
        )

    norms = np.linalg.norm(arr, axis=-1)
    if np.any(norms == 0):
        raise ValueError("quaternion_to_rotation_matrix: zero-magnitude quaternion.")
    arr = arr / norms[:, None]

    qw = arr[:, 0]
    qx = arr[:, 1]
    qy = arr[:, 2]
    qz = arr[:, 3]

    R = np.empty((arr.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2.0 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2.0 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2.0 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2.0 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2.0 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2.0 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)

    return R[0] if single else R
