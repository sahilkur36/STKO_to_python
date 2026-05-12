"""Unit tests for ``STKO_to_python.model.transforms``.

The transforms module is pure-math: every test below operates on a
fixed quaternion and checks the resulting matrix. No HDF5 / fixture
access needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.model.transforms import quaternion_to_rotation_matrix


def test_identity_quaternion_gives_identity_matrix() -> None:
    R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_single_input_returns_2d_matrix() -> None:
    R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert R.shape == (3, 3)


def test_batched_input_returns_3d_array() -> None:
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],   # identity
            [0.0, 1.0, 0.0, 0.0],   # 180° about X
        ]
    )
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (2, 3, 3)
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        R[1], np.diag([1.0, -1.0, -1.0]), atol=1e-15
    )


def test_column_quaternion_maps_local_x_to_global_z() -> None:
    """The STKO column quaternion from the elasticFrame fixture.

    A column running along global Z has a local-x axis aligned with
    global Z. So R @ (1, 0, 0) (local x basis vector) must equal
    (0, 0, 1) (global Z).
    """
    s = np.sqrt(0.5)
    R = quaternion_to_rotation_matrix(np.array([0.0, s, 0.0, s]))
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_returned_matrix_is_orthogonal() -> None:
    """R @ R.T = I for any unit quaternion."""
    q = np.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    R = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_normalizes_off_unit_input() -> None:
    """STKO writes quaternions to six digits, so |q|² is not exactly 1.

    The function must normalize internally so a 1.0000046-magnitude
    input still produces an exactly orthogonal matrix.
    """
    q_off = np.array([0.0, 0.707107, 0.0, 0.707107])  # exactly as STKO emits
    assert abs(np.linalg.norm(q_off) - 1.0) > 0
    R = quaternion_to_rotation_matrix(q_off)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_zero_quaternion_raises() -> None:
    with pytest.raises(ValueError, match="zero-magnitude"):
        quaternion_to_rotation_matrix(np.zeros(4))


def test_wrong_dim_raises() -> None:
    with pytest.raises(ValueError, match="size 4"):
        quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0]))


def test_conjugate_quaternion_gives_inverse_matrix() -> None:
    """For unit q with conjugate q* = (qw, -qx, -qy, -qz), R(q*) = R(q).T."""
    q = np.array([0.5, 0.5, 0.5, 0.5])  # arbitrary unit quaternion
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    R = quaternion_to_rotation_matrix(q)
    R_inv = quaternion_to_rotation_matrix(q_conj)
    np.testing.assert_allclose(R_inv, R.T, atol=1e-12)


def test_batch_agrees_with_per_element() -> None:
    """Vectorized result must equal looped scalar results."""
    rng = np.random.default_rng(seed=42)
    q = rng.normal(size=(5, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    batched = quaternion_to_rotation_matrix(q)
    one_by_one = np.stack(
        [quaternion_to_rotation_matrix(q[i]) for i in range(len(q))]
    )
    np.testing.assert_allclose(batched, one_by_one, atol=1e-15)
