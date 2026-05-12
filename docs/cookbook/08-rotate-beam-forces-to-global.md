# Rotate beam-local forces and moments to the global frame

> Take OpenSees ``localForce`` output and rotate it back to global
> coordinates using the per-element quaternions STKO writes in the
> ``*LOCAL_AXES`` section of the ``.cdata`` sidecar. End-to-end
> verification against ``globalForces`` on the same elements.

The engineering question is: *I have section forces / moments in the
element-local frame and I need them in global — what's the rotation?*
STKO writes one quaternion per element to the ``.cdata`` sidecar;
v1.3.0+ parses it. v1.4.0 turns it into a rotation matrix you can
multiply through.

The fixture used here is
``stko_results_examples/elasticFrame/elasticFrame_mesh_results`` — 11
``5-ElasticBeam3d`` elements (six columns running along global Z, five
beams running along global X). It happens to record *both* ``force``
(global) and ``localForce`` (element-local), which means we can
**verify the rotation by round-tripping**: rotate ``localForce`` with
the cdata quaternion and check it matches ``force``.

```python
from pathlib import Path
import numpy as np

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "elasticFrame_mesh_results"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

---

## 1. Inspect the per-element orientation

`ds.cdata.local_axes` carries one unit quaternion per element, in
`(qw, qx, qy, qz)` order. STKO writes it relative to the global frame
— it describes how the *element-local* axes sit in global. Visiting
two elements with different orientations:

```python
print("column q:", ds.cdata.local_axes[1])
# column q: [0.       0.707107 0.       0.707107]
print("beam q  :", ds.cdata.local_axes[7])
# beam q  : [1. 0. 0. 0.]
```

The beam quaternion is the identity (beams run along global X — no
rotation needed). The column quaternion is a 180° rotation about the
in-plane diagonal that makes the element-local x-axis point along
global Z.

`ds.cdata.rotation_matrix(elem_id)` turns the quaternion into a
``(3, 3)`` matrix you apply to local-frame 3-vectors:

```python
R = ds.cdata.rotation_matrix(1)
print(R)
# [[-0.  0.  1.]
#  [ 0. -1.  0.]
#  [ 1.  0. -0.]]
print(R @ np.array([1.0, 0.0, 0.0]))   # local x in global coords
# [-2.22e-16  0.  1.]
```

The first column of `R` is `local-x in global`. For the column it's
``(0, 0, 1)`` — exactly along global Z. ✓

---

## 2. Compare a single time step in both frames

Fetch the same instant from both recorders and stack the results
side-by-side. The last load step:

```python
er_global = ds.elements.get_element_results(
    results_name="force",      element_type="5-ElasticBeam3d",
    model_stage="MODEL_STAGE[1]", element_ids=[1],
)
er_local = ds.elements.get_element_results(
    results_name="localForce", element_type="5-ElasticBeam3d",
    model_stage="MODEL_STAGE[1]", element_ids=[1],
)
print(er_global.df.iloc[-1])
# Px_1   -7.644e+03
# Py_1    0.000e+00
# Pz_1    2.500e+04
# Mx_1    0.000e+00
# My_1   -7.588e+06
# Mz_1    0.000e+00
# ...

print(er_local.df.iloc[-1])
# N_1     2.500e+04   ← axial = force along local x = global Z
# Vy_1    0.000e+00
# Vz_1   -7.644e+03   ← transverse shear = -global X
# T_1    -0.000e+00
# My_1    7.588e+06
# Mz_1    0.000e+00
# ...
```

The numbers line up the way the rotation matrix says they should:
applying ``R`` to local ``(N=25000, Vy=0, Vz=-7644)`` gives global
``(Px=-7644, Py=0, Pz=25000)``.

---

## 3. Rotate `localForce` and verify it matches `force`

`force` carries 12 components per step in the order
``(Px_1, Py_1, Pz_1, Mx_1, My_1, Mz_1, Px_2, Py_2, Pz_2, Mx_2, My_2, Mz_2)``:
force then moment at each of the two end nodes. `localForce` has the
same layout in the element-local frame
(``N, Vy, Vz, T, My, Mz`` per node). So the rotation is four
``(3,)``-vector rotations per row:

```python
R = ds.cdata.rotation_matrix(1)
local_df = er_local.df

# Right-multiply by R.T so each row's four 3-vectors get rotated.
rotated = local_df.values.copy()
for k in range(4):
    cols = slice(3 * k, 3 * (k + 1))
    rotated[:, cols] = local_df.values[:, cols] @ R.T

diff = np.max(np.abs(rotated - er_global.df.values))
print(f"max |R @ localForce - force| = {diff:.3e}")
# max |R @ localForce - force| = 3.725e-09     ← machine precision
```

The reason the round-trip is exact (not 1e-6) is that
`quaternion_to_rotation_matrix` normalizes the input quaternion
before building `R`. STKO writes ``0.707107`` to six digits — without
normalization the resulting matrix is off-orthogonality by ~1e-6,
which compounds to errors of a few units on 1e6-magnitude moments.

---

## 4. Vectorized: every element at once

For real models, fetch every rotation matrix in one batch and apply
them to the result tensor with `np.einsum`:

```python
ids, R = ds.cdata.rotation_matrices()        # (N,), (N, 3, 3)
print(ids.shape, R.shape)
# (11,) (11, 3, 3)
```

If you have a ``(steps, n_elements, 3)`` local-frame array of shears
(or any 3-vector quantity) you can rotate every row in one shot:

```python
# Imagine V_local has shape (n_steps, n_elements, 3) in element-local.
# Align it row-for-row with ids, then:
V_global = np.einsum("eij,sej->sei", R, V_local)
```

Same shape for moments. For the full 12-component beam output you
just do the einsum on each `(steps, n_elements, 3)` slice (`F_1`,
`M_1`, `F_2`, `M_2`).

---

## Variations

- **Going the other way (global → local).** ``R`` is orthogonal,
  so the inverse is the transpose: ``v_local = R.T @ v_global``.
  Useful when you have global section forces and want them in the
  local frame to interpret as N / V / M.
- **Section forces at integration points.** ``section.force`` is
  local-frame per IP. Apply the same per-element rotation across
  every IP slice — the orientation doesn't vary along an element.
- **Other beam-output names.** OpenSees emits ``force`` (global) and
  ``localForce`` (local) by default for elastic beams. Force-based
  fiber elements may only record ``localForce``; check
  `ds.element_results_names`.
- **Use the free function directly.** ``quaternion_to_rotation_matrix``
  is re-exported at the top level if you have quaternions from
  somewhere other than STKO:

  ```python
  from STKO_to_python import quaternion_to_rotation_matrix
  R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
  # identity
  ```

  The batched form accepts ``(N, 4)`` and returns ``(N, 3, 3)``.
