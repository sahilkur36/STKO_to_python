# Find the highly-displaced nodes in a region — node selector + mask pipeline

> Locate every node inside a chosen plan region whose absolute peak
> displacement magnitude over the seismic window exceeds a threshold,
> then plot the survivors. Demonstrates the chainable
> `NodeSelector` + `NodeResultMask` pipeline end-to-end. The shape of
> the recipe is the mirror image of
> [the element pipeline](05-selector-and-mask-pipeline.md) — the same
> two layers, the same boolean algebra, the same time-spec grammar.

The pipeline has two layers, executed in this order:

1. **`ds.nodes.select()` — pre-fetch.** Chainable, lazy queries over the
   cached node index. Decides *which nodes* to read from disk before any
   HDF5 access. Spatial primitives, selection / id anchors,
   `attached_to(...)` to bridge to elements, and `&` / `|` / `~`
   boolean composition.
2. **`nr.where(...)` — post-fetch.** Builds a per-node boolean
   `NodeResultMask` from a value condition over a time window. Compose
   masks with `&` / `|` / `~`; apply with `nr[mask]` to get a fresh
   `NodalResults` trimmed to the matched nodes.

The example below uses the
`stko_results_examples/elasticFrame/elasticFrame_mesh_displacementBased_results`
fixture (the same one cookbook 05 builds on): a planar moment frame
with 12 nodes at four z-levels (`0, 1000, 2000, 3000` mm) and 11
`64-DispBeamColumn3d` elements. `MODEL_STAGE[1]` is gravity,
`MODEL_STAGE[2]` is the pushover.

```python
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = (
    REPO_ROOT / "stko_results_examples" / "elasticFrame"
    / "elasticFrame_mesh_displacementBased_results"
)
PUSHOVER_STAGE = "MODEL_STAGE[2]"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

---

## 1. Pre-fetch: pick nodes by geometry

Build a selector, inspect the resolved id list, then hand it to the
fetch path. The selector is **lazy** — nothing happens until you call
`.ids()`, `.df()`, `.mask()`, `.count()`, or pass it through
`get_nodal_results(selector=…)`.

### 1a. The most common pick: a story floor

```python
roof = ds.nodes.select().at_level("z", value=3000.0, tol=1e-3)
print(roof)                  # NodeSelector(AtLevelOp)
print(roof.count())          # 6 (one node per beam end at z=3000 mm)
print(roof.ids().tolist())   # int64 ids
```

`at_level("z", v, tol=…)` is sugar for "every node within `tol` of
elevation `v`". Use it for support rows (`at_level("z", 0)`), story
floors, or any axis-aligned slice.

### 1b. Box, sphere, plane, line

```python
# Inside an axis-aligned box
in_box = ds.nodes.select().within_box(min=(0, -1, 1000), max=(2500, 1, 2000))

# Within a radius of a point
near_pt = ds.nodes.select().within_distance((2500, 0, 1500), radius=400.0)

# k nearest nodes to a point (rows sorted by ascending distance)
nearest8 = ds.nodes.select().nearest_to((2500, 0, 1500), k=8)

# On an arbitrary plane (point + unit normal); axis-aligned form below
plane_y = ds.nodes.select().on_plane(point=(0, 0, 0), normal=(0, 1, 0), tol=1e-6)
plane_z = ds.nodes.select().on_plane(z=1500.0, tol=1e-3)   # equivalent of at_level

# Within a tube around a line segment
along_col = ds.nodes.select().near_line((0, 0, 0), (0, 0, 3000), radius=50.0)

# One-sided coordinate range
upper_half = ds.nodes.select().coord_in("z", lo=1500.0)   # hi defaults to +∞

print(in_box.count(), near_pt.count(), nearest8.count(),
      plane_y.count(), along_col.count(), upper_half.count())
```

`sel.df()` returns the matching rows of the node-index DataFrame
(`node_id, file_id, index, x, y, z`) — useful for sanity checking.

### 1c. Custom predicate (escape hatch)

```python
# Anything not covered by a primitive
upper_left = ds.nodes.select().where(
    lambda df: (df["x"].to_numpy() < 1500) & (df["z"].to_numpy() > 1500)
)
print(upper_left.count())
```

The callable receives the node-index DataFrame and must return a
boolean numpy array of equal length.

---

## 2. Anchoring the universe

By default a `NodeSelector` has no anchor — its universe is *every node
in the model*, which is fine for spatial filtering. **Negation (`~`)**
needs an explicit universe; otherwise `~sel` would silently complement
against the whole model. There are two anchors:

### 2a. By selection set (name or id)

```python
# Named selection set (case-insensitive)
roof_set = ds.nodes.select().from_selection("Roof")

# Or by selection-set id
roof_set = ds.nodes.select().from_selection(2)

# Multiple sets unioned
combined = ds.nodes.select().from_selection(["Roof", "Mezzanine"])
```

### 2b. By explicit ids

```python
sel = ds.nodes.select().with_ids([10, 42, 99, 1234])
```

`from_selection` and `with_ids` can be combined with the spatial
primitives — anchor first, then narrow:

```python
# Roof nodes inside a plan box
roof_in_box = (
    ds.nodes.select()
    .from_selection("Roof")
    .within_box(min=(1000, -1, 0), max=(4000, 1, 4000))
)
```

---

## 3. Bridge to elements: `attached_to(...)`

`attached_to(...)` returns "every node that participates in the
connectivity of these elements" — it closes the loop between node and
element selectors. Pass either explicit element ids or a live
`ElementSelector`:

```python
# Every node attached to the lower-half beams (z ∈ [0, 1500])
lower_beams = (
    ds.elements.select()
    .of_type("DispBeamColumn3d")
    .centroid_in("z", lo=0.0, hi=1500.0)
)
lower_nodes = ds.nodes.select().attached_to(element_selector=lower_beams)

print(f"{lower_beams.count()} beams → {lower_nodes.count()} unique nodes")
```

In a richer model with mixed element classes (shells, solids, ties)
the same call works without changes: pass any `ElementSelector`,
including a boolean combination of element selectors:

```python
mixed = (
    ds.elements.select().of_type("ASDShellQ4").centroid_in("z", lo=0, hi=2)
    | ds.elements.select().of_type("DispBeamColumn3d").centroid_in("z", lo=0, hi=2)
)
nodes_in_diaphragm = ds.nodes.select().attached_to(element_selector=mixed)
```

Or with explicit ids:

```python
nodes_of_two = ds.nodes.select().attached_to(element_ids=[101, 102])
```

This is the most powerful primitive when you want to do "everything in
the diaphragm at level X": pick the elements that make up the
diaphragm, then derive their node set. The reverse-style operation
(elements via nodes) is already covered by `ElementSelector.where(fn)`.

---

## 4. Boolean composition

The selector algebra is the same as the element side:

```python
roof   = ds.nodes.select().from_selection("Roof")
ground = ds.nodes.select().from_selection("Base")

both    = roof & ground            # intersection (likely empty here)
either  = roof | ground            # union
just_roof_corner = roof & ds.nodes.select().within_box(
    min=(0, -1, 0), max=(0, 1, 4000)
)

# Negation requires an anchor on the *leaf* selectors
roof_inner = roof & ~ds.nodes.select().from_selection("Roof").within_box(
    min=(-1e-6, -1, -1e-6), max=(0,  1, 4000)
)
```

The `~` rule is strict: a leaf must declare its universe via
`.from_selection(...)` or `.with_ids(...)` for negation to be defined.
An unanchored `~sel` raises a `ValueError` rather than silently
negating against every node in the model.

`(a & b)`, `(a | b)`, `(~a)` are themselves selectors — they expose
`.ids()`, `.count()`, `.df()`, `.mask()`. They cannot be re-anchored
or take new spatial primitives (would be ambiguous as to which leaf
gets the new op); apply primitives to the leaves, not the combinator.

---

## 5. Fetch results, scoped by the selector

`get_nodal_results(...)` accepts a selector directly — its resolved
ids are unioned with anything else you pass through `node_ids` /
`selection_set_*`:

```python
sel = ds.nodes.select().at_level("z", 3000.0, tol=1e-3)

nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selector=sel,
    model_stage=PUSHOVER_STAGE,
)
print(nr)
# NodalResults(name=..., results=('DISPLACEMENT',), components=('1','2','3'), ...)
print(nr.df.shape)        # (n_nodes * n_steps, 3)
print(nr.info.nodes_ids)  # tuple of node ids in the result
```

You can mix selector-driven and explicit-id selection on the same call;
the resolved id list is the union:

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selector=sel,
    node_ids=[10, 42],                 # add two more
    selection_set_name="ControlPoints",  # and union a named set
    model_stage=PUSHOVER_STAGE,
)
```

---

## 6. Build a result mask

`nr.where(...)` opens a chain that ends in a `NodeResultMask`. There
are two shapes for picking the value: **single component** or **vector
magnitude**.

### 6a. Single component → reduce → threshold

```python
# Roof X-displacement, absolute peak over the pushover, above 0.1 mm
mask = (
    nr.where(time=(0.0, 5.0))                # default time window
      .component("DISPLACEMENT", 1)          # (result_name, component)
      .abs_peak()                            # |max| over the window
      .gt(0.1)
)
print(mask)              # NodeResultMask(n_true=…, n_total=…)
print(mask.ids())        # node ids that pass
hot = nr[mask]           # fresh NodalResults, trimmed to matched ids
```

The available reductions on a component:

| Reduction | Meaning |
|---|---|
| `at_step(s)` | scalar at one step (int step index) |
| `at_time(t)` | scalar at the step nearest to time `t` |
| `peak(time=…)` | signed max over the window |
| `trough(time=…)` | signed min over the window |
| `abs_peak(time=…)` | max of `|·|` over the window |
| `mean(time=…)` | arithmetic mean over the window |
| `residual(time=…)` | last step in the window |
| `over_threshold(v, time=…)` | fraction of steps where value > `v` |

The `time=` argument on each reduction overrides the chain default.
The `time` grammar accepts:

```python
nr.where(time=None)                          # all steps (the default)
nr.where(time=42)                            # int step index
nr.where(time=2.5)                           # nearest step to time 2.5
nr.where(time=slice(0.0, 10.0))              # half-open time range
nr.where(time=(0.0, 10.0))                   # tuple form, same meaning
nr.where(time=[0, 5, 10])                    # explicit step indices
nr.where(time=[0.0, 1.0, 5.0])               # nearest step per time
```

### 6b. Vector magnitude

For 3-DOF nodal vectors (`DISPLACEMENT`, `VELOCITY`, `ACCELERATION`),
`magnitude(...)` reduces to a per-`(node, step)` scalar via
`sqrt(sum(comp_i**2))` *before* the time-axis reduction. With
`components=()` (default) every component for the result is used:

```python
# |U| peak over the full record, > 0.1 mm
hot_mag = nr.where().magnitude("DISPLACEMENT").peak().gt(0.1)
print(hot_mag.count())

# Planar magnitude (X-Y only)
hot_planar = (
    nr.where()
    .magnitude("DISPLACEMENT", components=(1, 2))
    .abs_peak()
    .ge(0.1)
)
```

### 6c. Comparators

The reductions return a `_ScalarPerNode` object — call any of:

```python
.gt(v)                  # >
.lt(v)                  # <
.ge(v)                  # >=
.le(v)                  # <=
.between(lo, hi, inclusive=True)
.outside(lo, hi, inclusive=False)
.eq(v, atol=0.0)        # equal (with optional absolute tolerance)
.near(v, atol=…)        # |x - v| <= atol
```

Each returns a `NodeResultMask` ready for composition or `nr[mask]`.

### 6d. Predicate escape hatch

The full-`(node_id, step)` index is exposed if you need an arbitrary
shape of comparison. The callable is reduced via `any` per node:

```python
# Any step where Ux beats Uy — for orbit-style screening
mask = nr.where().predicate(
    lambda df: df[("DISPLACEMENT", 1)].abs() > df[("DISPLACEMENT", 2)].abs()
)
print(mask.count())
```

If the callable returns a per-node bool array (length =
`len(nr.info.nodes_ids)`), it's used as-is; if it returns a per-row
array (length = `len(nr.df)`), it's `groupby("node_id").any()`-reduced.

---

## 7. Compose: selector AND mask in one shot

Spatial filter (Layer A) and value filter (Layer B) compose naturally
because they live in different objects: the selector produces ids that
go into the fetch; the mask produces a boolean over the fetched ids
that goes into `nr[...]`.

```python
# Story-floor X-displacement above 0.1 mm AND |U| above 0.12 mm
story = ds.nodes.select().at_level("z", 3000.0, tol=1e-3)

nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selector=story,
    model_stage=PUSHOVER_STAGE,
)

m_x   = nr.where().component("DISPLACEMENT", 1).abs_peak().gt(0.10)
m_mag = nr.where().magnitude("DISPLACEMENT").peak().gt(0.12)

both    = m_x & m_mag
either  = m_x | m_mag
not_x   = ~m_x

hot = nr[both]
print(f"{hot.info.nodes_ids} pass both criteria")
```

`hot` is a fully-formed `NodalResults` — pickle-able, plottable, and
every aggregation (`drift`, `interstory_drift_envelope`, `orbit`, …)
works on it.

---

## 8. Tune thresholds from the data

Every reduction is exposed as a `pd.Series` so you can pick a
threshold informed by the distribution rather than guessing:

```python
# All roof nodes; what does the |U| peak distribution look like?
roof = ds.nodes.select().at_level("z", 3000.0, tol=1e-3)
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT", selector=roof, model_stage=PUSHOVER_STAGE
)

peaks = nr.where().magnitude("DISPLACEMENT").peak().values()
print(peaks.describe())          # min/max/mean/std per node
threshold = float(peaks.quantile(0.75))   # top 25 %

mask = nr.where().magnitude("DISPLACEMENT").peak().gt(threshold)
hot = nr[mask]
```

`over_threshold(v)` returns the *fraction* of steps in the window where
`value > v`, so chain a comparator to find nodes that *spend* a
significant share of the window above some level — useful for shake-
table or buffeting analyses where the all-time peak is less indicative
than sustained excursions:

```python
mask = (
    nr.where()
      .component("DISPLACEMENT", 1)
      .over_threshold(0.05)     # fraction of steps with Ux > 0.05 mm
      .gt(0.25)                 # at least 25 % of the window
)
```

---

## 9. Plot the survivors

Plotting helpers carry over verbatim — `hot` behaves exactly like the
original `nr`, just with fewer rows:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
for nid in hot.info.nodes_ids:
    ts = hot.fetch("DISPLACEMENT", component=1, node_ids=[nid])
    ax.plot(hot.time, ts.to_numpy(), lw=0.7, label=str(nid))
ax.axhline(0.1, color="k", lw=0.7, ls="--")
ax.set_xlabel("time (s)")
ax.set_ylabel("$U_x$ (mm)")
ax.set_title(f"{len(hot.info.nodes_ids)} roof nodes above the threshold")
ax.legend(fontsize="x-small", ncol=2)
plt.show()
```

---

## 10. Complete worked example

The recipe end-to-end on `QuadFrame_results`:

```python
from pathlib import Path
import numpy as np
from STKO_to_python import MPCODataSet

ds = MPCODataSet(
    str(Path("stko_results_examples/elasticFrame"
             "/elasticFrame_mesh_displacementBased_results")),
    "results",
)

# --- Layer A: pre-fetch selection -------------------------------------
# Roof nodes inside a central plan band (skip the outermost columns).
roof_band = (
    ds.nodes.select()
      .at_level("z", 3000.0, tol=1e-3)
      .coord_in("x", lo=500.0, hi=4500.0)
)
print(f"{roof_band.count()} candidate nodes")

# --- Read just those nodes -------------------------------------------
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selector=roof_band,
    model_stage="MODEL_STAGE[2]",
)

# --- Layer B: value-driven trimming ----------------------------------
# Adaptive threshold: top 25 % of the |U| peak distribution
peaks = nr.where().magnitude("DISPLACEMENT").peak().values()
threshold = float(peaks.quantile(0.75))

# AND together: peak |U| above the adaptive threshold AND positive Ux peak
mask = (
    nr.where().magnitude("DISPLACEMENT").peak().gt(threshold)
    & nr.where().component("DISPLACEMENT", 1).peak().gt(0.0)
)

hot = nr[mask]
print(f"{len(hot.info.nodes_ids)} nodes match")

# --- Persist the trimmed result --------------------------------------
hot.save_pickle("hot_roof_nodes.pkl.gz")
```

---

## Cheat sheet

| Goal | One-liner |
|---|---|
| All nodes in the model | `ds.nodes.select().ids()` (no anchor needed) |
| Story floor at `z` | `ds.nodes.select().at_level("z", value=3000.0).ids()` |
| In a 3-D AABB | `ds.nodes.select().within_box(min=(0,0,0), max=(10,10,10)).ids()` |
| Within radius of a point | `ds.nodes.select().within_distance((5,5,5), r=2.0).ids()` |
| k nearest to a point | `ds.nodes.select().nearest_to((5,5,5), k=10).ids()` |
| On a plane | `ds.nodes.select().on_plane(z=2.5).ids()` (or `point=, normal=`) |
| Inside a slab | `ds.nodes.select().coord_in("z", lo=2.0, hi=4.0).ids()` |
| Tube around a line | `ds.nodes.select().near_line((0,0,0),(0,0,3),radius=0.1).ids()` |
| In named selection set | `ds.nodes.select().from_selection("Roof").ids()` |
| Nodes attached to elements | `ds.nodes.select().attached_to(element_selector=esel).ids()` |
| Custom predicate | `ds.nodes.select().where(lambda df: df["x"] > 5).ids()` |
| Combine | `(a & b).ids()`, `(a | b).ids()`, `(~a).ids()` |
| Threshold abs-peak component | `nr.where().component(rn, c).abs_peak().gt(v)` |
| Threshold |vector| peak | `nr.where().magnitude(rn).peak().gt(v)` |
| Planar magnitude | `nr.where().magnitude(rn, components=(1,2)).peak().gt(v)` |
| At a step | `nr.where().component(rn, c).at_step(s).between(lo, hi)` |
| Over a time window | `nr.where(time=(t0,t1)).component(rn, c).peak().gt(v)` |
| Sustained excursion | `nr.where().component(rn, c).over_threshold(v).gt(0.25)` |
| Composing masks | `m1 & m2`, `m1 | m2`, `~m1` |
| Apply | `nr[mask]` → fresh `NodalResults` |

For the full API reference and the time-spec grammar see
[NodalResults — Node selectors](../api/nodal-results.md#node-selectors-pre-fetch)
and [NodalResults — Result masks](../api/nodal-results.md#result-masks-post-fetch).
