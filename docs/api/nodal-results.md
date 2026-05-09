# NodalResults

Self-contained view over nodal result data. Returned by
`ds.nodes.get_nodal_results()`. Holds a reference to a
`NodalResultsQueryEngine` and pre-resolved IDs. Every engineering
aggregation (`drift`, `interstory_drift_envelope`, `orbit`, …) is a
forwarder to [AggregationEngine](aggregation-engine.md).

Pickle-stable: saves and reloads without the original HDF5 files.

For the comprehensive usage guide (with full worked examples) see
[Usage guides → NodalResults](../NodalResults.md).

---

## Construction

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],           # explicit IDs, or…
    # selection_set_name="roof",      # …named selection set
    # selection_set_id=2,             # …selection set by index
)
```

---

## Key attributes

| Attribute | Type | Description |
|---|---|---|
| `nr.df` | `pd.DataFrame` | MultiIndex `(node_id, step)` × component columns |
| `nr.time` | `np.ndarray \| dict` | Time per step (dict for multi-stage) |
| `nr.node_ids` | `tuple[int, ...]` | Node IDs in this result |
| `nr.n_steps` | `int` | Number of recorded steps |

---

## Introspection

```python
nr.list_results()                     # tuple of result names, e.g. ('DISPLACEMENT',)
nr.list_components('DISPLACEMENT')    # tuple of component labels, e.g. ('1', '2', '3')
```

---

## Fetching data

```python
# Full result as DataFrame
df = nr.fetch(result_name="DISPLACEMENT")

# Single component
s = nr.fetch(result_name="DISPLACEMENT", component=1)

# Filtered by node IDs
s = nr.fetch(result_name="DISPLACEMENT", component=1, node_ids=[1, 4])

# Dynamic attribute style (equivalent to fetch)
view = nr.DISPLACEMENT[1]           # all nodes, component 1
view = nr.DISPLACEMENT[1, [1, 4]]   # nodes [1, 4], component 1
```

---

## Engineering aggregations

All aggregations forward to [AggregationEngine](aggregation-engine.md).

### Drift

```python
ts = nr.drift(top=4, bottom=1, component=1)
# pd.Series indexed by step — relative displacement u_top - u_bottom
```

### Interstory drift envelope

```python
env = nr.interstory_drift_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,          # Z-coordinate tolerance for story pairing
)
# DataFrame with columns: z_lower, z_upper, lower_node, upper_node,
#                         dz, max_drift, min_drift, max_abs_drift
```

### Residual drift

```python
resid = nr.residual_drift(
    top=4, bottom=1, component=1,
    tail=3,     # number of final steps to average
    agg="mean", # "mean" | "last"
)
# float — mean (or last) drift value at the end of the record
```

### Interstory drift envelope profile

```python
pd_env = nr.interstory_drift_envelope_pd(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,
)
# Wide DataFrame — one column per story pair, rows are time steps
```

### Residual interstory drift profile

```python
profile = nr.residual_interstory_drift_profile(
    component=1,
    node_ids=[1, 2, 3, 4],
    tail=3,
    agg="mean",
)
```

### Story PGA envelope

```python
pga = nr.story_pga_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    dz_tol=1e-3,
    to_g=True,
    g_value=9810,
)
```

### Roof torsion

```python
torsion_ts, ratio_ts = nr.roof_torsion(
    node_a_ids=[10, 11],
    node_b_ids=[20, 21],
    component=1,
    dz_tol=1e-3,
)
```

### Base rocking

```python
theta = nr.base_rocking(
    component=3,          # vertical DOF
    node_ids=[1, 2, 3],
    dz_tol=1e-3,
)
```

### ASCE torsional irregularity

```python
result = nr.asce_torsional_irregularity(
    component=1,
    side_a_top=(0.0, 0.0, 6.0),
    side_a_bottom=(0.0, 0.0, 3.0),
    side_b_top=(20.0, 0.0, 6.0),
    side_b_bottom=(20.0, 0.0, 3.0),
)
# dict with keys: ratio, delta_a, delta_b, avg_drift, irregular
```

### Orbit

```python
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
# Two Series — displacement trajectory for a single node
```

### Residual drift envelope

```python
env = nr.residual_drift_envelope(
    component=1,
    node_ids=[1, 2, 3, 4],
    tail=3,
    agg="mean",
)
```

---

## Node selectors (pre-fetch)

Build chainable, composable node-id queries against the cached node
index — no HDF5 reads required. Pass the resolved selector into
`get_nodal_results(selector=...)` to fetch only what you need.

```python
sel = (ds.nodes.select()
       .from_selection("Roof")                   # universe anchor (optional)
       .within_box(min=(0, 0, 0), max=(10, 10, 30))
       .nearest_to((5, 5, 30), k=8))

ids = sel.ids()          # np.ndarray[int64]
df  = sel.df()           # node-index rows (node_id, file_id, index, x, y, z)
mask = sel.mask()        # bool Series indexed by node_id over the universe
n   = sel.count()        # int

nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selector=sel,
    model_stage="MODEL_STAGE[2]",
)
```

**Anchors** — set the universe used for negation. Without an anchor the
universe is "all nodes" (fine for spatial filtering and unions; `~`
raises `ValueError`).

| Anchor | Purpose |
|---|---|
| `.from_selection(name_or_id)` | one or more selection sets (case-insensitive on names) |
| `.with_ids(ids)` | explicit node IDs |

**Spatial primitives** — every primitive narrows AND-style in chain
order, against the actual node coordinates:

| Primitive | Notes |
|---|---|
| `.within_box(min, max)` | axis-aligned bounding box |
| `.within_distance(point, radius)` | Euclidean distance |
| `.nearest_to(point, k=1)` | k-NN; result rows sorted by ascending distance |
| `.on_plane(z=…)` / `.on_plane(point=, normal=)` | node coordinates within `tol` of the plane |
| `.near_line(p0, p1, radius)` | distance to line segment |
| `.coord_in(axis, lo=, hi=)` | one-sided or two-sided range on `x`, `y`, or `z` |
| `.at_level(axis="z", value=, tol=1e-6)` | story-floor sugar for an axis-aligned plane |
| `.attached_to(element_ids=… \| element_selector=…)` | every node referenced by the element set |
| `.where(fn)` | predicate escape hatch — `fn(df) -> bool_mask` |

**Bridge to elements** — `attached_to` accepts either explicit element
ids or a live `ElementSelector`, so you can build a node set from a
geometric-element pick without manual id wrangling:

```python
shells = (ds.elements.select()
          .of_type("ASDShellQ4")
          .centroid_in("z", lo=0, hi=3))
diaphragm = ds.nodes.select().attached_to(element_selector=shells)
```

**Boolean composition** — combine selectors:

```python
a = ds.nodes.select().from_selection("Roof").coord_in("x", lo=0, hi=10)
b = ds.nodes.select().at_level("z", 30.0)

(a & b).ids()       # intersection
(a | b).ids()       # union
(~a).ids()          # complement WITHIN a's anchored universe
```

Negation requires an anchor (`from_selection` / `with_ids`) on every
leaf — otherwise the call raises. The combinator's universe is the
intersection of its leaves' universes for `&`, the union for `|`.

---

## Result masks (post-fetch)

`nr.where(...)` builds a per-node boolean mask from a value condition.
Combine with `& / | / ~` and apply via `nr[mask]` to get a fresh
trimmed `NodalResults`.

```python
# Single component
mask = (nr.where(time=(0.0, 10.0))             # default time window
        .component("DISPLACEMENT", 1)           # (result_name, component)
        .abs_peak()                             # reduction over the window
        .gt(0.05))                              # comparator → NodeResultMask

# Vector magnitude (the typical 3-DOF case)
mask = nr.where().magnitude("DISPLACEMENT").peak().gt(0.05)

# Planar magnitude
mask = (nr.where()
        .magnitude("DISPLACEMENT", components=(1, 2))
        .abs_peak()
        .gt(0.05))

hot = nr[mask]              # fresh NodalResults with only matched ids
ids = mask.ids()            # int64 array
n   = mask.count()          # int
```

**Reductions over time**

| Reduction | Returns one scalar per node |
|---|---|
| `at_step(s)` | value at exactly step `s` |
| `at_time(t)` | value at the step nearest to time `t` |
| `peak(time=...)` | signed maximum over the window |
| `trough(time=...)` | signed minimum over the window |
| `abs_peak(time=...)` | maximum of `|·|` over the window |
| `mean(time=...)` | mean over the window |
| `residual(time=...)` | last step in the window |
| `over_threshold(v, time=...)` | fraction of steps above `v` (chain a comparator) |

**Comparators** — `gt`, `lt`, `ge`, `le`, `between(lo, hi, inclusive=True)`,
`outside(lo, hi)`, `eq(v, atol=0)`, `near(v, atol=...)`.

**Time-spec grammar** — the `time=` argument on `nr.where()` and on
every reduction accepts:

| Spec | Meaning |
|---|---|
| `None` | all steps in `nr.time` |
| `int` | one step index (negative wraps) |
| `float` | step nearest to that time value |
| `slice(t0, t1)` | half-open *time* range `t0 ≤ time < t1` |
| `(t0, t1)` tuple | same as the slice form |
| `list[int]` / `np.ndarray[int]` | explicit step indices |
| `list[float]` / `np.ndarray[float]` | nearest step for each |

The chain inherits the default window from `nr.where(time=...)`; any
reduction may override with its own `time=` argument.

**Composition**

```python
m1 = nr.where().component("DISPLACEMENT", 1).abs_peak().gt(0.05)
m2 = nr.where().magnitude("DISPLACEMENT").peak().gt(0.06)

mask = m1 & m2          # both conditions
mask = m1 | m2          # either condition
mask = ~m1              # complement (vs all nodes in `nr`)
```

Masks must come from the same `NodalResults` instance — combining
masks across instances raises `ValueError`.

**Predicate escape hatch**

```python
# Per-row mask — full (node_id, step) index, reduced via `any` per node
mask = nr.where().predicate(
    lambda df: df[("DISPLACEMENT", 1)].abs() > df[("DISPLACEMENT", 2)].abs()
)

# Per-node mask — length == len(nr.info.nodes_ids), used directly
mask = nr.where().predicate(
    lambda df: np.array([True, False, True])
)
```

For the full worked example see
[Cookbook 06: node selector + mask pipeline](../cookbook/06-node-selector-and-mask-pipeline.md).

---

## Plotting

```python
ax, meta = nr.plot.xy(
    y_results_name="DISPLACEMENT",
    y_direction=1,
    y_operation="Max",
    x_results_name="TIME",
)

fig, meta = nr.plot.plot_TH(
    result_name="DISPLACEMENT",
    component=1,
    node_ids=[1, 2, 3, 4],
    split_subplots=True,
)
```

See [Plotting](plotting.md) for the full parameter reference.

---

## Pickle serialization

```python
nr.save_pickle("nr.pkl")
nr.save_pickle("nr.pkl.gz")          # compressed

from STKO_to_python.results.nodal_results_dataclass import NodalResults
nr = NodalResults.load_pickle("nr.pkl")
```

---

## API reference

::: STKO_to_python.results.nodal_results_dataclass.NodalResults

## The dynamic-view proxy

`NodalResults` builds an internal `_ResultView` per result name so
you can write `nr.ACCELERATION[1, [14, 25]]` instead of
`nr.fetch(result_name="ACCELERATION", component=1, node_ids=[14, 25])`.

::: STKO_to_python.results.nodal_results_dataclass._ResultView
    options:
      filters: []
      show_root_heading: true

---

## NodeSelector (selector reference)

::: STKO_to_python.nodes.selector.NodeSelector
    options:
      show_root_heading: true
      members_order: source

---

## NodeResultMask (mask reference)

::: STKO_to_python.nodes.result_mask.NodeResultMask
    options:
      show_root_heading: true
      members_order: source

::: STKO_to_python.nodes.result_mask._ResultQuery
    options:
      show_root_heading: true
      members_order: source
      filters: []

::: STKO_to_python.nodes.result_mask._ComponentQuery
    options:
      show_root_heading: true
      members_order: source
      filters: []

::: STKO_to_python.nodes.result_mask._MagnitudeQuery
    options:
      show_root_heading: true
      members_order: source
      filters: []

::: STKO_to_python.nodes.result_mask._ScalarPerNode
    options:
      show_root_heading: true
      members_order: source
      filters: []
