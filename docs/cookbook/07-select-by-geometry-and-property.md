# Select elements by STKO geometry and property name

> Pick elements directly by the *names* you gave them in STKO — the
> parent geometry, the sub-geometry type (Edge / Face / Solid), and
> the physical or element property — without writing the selection
> sets out by hand first.

The engineering question is: *I built this model in STKO, I know which
geometries and properties hold what — can I just ask for "every
element whose physical property is `BasementsWalls_Elastic`" or
"every shell on the Slab geometry"?* Before v1.3.0 the only path was
to define selection sets in STKO, write them to the `.cdata`, then
resolve them by name. From v1.3.0 the `*ELEMENT_INFO` section of the
`.cdata` file is parsed too, so STKO names are first-class selector
anchors.

The fixture used here is
`stko_results_examples/elasticFrame/QuadFrame_results` — a 5 m × 3 m
wall panel (coordinates in mm) meshed with `203-ASDShellQ4` shells
plus edge beams. Two partition files (`results.part-0.mpco`,
`results.part-1.mpco`) are merged transparently. The shells were
modelled in STKO under a `Face` sub-geometry with an element property
named `Q4`; the beams under `Edge` sub-geometries with
`elasticBeamCol`.

```python
from pathlib import Path
from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "QuadFrame_results"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

---

## 1. Inspect what STKO wrote out

Every element gets an `ElementInfo` record exposing its parent
geometry, sub-geometry type, and the two property names STKO
assigned. The dict lives on `ds.cdata.element_info`:

```python
sample_id = 26  # any beam element
ei = ds.cdata.element_info[sample_id]
print(ei)
# ElementInfo(element_id=26, geom_id=9, geom_name='Merge',
#             sub_geom_idx=1, sub_geom_type='Edge',
#             physical_property_id=1, physical_property_name='elastic',
#             element_property_id=2, element_property_name='elasticBeamCol')
```

To discover what's *available* without grepping the file:

```python
infos = ds.cdata.element_info.values()
print(sorted({ei.sub_geom_type for ei in infos}))
# ['Edge', 'Face']
print(sorted({ei.element_property_name for ei in infos}))
# ['None', 'Q4', 'elasticBeamCol']
print(sorted({ei.physical_property_name for ei in infos}))
# ['None', 'New physical property', 'elastic']
```

The `'None'` entries are rigid-link edges that STKO emits without a
property assignment — present in the mesh but not load-bearing.

---

## 2. Every shell in the wall — `of_sub_geom_type("Face")`

The shells live on `Face` sub-geometries; every other element type
(beams, rigid links) is on `Edge`. The cleanest pick is by topology:

```python
shells = ds.elements.select().of_sub_geom_type("Face")
print(shells.count(), "shell elements")
print(shells.ids().tolist()[:5], "...")
```

Compare with the equivalent property-name pick:

```python
shells_by_property = ds.elements.select().of_element_property("Q4")
assert set(shells_by_property.ids()) == set(shells.ids())
```

Both routes land on the same set, but `of_element_property` is the
form you reach for when the *property* itself is the engineering
identity (e.g. you have several wall types and want only one of
them).

---

## 3. Beams — and excluding the rigid links

`of_sub_geom_type("Edge")` would give you the rigid-link constraint
elements too. To get only the load-bearing beams, filter by the
physical property:

```python
beams = ds.elements.select().of_physical_property("elastic")
print(beams.count(), "beam elements")
```

Anchors AND-narrow within a single chain, so combining is identical
to writing the constraint twice:

```python
beams_via_anchor_pair = (
    ds.elements.select()
    .of_sub_geom_type("Edge")
    .of_element_property("elasticBeamCol")
)
assert set(beams_via_anchor_pair.ids()) == set(beams.ids())
```

---

## 4. Mixing STKO names with spatial primitives

The new anchors compose freely with the spatial filters from
[selector + mask pipeline](05-selector-and-mask-pipeline.md). For
"every shell in the lower half of the wall" — remembering that the
fixture's coordinates are in mm and the wall is ~3 m tall:

```python
lower_shells = (
    ds.elements.select()
    .of_sub_geom_type("Face")
    .centroid_in("z", hi=1500.0)     # half-height of the ~3 m wall
)
print(lower_shells.count(), "shells below z=1500 mm")
# 325 shells below z=1500 mm
```

And with boolean composition — say "every shell, *minus* the top
edge":

```python
all_shells   = ds.elements.select().of_sub_geom_type("Face")
top_edge     = (ds.elements.select()
                .of_sub_geom_type("Face")
                .centroid_in("z", lo=2800.0))   # top ~140 mm strip
inner_shells = all_shells & ~top_edge          # set-difference idiom
print(inner_shells.count(), "shells away from the top edge")
# 575 shells away from the top edge
```

The `~` negation needs the inner selector to carry an anchor (so its
universe is well-defined). `of_sub_geom_type("Face")` provides it.

---

## 5. Use the result like any other selector

Once you have ids, the rest of the pipeline is unchanged. Fetch
results, build masks, plot:

```python
PUSHOVER_STAGE = "MODEL_STAGE[2]"

er = ds.elements.get_element_results(
    results_name="section.force",
    element_type="203-ASDShellQ4",
    model_stage=PUSHOVER_STAGE,
    element_ids=lower_shells.ids().tolist(),
)
print(er.list_canonicals())
# ('bending_moment_xx', 'bending_moment_xy', ...)
```

---

## Variations

- **Multiple property names.** Build one selector per name and union
  them: `(sel.of_physical_property("Slab_Elastic") |
  sel.of_physical_property("ShearWalls_Elastic"))`.
- **Combine with selection sets.** STKO selection sets and STKO
  property names live side by side — anchor with `.from_selection(...)`
  on the left of an `&` and `.of_physical_property(...)` on the right
  for "members of *this* set whose property is *that*".
- **Discoverability.** When prototyping, dump
  `{ei.physical_property_name for ei in ds.cdata.element_info.values()}`
  to a one-liner before reaching for the API docs.
- **Other accessors on `ds.cdata`.** The `.cdata` parser also exposes
  `ds.cdata.local_axes`, `ds.cdata.section_offsets`,
  `ds.cdata.beam_profiles`, and `ds.cdata.beam_profile_assignments`.
  These are read-only metadata dicts, not yet wired into the selector,
  but you can use them directly to build custom filters via
  `.where(lambda df: ...)`.
