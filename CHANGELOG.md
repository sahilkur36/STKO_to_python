# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/) as
spelled out in [`CLAUDE.md`](CLAUDE.md#versioning-policy):

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features.
- **PATCH** (`v1.x.Y`) — bug fixes, docs, internal refactors with no API change.

## [Unreleased]

_Nothing pending._

---

## [1.3.0] — 2026-05-11

Adds full parsing of the `.cdata` sidecar file (every section, not just
`*SELECTION_SET`) and turns the new metadata into a user-facing
"select by STKO geometry/property name" workflow on `ElementSelector`.
Verified end-to-end against the bundled 95k-line example.

### Added

- **`CDataReader` now parses every `.cdata` section** in one pass per
  partition. New `@cached_property` accessors on the reader (and
  reachable via `dataset.cdata.X`):
  - `local_axes` — `{elem_id: ndarray([qw, qx, qy, qz])}` per-element
    rotation quaternion ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `section_offsets` — `{elem_id: ndarray([yOff, zOff])}` in
    element-local coords ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `element_info` — `{elem_id: ElementInfo}` parent geometry,
    sub-geometry type, and physical/element property names
    ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `beam_profiles` — `{profile_id: BeamProfile}` 2D cross-section
    geometry (points, triangulation, edge outline, sweep indices)
    ([#59](https://github.com/nmorabowen/STKO_to_python/pull/59))
  - `beam_profile_assignments` — `{elem_id: [(profile_id, weight), ...]}`
    element-to-profile mapping ([#59](https://github.com/nmorabowen/STKO_to_python/pull/59))
- **`ElementInfo`** and **`BeamProfile`** frozen dataclasses,
  re-exported at the top level (`STKO_to_python.ElementInfo`,
  `STKO_to_python.BeamProfile`).
- **Four new `ElementSelector` anchor primitives** resolving against
  `.cdata` `*ELEMENT_INFO` ([#60](https://github.com/nmorabowen/STKO_to_python/pull/60)):
  - `.of_geometry(name)` — STKO parent geometry name
  - `.of_physical_property(name)` — material/section property name
  - `.of_element_property(name)` — element class property name
  - `.of_sub_geom_type(t)` — `"Edge"` / `"Face"` / `"Solid"`
- **`STKO_to_python.model.cdata_format.CDataFormatPolicy`** — pure-
  functional policy class mirroring `MpcoFormatPolicy`. Owns the six
  section marker tokens plus `known_markers()`, `is_section_marker()`,
  `is_any_marker()` ([#61](https://github.com/nmorabowen/STKO_to_python/pull/61)).
- Cookbook recipe
  [`07-select-by-geometry-and-property.md`](docs/cookbook/07-select-by-geometry-and-property.md) —
  end-to-end walk-through of the new selector anchors.

### Changed

- **Selection-set id-list parsing is now wrap-width agnostic.** The
  previous parser assumed exactly 10 ids per line via `(n + 9) // 10`;
  the new `_consume_ids` helper scans forward consuming integers until
  the expected count is reached, so any wrap width parses correctly
  ([#61](https://github.com/nmorabowen/STKO_to_python/pull/61)).
- **`.cdata` parse failures fail fast.** Previously a malformed file
  silently returned `[]`, producing a partial `dataset.selection_set`
  that broke downstream queries far from the cause. Now the parser
  logs the offending file via `logger.exception` and re-raises, so
  dataset construction fails loudly at the source
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).

### Fixed

- `.cdata` files are now opened with `encoding="utf-8", errors="replace"` —
  fixes a `UnicodeDecodeError` on Windows when files contain non-ASCII
  bytes (`cp1252` was the previous default)
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Latent `UnboundLocalError` when a selection set had `NNODES=0` and
  `NELEMENTS>0` (the element parsing branch referenced an unbound
  `nodes_end_line`). Not triggered by current STKO output but real
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- `print_selection_set_names` now routes through the module logger
  (`logger.info`) instead of `print`, matching every sibling reader
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Dropped a wasteful `np.array(lines, dtype=str)` wrapper in the
  selection-set parser — it allocated a fixed-width `U`-array sized
  to the longest line for no speedup ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Removed a stale docstring on `_extract_selection_set_ids`
  referencing a `fileName` argument that the method does not take
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).

### Test coverage

- 30 new unit tests for the `.cdata` parser (selection-set, every new
  section, the width-agnostic id consumer, format policy, error paths).
- 13 new unit tests for the `ElementSelector` anchors backed by
  `*ELEMENT_INFO`.
- Real-file smoke checks against the bundled `QuadFrame_results` and
  the 95k-line `examples/New/results_nodes.mpco.cdata` sidecar.
- Total unit suite: **613 passed** (was 566 before the stack), 1
  pre-existing skip.

---

## [1.2.0] — 2026-05-09

### Added

- **`ElementSelector`** — lazy, chainable, composable element-id
  queries with spatial primitives (`within_box`, `within_distance`,
  `nearest_to`, `on_plane`, `near_line`, `centroid_in`, `where`)
  and Boolean composition (`&` / `|` / `~`).
- **`NodeSelector` + `NodeResultMask`** — node-side equivalents,
  with the same anchor / filter-op / boolean-algebra design.
- **`ResultMask`** — per-element boolean mask built from value
  conditions over a time window, applied via `er[mask]`.
- Top-level [selector + mask pipeline guide](docs/selector_and_mask_pipeline.md)
  promoted to a primary doc.
- Cookbook recipes 05 (element pipeline) and 06 (node pipeline).
- Test CI workflow, benchmark CI workflow.

### Changed

- Group B file renames: managers are now on canonical paths
  (`nodes.node_manager`, `elements.element_manager`, etc.).
  Legacy paths emit `DeprecationWarning` via PEP 562 `__getattr__`
  shims and continue to import.
- `Gauss` / `shape` modules relocated from `utilities/` to `format/`.
- `__slots__` applied to `NodalResults` and `_ResultView` for memory
  footprint (Phase 4a).

---

## [1.1.0] — 2026-04 / 2026-05

### Added

- Layered shells support (MPCO recorder + `ElementResults`).
- Per-element fixtures and demo notebooks under `examples/`.
- API stubs filled in across the doc tree.
- Spatial-query polish for `ElementResults`.

### Changed

- Examples nav and index restructuring; element-results API and
  navigation overhauled.

See the merge log for full detail:
`git log --merges v1.0.0..v1.1.0`.

---

## [1.0.0] — initial public release

Initial release. `MPCODataSet` opens MPCO HDF5 recorder output;
`NodalResults` / `ElementResults` expose result DataFrames;
`Aggregator` provides cross-element/cross-step engineering reductions
(interstory drift, envelopes, residuals, orbits); `NodalResultsPlotter`
produces publication figures. Multi-partition MP output is read
transparently.

See `git log --merges v1.0.0` for the full pre-1.0 history.

[Unreleased]: https://github.com/nmorabowen/STKO_to_python/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/nmorabowen/STKO_to_python/releases/tag/v1.0.0
