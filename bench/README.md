# Benchmarks

`pytest-benchmark` suite that pins the cost of the hot paths the
library is built on: HDF5 fetches, drift / envelope aggregations,
pickle round-trips, and dataset construction.

The suite has two jobs:

1. **Regression guard.** Numbers run on the small checked-in fixtures
   (`elasticFrame/results`, `elasticFrame/QuadFrame_results`) and live
   alongside the source so CI / pre-merge runs can catch >25% slowdowns
   on a given operation before they ship.
2. **Anchor for the §6 perf targets.** The
   [architecture refactor proposal](../docs/architecture-refactor-proposal.md#measured-targets)
   lists targets on a 100-partition / 10k-step realistic file. Our
   fixtures are 1 / ~4 partitions, so absolute numbers won't match the
   spec table — they establish the *per-call floor* that the spec
   targets sit on top of.

## Running

```bash
pip install -e ".[bench]"
pytest bench/ -q --benchmark-only
```

The bench suite is excluded from the default `pytest tests/` collection
via `testpaths = ["tests"]` in `pyproject.toml`. Every bench case is
marked `pytest.mark.bench` so they can be filtered explicitly.

## v1.5.0 baseline

Hardware: Windows 11 / Python 3.11.9. Reported as **median** μs (more
stable than mean across the warmup-3, min-rounds=20 settings used in
this measurement). Numbers are reproducible to ~10% on the same
machine; cross-machine comparisons are noise-dominated.

| Operation | Median | Notes |
|---|---:|---|
| `test_bench_setstate_with_unknown_keys` | 2.65 μs | tolerant `__setstate__` cost on a 4-node `NodalResults` |
| `test_bench_fetch_warm_different_component` | 6.2 μs | LRU hit, narrowed `node_ids` |
| `test_bench_fetch_warm_cache_hit` | 6.4 μs | LRU hit, identical args (returns cached object) |
| `test_bench_resolve_story_nodes` | 59.8 μs | z-tolerance clustering on 4 nodes |
| `test_bench_pickle_loads_only` | 215.6 μs | `pickle.loads` + `_build_views` on a 4-node `NodalResults` |
| `test_bench_pickle_roundtrip` | 359.3 μs | `dumps` + `loads` |
| `test_bench_orbit` | 650.9 μs | dual-component fetch + align |
| `test_bench_drift_pairwise` | 677.7 μs | one drift time-history via the engine |
| `test_bench_interstory_drift_envelope` | 2.43 ms | story clustering + per-pair drift over 2 stories |
| `test_bench_dataset_construction_single_partition` | 13.58 ms | `MPCODataSet(...)` on 1-partition fixture (lazy cdata, no parse) |
| `test_bench_dataset_first_cdata_touch` | 20.60 ms | construction **+** first `selection_set` access (parses .cdata) |
| `test_bench_dataset_construction_multi_partition` | 25.16 ms | `MPCODataSet(...)` on ~4-partition fixture |
| `test_bench_fetch_cold` | 25.93 ms | construction + first `get_nodal_results` |

### What the numbers say about v1.5.0

- **Lazy cdata parse is doing its job.** Construction alone is
  13.58 ms; touching `selection_set` afterward adds ~7 ms for the
  .cdata parse. Workflows that never touch selection sets skip that
  cost.
- **Warm fetches are essentially free.** 6 μs per cache hit is the
  cost of a dict lookup + an LRU promotion; no HDF5 traffic. Repeat
  plots / aggregations off the same result trace are bottlenecked by
  rendering / pandas, not the fetch path.
- **Pickle round-trip is dominated by the view-rebuild pass.**
  `loads_only` (215 μs) is 60% of `roundtrip` (359 μs) on tiny data
  because `dumps` is fast on small dicts. On the spec's 500 MB pickle
  the proportions flip.
- **Multi-partition construction scales sublinearly.** 4 partitions
  cost ~1.85× a single partition (25 ms vs 13.5 ms), not 4×. The
  partition pool keeps the overhead bounded; per-partition incremental
  cost is ~3 ms on this machine.

### Distance from spec §6 targets

The proposal's targets assume a 100-partition / 10k-step file. Using
the v1.5.0 per-partition incremental cost (~3 ms) as a rough
extrapolation:

- **Dataset construction (100 partitions)** — estimated
  ~300 ms on the per-partition extrapolation, vs the spec's ~1.5 s
  target. We may already meet it, but the extrapolation assumes
  per-partition cost stays linear at scale (it won't — file-system
  cache, parallel reads etc. shift the picture). Needs validation on
  a real 100-partition fixture.
- **Single-node fetch (cold)** — 25.9 ms today vs ~25 ms target.
  Comfortably under.
- **Single-node fetch (warm)** — 6 μs today vs ~5 ms target. Three
  orders of magnitude under.
- **Multi-case drift profile (10 cases, serial)** — not directly
  comparable; our `interstory_drift_envelope` covers a single case at
  2.4 ms. A 10-case loop would be ~25 ms before any per-case work.

Verifying these properly requires a larger fixture. Until then the
spec table stays an open question and these benches act as a
regression guard against the v1.5.0 baseline.

## Adding a new bench

Drop a new `test_*.py` under `bench/` and add `pytestmark =
pytest.mark.bench` at module scope. Re-run `pytest bench/ -q
--benchmark-only` and append the median row to the table above when
the numbers stabilize. Keep the bench focused — one benchmark per
behavior, no `parametrize` explosions.
