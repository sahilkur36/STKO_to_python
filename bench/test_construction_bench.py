"""Phase 5.1.d — dataset construction benchmarks.

Isolates ``MPCODataSet.__init__`` cost from the first fetch (which the
``test_bench_fetch_cold`` case in ``test_fetch_bench.py`` bundles
together). Spec §6 targets dataset construction on a 100-partition
file at ~1.5 s; our checked-in fixtures are 1 / 4 partitions so the
absolute numbers are only useful as regression guards. Comparing the
single- vs multi-partition cost exposes the per-partition fixed cost
that the spec target turns on.

Note: as of v1.5.0, ``selection_set`` and ``_selection_resolver``
parse lazily on first access via ``@cached_property``. The
construction path therefore does **not** parse the ``.cdata``
sidecar; that cost shows up under ``test_bench_dataset_first_cdata_touch``.

Run with::

    pip install -e ".[bench]"
    pytest bench/test_construction_bench.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from STKO_to_python import MPCODataSet


pytestmark = pytest.mark.bench


def test_bench_dataset_construction_single_partition(
    benchmark, elastic_frame_dir: Path
):
    """``MPCODataSet(...)`` on the single-partition elasticFrame fixture.

    4 nodes, 3 beams, 1 partition. Establishes the per-call fixed cost
    that scales with partition count on real-world models.
    """
    path_str = str(elastic_frame_dir)

    def _construct():
        return MPCODataSet(path_str, "results", verbose=False)

    ds = benchmark(_construct)
    assert ds.recorder_name == "results"
    assert len(ds.results_partitions) == 1


def test_bench_dataset_construction_multi_partition(
    benchmark, quad_frame_dir: Path
):
    """``MPCODataSet(...)`` on the multi-partition QuadFrame fixture.

    676 nodes, 700 elements (75 beams + 625 shells), multiple
    partitions. The delta against ``..._single_partition`` exposes the
    per-partition overhead (file open + index assembly) that dominates
    the spec §6 100-partition construction target.
    """
    path_str = str(quad_frame_dir)

    def _construct():
        return MPCODataSet(path_str, "results", verbose=False)

    ds = benchmark(_construct)
    assert len(ds.results_partitions) > 1


def test_bench_dataset_first_cdata_touch(benchmark, elastic_frame_dir: Path):
    """First-access cost of the lazy ``selection_set`` ``@cached_property``.

    Dataset construction is built once per round; the benchmark measures
    only the parse step that the v1.5.0 lazy refactor moved out of
    ``__init__``. Pin this so a future change that re-eagers the parse
    has a visible perf signature.
    """
    path_str = str(elastic_frame_dir)

    def _touch_cdata():
        ds = MPCODataSet(path_str, "results", verbose=False)
        # First access triggers the parse; subsequent reads are O(1).
        _ = ds.selection_set
        return ds

    benchmark(_touch_cdata)
