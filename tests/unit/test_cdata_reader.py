"""Unit tests for the CDataReader text parser.

The parser is exercised directly against synthetic .cdata files on disk;
``MPCODataSet`` is bypassed because the reader only needs
``dataset.cdata_partitions``.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from STKO_to_python.model.cdata_reader import CDataReader


def _ids_block(ids: list[int]) -> str:
    """Render an id list as the cdata wraps it: ten ids per line."""
    chunks = [
        " ".join(str(x) for x in ids[start : start + 10])
        for start in range(0, len(ids), 10)
    ]
    body = "\n".join(chunks)
    return body + "\n" if ids else ""


def _make_reader(tmp_path, *cdata_texts: str) -> CDataReader:
    partitions = {}
    for idx, text in enumerate(cdata_texts):
        path = tmp_path / f"results.part-{idx}.mpco.cdata"
        path.write_text(text, encoding="utf-8")
        partitions[idx] = str(path)
    return CDataReader(SimpleNamespace(cdata_partitions=partitions))


def test_single_partition_simple_set(tmp_path) -> None:
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "12 ControlPoint\n"
        "1\n"
        "0\n"
        "624\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert set(sets.keys()) == {1}
    assert sets[1]["SET_NAME"] == "ControlPoint"
    assert sets[1]["NODES"] == [624]
    assert sets[1]["ELEMENTS"] == []


def test_wrapped_multiline_node_list(tmp_path) -> None:
    """25 nodes wrap to 3 lines (10/10/5) — pins the (n+9)//10 math."""
    nodes = list(range(1, 26))
    text = (
        "*SELECTION_SET\n"
        "7\n"
        "10 Many_Nodes\n"
        f"{len(nodes)}\n"
        "0\n"
        f"{_ids_block(nodes)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[7]["NODES"] == nodes


def test_nodes_and_elements(tmp_path) -> None:
    nodes = [10, 20, 30]
    elements = list(range(100, 115))  # 15 elements -> 2 lines
    text = (
        "*SELECTION_SET\n"
        "9\n"
        "4 Both\n"
        f"{len(nodes)}\n"
        f"{len(elements)}\n"
        f"{_ids_block(nodes)}"
        f"{_ids_block(elements)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[9]["NODES"] == nodes
    assert sets[9]["ELEMENTS"] == elements


def test_zero_nodes_zero_elements(tmp_path) -> None:
    """A set with no members keeps its name (matches real AbsorbingBoundary)."""
    text = (
        "*SELECTION_SET\n"
        "21\n"
        "17 AbsorbingBoundary\n"
        "0\n"
        "0\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[21]["SET_NAME"] == "AbsorbingBoundary"
    assert sets[21]["NODES"] == []
    assert sets[21]["ELEMENTS"] == []


def test_zero_nodes_with_elements(tmp_path) -> None:
    """Regression: NNODES=0 + NELEMENTS>0 previously referenced an
    unbound ``nodes_end_line`` and raised UnboundLocalError.
    """
    elements = [100, 200, 300]
    text = (
        "*SELECTION_SET\n"
        "11\n"
        "8 ElemOnly\n"
        "0\n"
        f"{len(elements)}\n"
        f"{_ids_block(elements)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[11]["NODES"] == []
    assert sets[11]["ELEMENTS"] == elements


def test_no_selection_set_section(tmp_path) -> None:
    """A .cdata file without any *SELECTION_SET (matches elasticFrame example)
    yields an empty dict, not an error.
    """
    text = (
        "#Begin LOCAL AXES definitions.\n"
        "*LOCAL_AXES\n"
        "1 0 0.707107 0 0.707107\n"
    )
    reader = _make_reader(tmp_path, text)
    assert reader._extract_selection_set_ids() == {}


def test_multi_partition_aggregation(tmp_path) -> None:
    """Same SET_ID across partitions merges members and dedupes."""
    part0 = (
        "*SELECTION_SET\n"
        "1\n"
        "4 Roof\n"
        "3\n"
        "0\n"
        "1 2 3\n"
    )
    part1 = (
        "*SELECTION_SET\n"
        "1\n"
        "4 Roof\n"
        "2\n"
        "0\n"
        "3 4\n"  # 3 is a duplicate across partitions
    )
    reader = _make_reader(tmp_path, part0, part1)
    sets = reader._extract_selection_set_ids()
    assert sets[1]["NODES"] == [1, 2, 3, 4]
    assert sets[1]["SET_NAME"] == "Roof"


def test_filter_by_selection_set_ids(tmp_path) -> None:
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        "1\n"
        "0\n"
        "100\n"
        "*SELECTION_SET\n"
        "2\n"
        "1 B\n"
        "1\n"
        "0\n"
        "200\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids(selection_set_ids=[2])
    assert set(sets.keys()) == {2}


def test_malformed_file_raises_not_silent(tmp_path) -> None:
    """Truncated SELECTION_SET block bubbles up instead of silently
    returning {}; otherwise downstream queries fail far from the cause.
    """
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        # missing NNODES, NELEMENTS, and data
    )
    reader = _make_reader(tmp_path, text)
    with pytest.raises(Exception):
        reader._extract_selection_set_ids()


def test_parse_errors_logged_not_printed(tmp_path, caplog) -> None:
    """Error path uses ``logger.exception``, not ``print``."""
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        # truncated
    )
    reader = _make_reader(tmp_path, text)
    with caplog.at_level(logging.ERROR, logger="STKO_to_python.model.cdata_reader"):
        with pytest.raises(Exception):
            reader._extract_selection_set_ids()
    assert any(
        rec.name == "STKO_to_python.model.cdata_reader" and rec.levelno >= logging.ERROR
        for rec in caplog.records
    )
