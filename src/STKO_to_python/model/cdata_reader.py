

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import numpy as np


if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

logger = logging.getLogger(__name__)


SelectionSetIdsArg = Union[int, "np.integer", list, None]


class CDataReader:
    """Canonical Layer 3 reader for MPCO ``.cdata`` sidecar files.

    The legacy name ``CData`` is preserved as an alias at the bottom of
    this module; new code should prefer ``CDataReader``.
    """

    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset

    def _extract_selection_set_ids_for_file(
        self,
        file_path: str,
        selection_set_ids: SelectionSetIdsArg = None,
    ) -> list[dict]:
        """Parse all ``*SELECTION_SET`` blocks from a single ``.cdata`` file.

        Args:
            file_path: Path to the ``.cdata`` file.
            selection_set_ids: Optional ID filter. If ``None``, every set is
                parsed.

        Returns:
            List of dicts, one per selection set, each with keys
            ``SET_ID``, ``SET_NAME``, ``NODES`` (numpy ``int`` array),
            ``ELEMENTS`` (numpy ``int`` array). ``NODES``/``ELEMENTS``
            keys are absent when the count is zero.

        Raises:
            OSError: If the file cannot be opened.
            ValueError, IndexError: If the file is malformed. The
                original traceback is logged with file context.
        """
        if isinstance(selection_set_ids, (int, np.integer)):
            selection_set_ids = [int(selection_set_ids)]

        if selection_set_ids is not None and not isinstance(selection_set_ids, list):
            raise ValueError("CData Error: selection_set_ids must be a list of integers or None.")

        selection_sets: list[dict] = []

        try:
            # errors="replace" guards against non-UTF-8 bytes in the file
            # (e.g. comments emitted by a legacy STKO build) without
            # silently dropping legitimate UTF-8 selection-set names.
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if line.strip() != "*SELECTION_SET":
                    continue

                set_id = int(lines[i + 1].strip())
                if selection_set_ids is not None and set_id not in selection_set_ids:
                    continue

                raw_set_name = lines[i + 2].strip()
                name_length = int(raw_set_name.split()[0])
                offset = len(str(name_length)) + 1
                set_name = raw_set_name[offset : offset + name_length]

                number_of_nodes = int(lines[i + 3].strip())
                number_of_elements = int(lines[i + 4].strip())

                selection_set: dict = {"SET_ID": set_id, "SET_NAME": set_name}

                # Elements come right after the (possibly empty) nodes
                # block. Initialize unconditionally so NNODES=0 with
                # NELEMENTS>0 doesn't reference an unbound name.
                elements_start_line = i + 5

                if number_of_nodes > 0:
                    nodes_end_line = elements_start_line + (number_of_nodes + 9) // 10
                    node_lines = lines[elements_start_line:nodes_end_line]
                    selection_set["NODES"] = np.fromstring(
                        " ".join(node_lines).strip(), sep=" ", dtype=int
                    )
                    elements_start_line = nodes_end_line

                if number_of_elements > 0:
                    elements_end_line = elements_start_line + (number_of_elements + 9) // 10
                    element_lines = lines[elements_start_line:elements_end_line]
                    selection_set["ELEMENTS"] = np.fromstring(
                        " ".join(element_lines).strip(), sep=" ", dtype=int
                    )

                selection_sets.append(selection_set)

        except Exception:
            # Re-raise so the dataset fails loudly at construction;
            # silently returning [] used to mask broken cdata files
            # until a downstream selection-set query crashed.
            logger.exception("CData parse error in %s", file_path)
            raise

        return selection_sets

    def _extract_selection_set_ids(
        self,
        selection_set_ids: SelectionSetIdsArg = None,
    ) -> dict[int, dict]:
        """Aggregate selection sets across every ``.cdata`` partition.

        Args:
            selection_set_ids: Optional ID filter. If ``None``, every set
                from every partition is aggregated.

        Returns:
            Dict mapping ``set_id -> {"SET_NAME": str, "NODES": list[int],
            "ELEMENTS": list[int]}``. Member lists are sorted and
            deduplicated across partitions.
        """
        if isinstance(selection_set_ids, (int, np.integer)):
            selection_set_ids = [int(selection_set_ids)]

        if selection_set_ids is not None and not isinstance(selection_set_ids, list):
            raise ValueError("selection_set_ids must be a list of integers or None.")

        aggregated_data: dict[int, dict] = {}
        file_mapping = self.dataset.cdata_partitions

        for file_path in file_mapping.values():
            selection_sets = self._extract_selection_set_ids_for_file(
                file_path, selection_set_ids=selection_set_ids
            )

            for selection_set in selection_sets:
                set_id = selection_set["SET_ID"]

                if set_id not in aggregated_data:
                    aggregated_data[set_id] = {
                        "SET_NAME": selection_set["SET_NAME"],
                        "NODES": set(selection_set.get("NODES", [])),
                        "ELEMENTS": set(selection_set.get("ELEMENTS", [])),
                    }
                else:
                    aggregated_data[set_id]["NODES"].update(selection_set.get("NODES", []))
                    aggregated_data[set_id]["ELEMENTS"].update(selection_set.get("ELEMENTS", []))

        for set_id in aggregated_data:
            aggregated_data[set_id]["NODES"] = sorted(aggregated_data[set_id]["NODES"])
            aggregated_data[set_id]["ELEMENTS"] = sorted(aggregated_data[set_id]["ELEMENTS"])

        return aggregated_data

    def print_selection_set_names(self) -> None:
        """Emit names of all available selection sets at INFO level."""
        selection_sets = self.dataset.selection_set
        logger.info("Available selection sets:")
        for key, payload in selection_sets.items():
            logger.info("  Set id: %s - Set name: %s", key, payload["SET_NAME"])


# The legacy ``CData`` alias lives on the ``STKO_to_python.model``
# package surface (quiet) and at the deep path
# ``STKO_to_python.model.cdata`` (DeprecationWarning via PEP 562
# ``__getattr__`` shim). It is intentionally not declared on this
# canonical module so that the library never trips its own warning.



