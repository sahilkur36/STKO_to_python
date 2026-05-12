"""Layer-level selection spec.

A frozen, hashable specification of "which entities does this layer
render". The grammar mirrors the existing
:class:`STKO_to_python.selection.resolver.SelectionSetResolver` API so
the Phase 2 step 2 adapter (``MPCODataSourceAdapter``) can translate
1-to-1.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SelectionSpec:
    """Selection criteria for a layer.

    All fields are **AND-combined**: a layer renders only the entities
    that match every non-``None`` criterion. The fields are explicit
    tuples (not arrays) so the dataclass stays hashable — selection
    specs are cache keys downstream.

    Attributes:
        selection_set_name: One or more named selection sets (e.g.
            ``"slab"`` or ``("slab", "perimeter")``).
        selection_set_id: One or more selection-set IDs.
        node_ids: Explicit node IDs to include.
        element_ids: Explicit element IDs to include.
        element_type: One or more decorated element type strings (e.g.
            ``"203-ASDShellQ4"``).
    """

    selection_set_name: str | tuple[str, ...] | None = None
    selection_set_id: int | tuple[int, ...] | None = None
    node_ids: tuple[int, ...] | None = None
    element_ids: tuple[int, ...] | None = None
    element_type: str | tuple[str, ...] | None = None

    @classmethod
    def empty(cls) -> "SelectionSpec":
        """Empty spec — matches every entity."""
        return cls()

    def is_empty(self) -> bool:
        """True if every field is ``None`` (the empty / match-all spec)."""
        return all(getattr(self, f.name) is None for f in fields(self))
