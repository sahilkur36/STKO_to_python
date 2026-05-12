"""Schema versions for persisted viewer state.

These integers pin the on-disk format of saved scenes and session
files so that a future viewer can reject — or migrate — a file written
by an older version. The viewer subpackage has no persisted format at
v1.9.0 (Phase 0 is namespace-only); pinning the schema versions here
lets us bump them deliberately as features land.

Bump rules:
    * Bump ``SCENE_SPEC_SCHEMA`` when the ``SceneSpec`` / ``LayerSpec``
      on-disk layout changes incompatibly.
    * Bump ``SESSION_SCHEMA`` when the Qt session-state file (window
      layout, dock geometry, theme) changes incompatibly.
    * Add a migration in ``viewer.core.specs`` for every bump.

The current values (``0``) signal "no on-disk format yet"; they will
move to ``1`` in Phase 2 (``SceneSpec``) and Phase 4 (``SESSION``).
"""
from __future__ import annotations

SCENE_SPEC_SCHEMA: int = 0
SESSION_SCHEMA: int = 0
