"""Reader for ``LayeredShell`` section definitions in OpenSees Tcl scripts.

The MPCO HDF5 file carries layered-shell **results** (one column per
``(gauss_point × thickness_layer)`` block under
``material.fiber.stress`` / ``section.fiber.stress``), but it does not
carry the geometric layer definition — the layer thicknesses and
materials live in the source ``sections.tcl`` script that builds the
OpenSees model. The per-layer section-cut breakdown
(:func:`STKO_to_python.cuts.kernels.shell.compute_shell_cut_per_layer`)
needs that geometry to weight each layer's stress by its thickness and
through-thickness offset.

This module parses the lines

::

    section LayeredShell <section_id> <n_layers> \\
         <mat_id_1> <t_1>  <mat_id_2> <t_2>  ...

and exposes them as a ``{section_id: tuple[LayerInfo, ...]}`` table.
``LayerInfo.z_offset`` is the layer midplane's signed distance from the
section midplane, computed as ``bottom_z + 0.5 * thickness`` with
``bottom_z = -T/2 + Σ t_j`` for layers ``j < k``.

The parser is intentionally lenient: lines that don't start with
``section LayeredShell`` are skipped, Tcl continuation backslashes are
collapsed, and unknown trailing tokens (e.g. ``-fName`` style options)
are dropped. The MPCO results carry the ground truth — this module is
only a convenience for callers who keep their model source alongside
their recorder output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayerInfo:
    """One through-thickness layer in a LayeredShell section.

    Attributes
    ----------
    material_id : int
        OpenSees material tag assigned to this layer.
    thickness : float
        Layer thickness (length units of the model).
    z_offset : float
        Signed distance of the layer midplane from the **section
        midplane**. Bottom layer has the most-negative offset; top
        layer has the most-positive. Used to compute through-thickness
        moments via ``M = ∫ σ · z dz ≈ Σ σ_k · t_k · z_offset_k``.
    """

    material_id: int
    thickness: float
    z_offset: float


# Tcl section command — captures the section id and the parameter
# tail. The tail is parsed token-by-token below; matching a flexible
# regex on the whole layer list would have to handle continuation
# backslashes and arbitrary whitespace, which is simpler done linearly.
_SECTION_HEADER_RE = re.compile(
    r"\bsection\s+LayeredShell\s+(\d+)\s+(\d+)\s+(.*)$",
    re.IGNORECASE,
)


def parse_sections_tcl(path: str | Path) -> dict[int, tuple[LayerInfo, ...]]:
    """Parse a Tcl script's ``LayeredShell`` definitions.

    Returns ``{section_id: (LayerInfo, ...)}``. Sections without a
    ``LayeredShell`` declaration (``ElasticMembranePlateSection``, etc.)
    are silently skipped.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a ``LayeredShell`` header declares N layers but the
        following body doesn't carry N ``(material_id, thickness)``
        pairs after the continuation backslashes are collapsed.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")

    # Collapse Tcl line continuations: ``\\\n`` becomes a single space
    # so each ``section ...`` statement lives on one logical line.
    logical = re.sub(r"\\\s*\n", " ", text)

    out: dict[int, tuple[LayerInfo, ...]] = {}
    for raw_line in logical.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SECTION_HEADER_RE.search(line)
        if m is None:
            continue
        section_id = int(m.group(1))
        n_layers = int(m.group(2))
        tail = m.group(3).strip()

        # Parse the tail as alternating <material_id> <thickness> pairs.
        # Tcl tolerates extra whitespace; ``.split()`` handles any
        # combination of spaces / tabs.
        tokens = tail.split()
        if len(tokens) < 2 * n_layers:
            raise ValueError(
                f"LayeredShell section {section_id}: declared "
                f"{n_layers} layers but the body has only {len(tokens)} "
                f"tokens (need at least {2 * n_layers})."
            )
        layers_raw: list[tuple[int, float]] = []
        for k in range(n_layers):
            mat_str = tokens[2 * k]
            thick_str = tokens[2 * k + 1]
            try:
                mat_id = int(mat_str)
                thick = float(thick_str)
            except ValueError as exc:
                raise ValueError(
                    f"LayeredShell section {section_id} layer {k}: "
                    f"cannot parse ({mat_str!r}, {thick_str!r}) as "
                    "(int, float)."
                ) from exc
            layers_raw.append((mat_id, thick))

        # Compute z_offset for each layer: midplane of layer k =
        # -T/2 + sum_{j<k} t_j + t_k / 2.
        total_t = float(sum(t for _, t in layers_raw))
        out[section_id] = _build_layer_infos(layers_raw, total_t)

    return out


def _build_layer_infos(
    layers_raw: list[tuple[int, float]], total_t: float,
) -> tuple[LayerInfo, ...]:
    """Compute centroid z-offsets given (material_id, thickness) pairs."""
    half_t = 0.5 * total_t
    out: list[LayerInfo] = []
    running_bottom = -half_t
    for mat_id, thick in layers_raw:
        midplane = running_bottom + 0.5 * thick
        out.append(LayerInfo(
            material_id=int(mat_id),
            thickness=float(thick),
            z_offset=float(midplane),
        ))
        running_bottom += thick
    return tuple(out)


def find_sections_tcl(dataset_directory: str | Path) -> Path | None:
    """Locate ``sections.tcl`` beside an MPCO recorder.

    Searches ``dataset_directory`` and its immediate parent for a file
    named ``sections.tcl`` (case-insensitive on case-preserving file
    systems). Returns the path of the first match, or ``None`` if no
    candidate is found. Callers should treat ``None`` as "no layer
    table available" and surface a useful error message.
    """
    base = Path(dataset_directory)
    candidates = [base / "sections.tcl", base.parent / "sections.tcl"]
    for cand in candidates:
        if cand.exists():
            return cand
    return None
