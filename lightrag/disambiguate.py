"""Entity-name disambiguation reference implementation (opt-in).

LightRAG merges entities by exact name after ``normalize_entity_name``, so the
same real-world entity written as "马斯克" / "Elon Musk" / "埃隆·马斯克" becomes
three disconnected nodes. This module provides an ``entity_disambiguation_hook``
that folds such names into one canonical key BEFORE the merge runs, so the folded
records flow through the existing merge (type voting, description dedup,
source-id join) unchanged.

Design is deliberately conservative — precision over recall. A wrong merge
pollutes source_id / description / relation chains and is hard to unwind, so:

* only pairs above ``high_threshold`` are auto-merged (L2);
* pairs in ``[low_threshold, high_threshold)`` are logged as "needs review" and
  left un-merged (L3);
* embedding failures leave ``all_nodes`` untouched rather than failing the merge.

This is a reference implementation for a single merge's candidate set (typically
one document's entities, so N is small and a pairwise pass is acceptable). Whole-
graph dedup of already-persisted nodes is a separate backfill concern, out of
scope here.
"""

from __future__ import annotations

from typing import Any

from .utils import cosine_similarity, logger, normalize_entity_name

DEFAULT_HIGH_THRESHOLD = 0.92
DEFAULT_LOW_THRESHOLD = 0.80


def _canonical_name(group: list[str], record_counts: dict[str, int]) -> str:
    """Pick the canonical name for a group of equivalent names.

    Prefer the name with the most entity records (most mentions); tie-break on
    the shortest name (fewer characters, more likely the canonical form).
    """
    return min(group, key=lambda name: (-record_counts.get(name, 0), len(name)))


async def disambiguate_by_embedding(
    all_nodes: dict[str, list[dict]],
    entity_vdb: Any,
    global_config: dict[str, Any],
    *,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
) -> dict[str, list[dict]]:
    """Fold entity names that refer to the same entity into one canonical key.

    Args:
        all_nodes: ``{entity_name: [entity_dict, ...]}`` — the merge candidate set.
        entity_vdb: the entity vector store (exposes ``embedding_func``).
        global_config: instance config (currently unused; reserved).
        high_threshold: cosine similarity at/above which names auto-merge.
        low_threshold: cosine similarity at/above which a pair is logged as
            "needs review" but NOT merged.

    Returns:
        A new ``all_nodes`` dict of the same shape. The merge treats only the
        dict key as the entity identity, so folding records under a canonical
        key is safe regardless of each record's own ``entity_name`` field.
    """
    if not all_nodes:
        return all_nodes

    embedding_func = getattr(entity_vdb, "embedding_func", None)
    if embedding_func is None:
        return all_nodes

    # One canonical key per normalized name. all_nodes keys are already
    # normalized by extraction, but normalize again so the hook is robust to
    # keys arriving from custom/API paths.
    normalized: dict[str, str] = {}
    for name in all_nodes:
        normalized.setdefault(normalize_entity_name(name), name)

    names = list(normalized.keys())
    if len(names) < 2:
        return all_nodes

    try:
        embeddings = await embedding_func(names)
    except Exception as exc:  # fail-safe: never let the hook break the merge
        logger.warning(f"[disambiguate] embedding failed, skipping: {exc}")
        return all_nodes

    embeddings = _as_2d(embeddings)
    if embeddings is None or len(embeddings) != len(names):
        logger.warning(
            "[disambiguate] unexpected embedding shape, skipping "
            f"(expected {len(names)} rows)"
        )
        return all_nodes

    n = len(names)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Pairwise comparison on the pre-computed name embeddings. O(N^2) is fine
    # for a single merge's candidate set.
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= high_threshold:
                union(i, j)
            elif sim >= low_threshold:
                logger.info(
                    "[disambiguate] near-duplicate (needs review): "
                    f"{names[i]!r} ~ {names[j]!r} (sim={sim:.3f})"
                )

    # Group by union-find root; skip groups of size 1.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    record_counts = {name: len(all_nodes[name]) for name in all_nodes}
    merged: dict[str, list[dict]] = dict(all_nodes)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        group_names = [normalized[names[i]] for i in indices]
        canonical = _canonical_name(group_names, record_counts)
        bucket = merged.setdefault(canonical, [])
        for i in indices:
            name = normalized[names[i]]
            if name == canonical:
                continue
            bucket.extend(merged.pop(name, []))
        logger.info(
            f"[disambiguate] merged {len(group_names)} names -> {canonical!r}: "
            + ", ".join(repr(n) for n in group_names)
        )

    return merged


def _as_2d(embeddings: Any):
    """Return ``embeddings`` as a 2-D sequence of row vectors, or ``None``."""
    if embeddings is None:
        return None
    # numpy array is the normal EmbeddingFunc return; tolerate nested lists too.
    import numpy as np

    arr = np.asarray(embeddings)
    if arr.ndim == 1:
        return None
    return arr


async def candidates_by_embedding(
    chunk_text: str,
    entity_vdb: Any,
    global_config: dict[str, Any],
    *,
    top_k: int = 20,
) -> list[str]:
    """Return existing entity names most related to ``chunk_text``.

    Intended as an ``entity_name_candidate_hook``: the returned names are
    injected into the extraction prompt as ``---Existing Entities---`` so the
    model reuses an existing name when the chunk refers to the same real-world
    entity. Fail-safe: a missing/erroring vector store returns ``[]`` and the
    prompt stays unchanged.
    """
    if entity_vdb is None:
        return []
    try:
        results = await entity_vdb.query(chunk_text, top_k=top_k)
    except Exception as exc:  # fail-safe: never let the hook break extraction
        logger.warning(f"[candidates] entity_vdb.query failed, skipping: {exc}")
        return []
    names = [r.get("entity_name") for r in results if r.get("entity_name")]
    # Dedup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped

