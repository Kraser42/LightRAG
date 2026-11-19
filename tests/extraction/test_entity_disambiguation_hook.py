"""Tests for the optional entity-name disambiguation hook.

``LightRAG.entity_disambiguation_hook`` runs once per merge, after the
``all_nodes`` candidate set is aggregated and before any graph/vector mutation.
``None`` (the default) leaves the merge unchanged. The reference implementation
``lightrag.disambiguate.disambiguate_by_embedding`` folds names whose embeddings
are close enough into one canonical key; near-threshold pairs are logged but
never auto-merged (precision over recall).
"""

from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.disambiguate import disambiguate_by_embedding
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


# Deterministic embeddings keyed by name, so cosine similarity is under our
# control. "Elon Musk"/"马斯克" point the same way (same entity); "Musk-ish" is
# ~0.85 from them (near, needs review); "Apple" is orthogonal (distinct).
_NAME_VECTORS = {
    "Elon Musk": np.array([1.0, 0.0]),
    "马斯克": np.array([1.0, 0.0]),
    "Musk-ish": np.array([0.85, np.sqrt(1 - 0.85**2)]),
    "Apple": np.array([0.0, 1.0]),
}


async def _fake_embedding(texts: list[str]) -> np.ndarray:
    return np.asarray([_NAME_VECTORS[t] for t in texts], dtype=np.float32)


def _entity_record(name: str) -> dict:
    return {
        "entity_name": name,
        "entity_type": "PERSON",
        "description": f"Description of {name}",
        "source_id": "doc-001",
    }


class _FakeVDB:
    def __init__(self, embedding_func=None):
        self.embedding_func = embedding_func


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merges_names_with_high_similarity():
    all_nodes = {
        "Elon Musk": [_entity_record("Elon Musk")],
        "马斯克": [_entity_record("马斯克")],
        "Apple": [_entity_record("Apple")],
    }
    result = await disambiguate_by_embedding(
        all_nodes, _FakeVDB(embedding_func=_fake_embedding), {}
    )
    # "马斯克" folds into a single key with "Elon Musk" (same entity). On an
    # equal record count the tie-break prefers the SHORTER name, so "马斯克"
    # (3 chars) is canonical over "Elon Musk" (9 chars); "Apple" stays separate.
    assert sorted(result) == ["Apple", "马斯克"]
    assert len(result["马斯克"]) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_near_threshold_names_are_not_merged():
    # cosine("Elon Musk", "Musk-ish") ≈ 0.85 → below high (0.92), above low (0.80)
    all_nodes = {
        "Elon Musk": [_entity_record("Elon Musk")],
        "Musk-ish": [_entity_record("Musk-ish")],
    }
    result = await disambiguate_by_embedding(
        all_nodes, _FakeVDB(embedding_func=_fake_embedding), {}
    )
    assert sorted(result) == ["Elon Musk", "Musk-ish"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_distinct_names_stay_separate():
    all_nodes = {
        "Elon Musk": [_entity_record("Elon Musk")],
        "Apple": [_entity_record("Apple")],
    }
    result = await disambiguate_by_embedding(
        all_nodes, _FakeVDB(embedding_func=_fake_embedding), {}
    )
    assert sorted(result) == ["Apple", "Elon Musk"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_embedding_func_returns_unchanged():
    all_nodes = {"Elon Musk": [_entity_record("Elon Musk")]}
    result = await disambiguate_by_embedding(all_nodes, _FakeVDB(), {})
    assert result == all_nodes


@pytest.mark.offline
@pytest.mark.asyncio
async def test_embedding_failure_returns_unchanged():
    async def boom(texts):
        raise RuntimeError("embedding down")

    all_nodes = {"Elon Musk": [_entity_record("Elon Musk")]}
    result = await disambiguate_by_embedding(
        all_nodes, _FakeVDB(embedding_func=boom), {}
    )
    assert result == all_nodes


def _new_rag(tmp_path: Path, hook) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"disambig-{tmp_path.name}",
        llm_model_func=lambda *a, **k: "",
        embedding_func=EmbeddingFunc(
            embedding_dim=2, max_token_size=4096, func=_fake_embedding
        ),
        tokenizer=Tokenizer("dummy", DummyTokenizer()),
        entity_disambiguation_hook=hook,
    )


@pytest.mark.offline
def test_default_hook_is_none_in_global_config(tmp_path):
    rag = _new_rag(tmp_path, None)
    assert rag._build_global_config()["entity_disambiguation_hook"] is None


@pytest.mark.offline
def test_hook_lands_in_global_config(tmp_path):
    rag = _new_rag(tmp_path, disambiguate_by_embedding)
    config = rag._build_global_config()
    assert config["entity_disambiguation_hook"] is disambiguate_by_embedding
