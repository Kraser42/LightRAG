"""Tests for the optional per-chunk candidate-name hook.

``LightRAG.entity_name_candidate_hook`` runs once per chunk before extraction
and returns existing entity names that are injected into the extraction user
prompt as ``---Existing Entities---``. ``None`` (the default) leaves the prompt
byte-identical.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.disambiguate import candidates_by_embedding
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


_CHUNK_CONTENTS = {
    "chunk-alpha": "Alpha content about Alice.",
}


def _extraction_result(name: str) -> str:
    return f"(entity<|#|>{name}<|#|>CONCEPT<|#|>Description of {name})<|COMPLETE|>"


async def _fake_extract(prompt: str, *args, **kwargs) -> str:
    if "Alpha content about Alice." in prompt:
        return _extraction_result("ALPHA")
    raise AssertionError(f"prompt carried no known chunk content: {prompt[:200]!r}")


def _make_chunks() -> dict[str, dict]:
    return {
        key: {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": index,
            "file_path": f"{key}.md",
        }
        for index, (key, content) in enumerate(_CHUNK_CONTENTS.items())
    }


def _make_global_config(extract_func, candidate_hook=None) -> dict:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": tokenizer,
        "llm_model_max_async": 3,
        "entity_name_candidate_hook": candidate_hook,
    }


class _FakeVDB:
    def __init__(self, names: list[str] | None = None):
        self._names = names or []
        self.queries: list[str] = []

    async def query(self, query, top_k=None, query_embedding=None):
        self.queries.append(query)
        return [{"entity_name": n} for n in self._names]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_hook_injects_candidate_block_into_prompt():
    """The returned names reach the extraction prompt as ---Existing Entities---."""
    from lightrag.operate import extract_entities

    vdb = _FakeVDB(["Elon Musk", "Tesla"])
    seen: list[str] = []

    async def extract(prompt: str, *args, **kwargs) -> str:
        seen.append(prompt)
        return _extraction_result("ALPHA")

    config = _make_global_config(
        AsyncMock(side_effect=extract), candidate_hook=candidates_by_embedding
    )

    await extract_entities(
        chunks=_make_chunks(), global_config=config, entity_vdb=vdb
    )

    assert seen, "extraction should have run"
    prompt = seen[0]
    assert "---Existing Entities---" in prompt
    assert "Elon Musk" in prompt and "Tesla" in prompt
    # The hook received the chunk text to search.
    assert vdb.queries == ["Alpha content about Alice."]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_candidates_leave_prompt_unchanged():
    from lightrag.operate import extract_entities

    vdb = _FakeVDB([])  # empty result -> no block
    seen: list[str] = []

    async def extract(prompt: str, *args, **kwargs) -> str:
        seen.append(prompt)
        return _extraction_result("ALPHA")

    config = _make_global_config(
        AsyncMock(side_effect=extract), candidate_hook=candidates_by_embedding
    )
    await extract_entities(
        chunks=_make_chunks(), global_config=config, entity_vdb=vdb
    )

    assert "---Existing Entities---" not in seen[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_none_hook_never_injects_and_never_calls_vdb():
    from lightrag.operate import extract_entities

    vdb = _FakeVDB(["Elon Musk"])
    seen: list[str] = []

    async def extract(prompt: str, *args, **kwargs) -> str:
        seen.append(prompt)
        return _extraction_result("ALPHA")

    config = _make_global_config(AsyncMock(side_effect=extract), candidate_hook=None)
    await extract_entities(
        chunks=_make_chunks(), global_config=config, entity_vdb=vdb
    )

    assert "---Existing Entities---" not in seen[0]
    assert vdb.queries == [], "hook disabled: vector store must not be queried"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_candidates_by_embedding_fail_safe_on_query_error():
    class BoomVDB:
        async def query(self, *a, **k):
            raise RuntimeError("vdb down")

    result = await candidates_by_embedding(
        "some text", BoomVDB(), {}, top_k=20
    )
    assert result == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_candidates_by_embedding_none_vdb_returns_empty():
    assert await candidates_by_embedding("some text", None, {}) == []


def _new_rag(tmp_path: Path, hook) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"candidate-{tmp_path.name}",
        llm_model_func=lambda *a, **k: "",
        embedding_func=EmbeddingFunc(
            embedding_dim=2,
            max_token_size=4096,
            func=lambda texts: np.zeros((len(texts), 2)),
        ),
        tokenizer=Tokenizer("dummy", DummyTokenizer()),
        entity_name_candidate_hook=hook,
    )


@pytest.mark.offline
def test_default_hook_is_none_in_global_config(tmp_path):
    rag = _new_rag(tmp_path, None)
    assert rag._build_global_config()["entity_name_candidate_hook"] is None


@pytest.mark.offline
def test_hook_lands_in_global_config(tmp_path):
    rag = _new_rag(tmp_path, candidates_by_embedding)
    assert (
        rag._build_global_config()["entity_name_candidate_hook"]
        is candidates_by_embedding
    )
