"""Testes do app Streamlit (AppTest) com índice sintético e CLIP simulado."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from src.models.catalog_encoder import encode_catalog
from src.models.embedding_io import EmbeddingStore, StoredEmbeddings
from tests.conftest import FakeBackend

pytest.importorskip("chromadb")
st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.recommender.facets import compute_facets  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402
from src.vectorstore.indexer import index_embeddings  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
DIM = 8


class _StubClip(FakeBackend):
    """Substitui o ClipEmbedder real (que exigiria torch e download do modelo)."""

    def __init__(self, model_name: str = "fake-model", device: str = "auto") -> None:
        super().__init__(dim=DIM)


@pytest.fixture(autouse=True)
def _clean_streamlit_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def built_index(fake_catalog: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ChromaStore:
    """Indexa o catálogo sintético e aponta o app para ele via variáveis de ambiente."""
    catalog = fake_catalog.copy()
    catalog["gender"] = ["Men", "Women"] * 5
    catalog["article_type"] = ["Tshirts", "Dresses"] * 5
    catalog["base_colour"] = ["Red", "Unknown"] * 5
    catalog["price"] = [10.0 * (i + 1) for i in range(10)]
    backend = FakeBackend(dim=DIM)
    emb = EmbeddingStore(tmp_path / "emb", catalog["product_id"].tolist(), backend.name, backend.dim)
    emb.open()
    encode_catalog(backend, catalog, emb, batch_size=4, workers=2)
    emb.close()
    stored = StoredEmbeddings.load(tmp_path / "emb")
    stores = {
        m: ChromaStore(tmp_path / "vs", f"products_{m}", dim=DIM, fingerprint=stored.fingerprint, model_name=backend.name)
        for m in ("image", "text")
    }
    index_embeddings(stored, catalog, stores, batch_size=4)  # type: ignore[arg-type]

    monkeypatch.setenv("MMREC_VECTORSTORE_DIR", str(tmp_path / "vs"))
    stub = types.ModuleType("src.models.clip_embedder")
    stub.ClipEmbedder = _StubClip  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.models.clip_embedder", stub)
    return stores["image"]  # type: ignore[return-value]


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def _card_buttons(at: AppTest) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith("sim_")]


def test_facets_come_from_index(built_index: ChromaStore) -> None:
    f = compute_facets(built_index, batch_size=3)
    assert f.total == 9 and f.genders == ["Men", "Women"]
    assert f.article_types == ["Tshirts", "Dresses"] or set(f.article_types) == {"Tshirts", "Dresses"}
    assert f.colours == ["Red"]  # "Unknown" não vira opção de filtro
    assert (f.price_min, f.price_max) == (10.0, 100.0)


def test_empty_state_shows_examples_and_no_exception(built_index: ChromaStore) -> None:
    at = _run()
    assert not at.exception
    assert [b.label for b in at.button if b.key and b.key.startswith("ex_")][0] == "black running shoes"
    assert not _card_buttons(at)


def test_text_query_renders_k_cards(built_index: ChromaStore) -> None:
    at = _run()
    at.text_input(key="query_text").input("red shirt").run()
    assert not at.exception and not at.error
    assert len(_card_buttons(at)) == 5  # default_top_k
    at.sidebar.slider[0].set_value(3).run()
    assert len(_card_buttons(at)) == 3


def test_example_button_fills_query(built_index: ChromaStore) -> None:
    at = _run()
    next(b for b in at.button if b.key == "ex_red summer dress").click().run()
    assert at.text_input(key="query_text").value == "red summer dress"
    assert len(_card_buttons(at)) == 5


def test_gender_filter_and_empty_result(built_index: ChromaStore) -> None:
    at = _run()
    at.text_input(key="query_text").input("shirt").run()
    at.sidebar.multiselect[0].set_value(["Men"]).run()
    assert not at.exception and len(_card_buttons(at)) == 5  # 5 homens no índice (100,102,104,106,108)
    at.sidebar.multiselect[2].set_value(["Dresses"]).run()  # Men + Dresses: combinação inexistente
    assert not _card_buttons(at) and any("Nenhum produto" in i.value for i in at.info)


def test_similar_mode_excludes_source_and_can_be_cleared(built_index: ChromaStore) -> None:
    at = _run()
    at.text_input(key="query_text").input("shirt").run()
    first = _card_buttons(at)[0]
    source_id = first.split("_")[1]
    at.button(key=first).click().run()
    assert not at.exception and at.session_state["similar_to"]["id"] == source_id
    assert len(_card_buttons(at)) == 5
    assert all(not key.startswith(f"sim_{source_id}_") for key in _card_buttons(at))  # o próprio item não aparece
    next(b for b in at.button if b.label == "Limpar").click().run()
    assert at.session_state["similar_to"] is None and not _card_buttons(at)


def test_missing_index_shows_friendly_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMREC_VECTORSTORE_DIR", str(tmp_path / "nao_existe"))
    at = _run()
    assert any("Não foi possível iniciar" in e.value for e in at.error)
    assert not at.exception
