"""Testes do Passo 4: buscador por texto, imagem e combinado."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.models.catalog_encoder import encode_catalog
from src.models.embedding_io import EmbeddingStore, StoredEmbeddings
from src.vectorstore.base import Hit, VectorStoreError
from src.vectorstore.metadata import build_where
from tests.conftest import FakeBackend

pytest.importorskip("chromadb")

from src.recommender.search import ProductSearcher, SearchError, load_query_image  # noqa: E402
from src.recommender.visualize import make_contact_sheet  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402
from src.vectorstore.indexer import index_embeddings  # noqa: E402

DIM = 8


class CountingBackend(FakeBackend):
    """FakeBackend que conta chamadas de embedding de texto (para testar o cache)."""

    def __init__(self) -> None:
        super().__init__(dim=DIM)
        self.text_calls = 0

    def embed_texts(self, texts):  # type: ignore[no-untyped-def]
        self.text_calls += 1
        return super().embed_texts(texts)


@pytest.fixture
def env(fake_catalog: pd.DataFrame, tmp_path: Path) -> dict:
    """Índice completo (imagem + texto) sobre o catálogo sintético; item 103 é inválido."""
    catalog = fake_catalog.copy()
    catalog["gender"] = ["Men", "Women"] * 5
    catalog["price"] = [10.0 * (i + 1) for i in range(10)]
    backend = CountingBackend()
    emb = EmbeddingStore(tmp_path / "emb", catalog["product_id"].tolist(), backend.name, backend.dim)
    emb.open()
    encode_catalog(backend, catalog, emb, batch_size=4, workers=2)
    emb.close()
    stored = StoredEmbeddings.load(tmp_path / "emb")

    def open_store(name: str) -> ChromaStore:
        return ChromaStore(tmp_path / "chroma", name, dim=DIM, fingerprint=stored.fingerprint, model_name=backend.name)

    img, txt = open_store("products_image"), open_store("products_text")
    index_embeddings(stored, catalog, {"image": img, "text": txt}, batch_size=4)
    return {
        "backend": backend, "img": img, "txt": txt, "catalog": catalog, "tmp": tmp_path,
        "searcher": ProductSearcher(backend, img, txt, default_k=3),
        "image_path": lambda pid: Path(catalog.loc[catalog["product_id"] == pid, "image_path"].iloc[0]),
    }  # fmt: skip


# ------------------------------------------------------------------- por imagem
def test_search_by_image_returns_self_first_and_sorted(env: dict) -> None:
    hits = env["searcher"].search_by_image(env["image_path"]("105"))
    assert len(hits) == 3  # default_k
    assert hits[0].product_id == "105" and hits[0].score > 0.999
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert hits[0].metadata["price"] == 60.0 and hits[0].metadata["gender"] == "Women"
    assert "103" not in {h.product_id for h in env["searcher"].search_by_image(env["image_path"]("105"), k=20)}


def test_exclude_ids_still_fills_k(env: dict) -> None:
    hits = env["searcher"].search_by_image(env["image_path"]("105"), k=4, exclude_ids=["105"])
    assert len(hits) == 4 and "105" not in {h.product_id for h in hits}


def test_filters_apply_to_image_search(env: dict) -> None:
    hits = env["searcher"].search_by_image(env["image_path"]("105"), k=9, where=build_where(gender="Men"))
    assert hits and all(h.metadata["gender"] == "Men" for h in hits)
    cheap = env["searcher"].search_by_image(env["image_path"]("105"), k=9, where=build_where(max_price=30))
    assert {h.product_id for h in cheap} == {"100", "101", "102"}


# -------------------------------------------------------------------- por texto
def test_text_to_text_finds_exact_description(env: dict) -> None:
    hits = env["searcher"].search_by_text("Produto 105. Tshirts, Red", k=3, mode="text")
    assert hits[0].product_id == "105" and hits[0].score > 0.999


def test_text_to_image_default_mode_returns_k(env: dict) -> None:
    hits = env["searcher"].search_by_text("anything", k=5)
    assert len(hits) == 5 and all(isinstance(h, Hit) for h in hits)


def test_text_cache_avoids_recomputation(env: dict) -> None:
    backend: CountingBackend = env["backend"]
    before = backend.text_calls
    env["searcher"].search_by_text("red shirt")
    env["searcher"].search_by_text("  red shirt  ")  # mesma consulta após strip
    assert backend.text_calls - before == 1


# ------------------------------------------------------------------- combinada
def test_multimodal_weights_reduce_to_single_modality(env: dict) -> None:
    s, path = env["searcher"], env["image_path"]("105")
    only_img = s.search_multimodal(path, "blue jeans", k=5, image_weight=1.0)
    assert [h.product_id for h in only_img] == [h.product_id for h in s.search_by_image(path, k=5)]
    only_txt = s.search_multimodal(path, "blue jeans", k=5, image_weight=0.0)
    assert [h.product_id for h in only_txt] == [h.product_id for h in s.search_by_text("blue jeans", k=5)]
    with pytest.raises(SearchError, match="image_weight"):
        s.search_multimodal(path, "x", image_weight=1.5)


# ------------------------------------------------------------------- validações
@pytest.mark.parametrize("bad_k", [0, 101, -1, True, 2.5])
def test_invalid_k_rejected(env: dict, bad_k: object) -> None:
    with pytest.raises(SearchError, match="k deve"):
        env["searcher"].search_by_text("x", k=bad_k)  # type: ignore[arg-type]


def test_invalid_inputs_rejected(env: dict) -> None:
    s = env["searcher"]
    with pytest.raises(SearchError, match="vazia"):
        s.search_by_text("   ")
    with pytest.raises(SearchError, match="não encontrada"):
        s.search_by_image(env["tmp"] / "nope.jpg")
    with pytest.raises(SearchError, match="ler a imagem"):
        s.search_by_image(b"isto nao e uma imagem")
    with pytest.raises(SearchError, match="mode inválido"):
        s.search_by_text("x", mode="audio")  # type: ignore[arg-type]
    only_images = ProductSearcher(env["backend"], env["img"], None)
    with pytest.raises(SearchError, match="coleção de textos"):
        only_images.search_by_text("x", mode="text")


def test_model_or_dim_mismatch_detected_at_construction(env: dict) -> None:
    class OtherBackend(FakeBackend):
        name = "other-model"  # type: ignore[assignment]

    with pytest.raises(SearchError, match="mesmo modelo"):
        ProductSearcher(OtherBackend(dim=DIM), env["img"])
    with pytest.raises(SearchError, match="Dimensão"):
        ProductSearcher(FakeBackend(dim=16), env["img"])


# ------------------------------------------------------------ utilitários / store
def test_load_query_image_accepts_all_sources_and_downsizes(env: dict, tmp_path: Path) -> None:
    path = env["image_path"]("101")
    buf = io.BytesIO()
    Image.new("RGBA", (50, 40), (1, 2, 3, 4)).save(buf, format="PNG")
    for source in (path, str(path), path.read_bytes(), Image.open(path), io.BytesIO(buf.getvalue())):
        assert load_query_image(source).mode == "RGB"
    assert max(load_query_image(Image.new("RGB", (3000, 1000)), max_side=1024).size) == 1024


def test_open_existing_roundtrip_and_errors(env: dict) -> None:
    store = ChromaStore.open_existing(env["tmp"] / "chroma", "products_image")
    assert (store.dim, store.model_name, store.count()) == (DIM, "fake-model", 9)
    assert ProductSearcher(env["backend"], store).search_by_image(env["image_path"]("102"), k=1)[0].product_id == "102"
    with pytest.raises(VectorStoreError, match="não existe"):
        ChromaStore.open_existing(env["tmp"] / "chroma", "outra")
    with pytest.raises(VectorStoreError, match="não encontrado"):
        ChromaStore.open_existing(env["tmp"] / "inexistente", "products_image")
    assert not (env["tmp"] / "inexistente").exists()  # abrir para leitura não cria pastas


def test_contact_sheet_handles_missing_thumbnails(env: dict) -> None:
    hits = env["searcher"].search_by_image(env["image_path"]("105"), k=3)
    hits.append(Hit("x", 0.1, {"name": "Sem imagem", "image_path": "nao/existe.jpg"}))
    sheet = make_contact_sheet(hits, Image.new("RGB", (64, 64), "red"), caption="teste")
    assert sheet.mode == "RGB" and sheet.size[0] >= 5 * 180 and np.asarray(sheet).std() > 0
