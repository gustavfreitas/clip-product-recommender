"""Testes do Passo 3: metadados, filtros, ChromaStore e indexação."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.catalog_encoder import encode_catalog
from src.models.embedding_io import EmbeddingStore, StoredEmbeddings
from src.vectorstore.base import VectorStoreError
from src.vectorstore.metadata import build_where, row_to_metadata
from tests.conftest import FakeBackend

chromadb = pytest.importorskip("chromadb")

from src.vectorstore.chroma_store import ChromaStore  # noqa: E402
from src.vectorstore.indexer import index_embeddings, verify_self_retrieval  # noqa: E402

DIM = 8


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _store(tmp_path: Path, name: str = "products_image", fp: str = "fp1", recreate: bool = False) -> ChromaStore:
    return ChromaStore(tmp_path / "chroma", name, dim=DIM, fingerprint=fp, model_name="m", recreate=recreate)


# --------------------------------------------------------------------- metadados
def test_row_to_metadata_drops_missing_and_casts_types() -> None:
    row = pd.Series(
        {
            "product_id": "7", "name": "Camiseta", "gender": "Men", "base_colour": pd.NA,
            "price": np.float64(99.9), "year": pd.NA, "price_is_synthetic": np.bool_(True),
            "image_path": "data/processed/images/7.jpg",
        }
    )  # fmt: skip
    meta = row_to_metadata(row)
    assert meta == {
        "product_id": "7", "name": "Camiseta", "gender": "Men", "price": 99.9,
        "price_is_synthetic": True, "image_path": "data/processed/images/7.jpg",
    }  # fmt: skip
    assert type(meta["price"]) is float and type(meta["price_is_synthetic"]) is bool


def test_build_where_variants() -> None:
    assert build_where() is None
    assert build_where(gender="Men") == {"gender": {"$eq": "Men"}}
    assert build_where(article_type=["Shoes", "Sandals"]) == {"article_type": {"$in": ["Shoes", "Sandals"]}}
    assert build_where(gender="Men", max_price=200, min_price=50) == {
        "$and": [{"gender": {"$eq": "Men"}}, {"price": {"$gte": 50.0}}, {"price": {"$lte": 200.0}}]
    }
    assert build_where(gender=None, article_type=[]) is None
    with pytest.raises(ValueError, match="desconhecido"):
        build_where(color="red")


# ------------------------------------------------------------------- ChromaStore
def test_query_orders_by_cosine_and_scores_match_numpy(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    emb = _unit(rng, 50)
    ids = [str(i) for i in range(50)]
    store = _store(tmp_path)
    store.upsert(ids, emb, [{"product_id": i} for i in ids], [f"doc {i}" for i in ids])

    q = _unit(rng, 1)[0]
    hits = store.query(q, k=5)
    expected = np.argsort(-(emb @ q))[:5]
    assert [h.product_id for h in hits] == [str(i) for i in expected]
    assert np.allclose([h.score for h in hits], (emb @ q)[expected], atol=1e-4)  # score == cosseno
    assert all(hits[i].score >= hits[i + 1].score for i in range(4))


def test_filters_k_clamp_and_empty(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    emb = _unit(rng, 6)
    ids = [str(i) for i in range(6)]
    metas = [{"product_id": i, "gender": "Men" if int(i) % 2 == 0 else "Women", "price": 10.0 * (int(i) + 1)} for i in ids]
    store = _store(tmp_path)
    assert store.query(emb[0], k=3) == []  # coleção vazia
    store.upsert(ids, emb, metas)

    men = store.query(emb[0], k=10, where=build_where(gender="Men"))  # k > total é limitado
    assert {h.product_id for h in men} == {"0", "2", "4"}
    cheap_men = store.query(emb[0], k=10, where=build_where(gender="Men", max_price=30))
    assert {h.product_id for h in cheap_men} == {"0", "2"}
    assert store.query(emb[0], k=3, where=build_where(gender="Men", min_price=1000)) == []


def test_upsert_is_idempotent_and_updates_metadata(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    emb = _unit(rng, 4)
    ids = ["a", "b", "c", "d"]
    store = _store(tmp_path)
    store.upsert(ids, emb, [{"product_id": i, "price": 1.0} for i in ids])
    store.upsert(ids, emb, [{"product_id": i, "price": 2.0} for i in ids])
    assert store.count() == 4
    assert store.query(emb[0], k=1)[0].metadata["price"] == 2.0


def test_upsert_validation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    good = _unit(np.random.default_rng(4), 2)
    metas = [{"product_id": "a"}, {"product_id": "b"}]
    with pytest.raises(VectorStoreError, match="Formato"):
        store.upsert(["a", "b"], good[:, :4], metas)
    with pytest.raises(VectorStoreError, match="duplicados"):
        store.upsert(["a", "a"], good, metas)
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(VectorStoreError, match="NaN"):
        store.upsert(["a", "b"], bad, metas)
    with pytest.raises(VectorStoreError, match="Dimensão"):
        store.query(np.zeros(4, dtype=np.float32))


def test_fingerprint_mismatch_requires_recreate(tmp_path: Path) -> None:
    emb = _unit(np.random.default_rng(5), 3)
    s1 = _store(tmp_path, fp="v1")
    s1.upsert(["a", "b", "c"], emb, [{"product_id": i} for i in "abc"])
    with pytest.raises(VectorStoreError, match="--recreate"):
        _store(tmp_path, fp="v2")
    assert _store(tmp_path, fp="v1").count() == 3  # mesmo fingerprint reabre
    assert _store(tmp_path, fp="v2", recreate=True).count() == 0


# ------------------------------------------------------------------- indexação
@pytest.fixture
def stored_and_catalog(fake_catalog: pd.DataFrame, tmp_path: Path) -> tuple[StoredEmbeddings, pd.DataFrame]:
    catalog = fake_catalog.copy()
    catalog["gender"] = ["Men", "Women"] * 5
    catalog["price"] = [10.0 * (i + 1) for i in range(10)]
    catalog["year"] = pd.array([2011, None, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], dtype="Int64")
    backend = FakeBackend(dim=DIM)
    emb_store = EmbeddingStore(tmp_path / "emb", catalog["product_id"].tolist(), backend.name, backend.dim)
    emb_store.open()
    encode_catalog(backend, catalog, emb_store, batch_size=3, workers=2)
    emb_store.close()
    return StoredEmbeddings.load(tmp_path / "emb"), catalog


def test_index_embeddings_end_to_end(stored_and_catalog: tuple[StoredEmbeddings, pd.DataFrame], tmp_path: Path) -> None:
    stored, catalog = stored_and_catalog
    img = ChromaStore(tmp_path / "chroma", "products_image", dim=DIM, fingerprint=stored.fingerprint, model_name="m")
    txt = ChromaStore(tmp_path / "chroma", "products_text", dim=DIM, fingerprint=stored.fingerprint, model_name="m")

    stats = index_embeddings(stored, catalog, {"image": img, "text": txt}, batch_size=4)
    assert stats.indexed == 9 and stats.skipped_invalid == 1 and stats.missing_metadata == 0
    assert img.count() == txt.count() == 9

    # item 103 (imagem ausente) nunca chega ao banco; metadados ricos presentes
    assert "103" not in img.get_embeddings(["103"])
    hit = img.query(img.get_embeddings(["105"])["105"], k=1)[0]
    assert hit.product_id == "105" and hit.score > 0.9999
    assert hit.metadata["price"] == 60.0 and hit.metadata["gender"] == "Women"
    assert hit.metadata["image_path"].endswith("105.jpg") and "year" in hit.metadata

    # reexecução não duplica
    index_embeddings(stored, catalog, {"image": img}, batch_size=4)
    assert img.count() == 9

    report = verify_self_retrieval(img, stored, sample=50)
    assert report["n"] == 9 and report["recall_at_1"] == 1.0 and report["mean_top1_score"] > 0.9999


def test_index_skips_ids_missing_from_catalog(stored_and_catalog: tuple[StoredEmbeddings, pd.DataFrame], tmp_path: Path) -> None:
    stored, catalog = stored_and_catalog
    store = ChromaStore(tmp_path / "chroma", "products_image", dim=DIM, fingerprint=stored.fingerprint, model_name="m")
    stats = index_embeddings(stored, catalog[catalog["product_id"] != "107"], {"image": store}, batch_size=4)
    assert stats.missing_metadata == 1 and store.count() == 8
