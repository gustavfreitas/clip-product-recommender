"""Testes da avaliação offline: métricas puras e experimentos sobre clusters sintéticos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation.metrics import hit_at_k, precision_at_k, reciprocal_rank, summarize
from src.models.base import EmbeddingBackend

pytest.importorskip("chromadb")

from src.evaluation.runner import (  # noqa: E402
    Item,
    evaluate_image_to_image,
    evaluate_text_queries,
    format_markdown,
    load_items,
)
from src.recommender.search import ProductSearcher  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402

DIM = 8


# ---------------------------------------------------------------------- métricas
def test_precision_hit_and_reciprocal_rank() -> None:
    rels = [True, False, True, False]
    assert precision_at_k(rels, 2) == 0.5
    assert precision_at_k(rels, 4) == 0.5
    assert precision_at_k([True], 5) == 0.2  # resultados ausentes contam como erro
    assert hit_at_k(rels, 1) == 1.0 and hit_at_k([False, False, True], 2) == 0.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(rels, 0)


def test_summarize_mean_and_confidence_interval() -> None:
    assert summarize([]) == {"mean": 0.0, "ci95": 0.0, "n": 0}
    assert summarize([0.4]) == {"mean": 0.4, "ci95": 0.0, "n": 1}
    s = summarize([0.0, 1.0] * 50)
    assert s["mean"] == 0.5 and s["n"] == 100 and 0.09 < s["ci95"] < 0.11  # ≈ 1,96·0,5/√100


# ------------------------------------------------------------ clusters sintéticos
class ClusterBackend(EmbeddingBackend):
    """Texto com 'alpha' ou 'beta' cai no centro do respectivo cluster."""

    name = "cluster-model"  # type: ignore[assignment]
    dim = DIM  # type: ignore[assignment]

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0 if "alpha" in t.lower() else 1] = 1.0
        return out


@pytest.fixture
def clustered_store(tmp_path: Path) -> ChromaStore:
    """40 itens: tipos alpha/beta em clusters bem separados; cores red/blue independentes do vetor."""
    rng = np.random.default_rng(0)
    ids, vecs, metas = [], [], []
    for i in range(40):
        article = "Alpha" if i % 2 == 0 else "Beta"
        center = np.zeros(DIM, dtype=np.float32)
        center[0 if article == "Alpha" else 1] = 1.0
        v = center + 0.05 * rng.normal(size=DIM).astype(np.float32)
        ids.append(str(i))
        vecs.append(v / np.linalg.norm(v))
        metas.append(
            {
                "product_id": str(i), "article_type": article, "master_category": "Cat",
                "base_colour": "Red" if (i // 2) % 2 == 0 else "Blue",
            }
        )  # fmt: skip
    metas[0]["base_colour"] = "Unknown"
    store = ChromaStore(tmp_path / "vs", "products_image", dim=DIM, fingerprint="fp", model_name="cluster-model")
    store.upsert(ids, np.stack(vecs), metas)
    return store


def test_load_items_maps_unknown_to_none(clustered_store: ChromaStore) -> None:
    items = {i.product_id: i for i in load_items(clustered_store)}
    assert len(items) == 40
    assert items["0"].colour is None and items["0"].article_type == "Alpha"
    assert items["1"] == Item("1", "Beta", "Cat", "Red")


def test_image_to_image_report(clustered_store: ChromaStore) -> None:
    items = load_items(clustered_store)
    r = evaluate_image_to_image(clustered_store, items, k=10, sample=30, seed=1)
    assert r.n_queries == 30
    assert r.metrics["P@5 (tipo)"]["mean"] == 1.0 and r.metrics["MRR (tipo)"]["mean"] == 1.0
    assert r.baselines["P@5 (tipo)"] == pytest.approx(19 / 39)  # 19 outros itens do mesmo tipo entre 39
    strict = r.metrics["P@5 (tipo+cor)"]["mean"]
    assert 0.0 < strict < 1.0  # a cor não está no vetor: só parte dos vizinhos acerta
    assert r.baselines["MRR (tipo)"] is None


def test_text_queries_report_and_markdown(clustered_store: ChromaStore) -> None:
    items = load_items(clustered_store)
    searcher = ProductSearcher(ClusterBackend(), clustered_store, None)
    r = evaluate_text_queries(searcher, items, mode="image", k=10, max_queries=50, min_items=5)
    assert r.n_queries == 4  # (Alpha|Beta) x (Red|Blue), 10 itens cada (Alpha/Unknown cai para 9)
    assert r.metrics["P@5 (tipo)"]["mean"] == 1.0
    assert 0.0 < r.metrics["P@5 (tipo+cor)"]["mean"] < 1.0
    assert r.metrics["HitRate@10 (tipo+cor)"]["mean"] == 1.0

    md = format_markdown([r], model="cluster-model", n_items=40)
    assert "texto → imagem (4 consultas)" in md and "±" in md and "Ganho" in md
