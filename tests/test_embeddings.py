"""Testes do Passo 2: batching, persistência, retomada e tolerância a falhas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.batching import iter_catalog_batches, load_catalog
from src.models.base import EmbeddingError
from src.models.catalog_encoder import encode_catalog, iter_embedding_batches, sanity_search
from src.models.embedding_io import EmbeddingStore, EmbeddingStoreError, StoredEmbeddings
from tests.conftest import FakeBackend


def _ids(df: pd.DataFrame) -> list[str]:
    return df["product_id"].tolist()


def test_batches_cover_catalog_without_overlap(fake_catalog: pd.DataFrame) -> None:
    batches = list(iter_catalog_batches(fake_catalog, batch_size=4))
    assert [b.start for b in batches] == [0, 4, 8]
    assert [len(b.product_ids) for b in batches] == [4, 4, 2]
    assert sum((b.product_ids for b in batches), []) == _ids(fake_catalog)


def test_batches_resume_from_start(fake_catalog: pd.DataFrame) -> None:
    batches = list(iter_catalog_batches(fake_catalog, batch_size=4, start=4))
    assert batches[0].product_ids[0] == "104"


def test_batches_reject_bad_args(fake_catalog: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        list(iter_catalog_batches(fake_catalog, batch_size=0))
    with pytest.raises(ValueError):
        list(iter_catalog_batches(fake_catalog, batch_size=4, start=99))


def test_load_catalog_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "nope.parquet")
    pd.DataFrame({"product_id": ["1"]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="Colunas ausentes"):
        load_catalog(tmp_path / "bad.parquet")


def test_encoder_never_exceeds_batch_size_and_flags_missing_image(fake_catalog: pd.DataFrame) -> None:
    backend = FakeBackend()
    out = list(iter_embedding_batches(backend, fake_catalog, batch_size=4, workers=2))
    assert backend.max_images_per_call <= 4
    valid = np.concatenate([b.valid for b in out])
    assert valid.sum() == 9 and not valid[3]
    all_img = np.concatenate([b.image_embeddings for b in out])
    assert np.allclose(all_img[3], 0)  # linha da imagem ausente fica zerada
    norms = np.linalg.norm(all_img[valid], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # L2 normalizado


def test_full_roundtrip_and_valid_mask(fake_catalog: pd.DataFrame, tmp_path: Path) -> None:
    backend = FakeBackend()
    d = tmp_path / "emb"
    store = EmbeddingStore(d, _ids(fake_catalog), backend.name, backend.dim)
    store.open()
    stats = encode_catalog(backend, fake_catalog, store, batch_size=3, workers=2, checkpoint_every=1)
    store.close()
    assert stats["rows_encoded"] == 10 and stats["failed_images"] == 1

    stored = StoredEmbeddings.load(d)
    assert stored.image.shape == (10, backend.dim)
    assert stored.valid.tolist() == [True, True, True, False] + [True] * 6

    seen = [pid for ids, _, _ in stored.iter_batches(4) for pid in ids]
    assert seen == [p for p in _ids(fake_catalog) if p != "103"]  # inválidos nunca vão ao banco vetorial

    names = dict(zip(fake_catalog["product_id"], fake_catalog["name"], strict=True))
    hits = sanity_search(backend, stored, names, "Produto 105. Tshirts, Red", k=3)
    assert len(hits) == 3 and hits[0][2] >= hits[1][2] >= hits[2][2]


def test_resume_after_failure_matches_uninterrupted_run(fake_catalog: pd.DataFrame, tmp_path: Path) -> None:
    ref = FakeBackend()
    ref_store = EmbeddingStore(tmp_path / "ref", _ids(fake_catalog), ref.name, ref.dim)
    ref_store.open()
    encode_catalog(ref, fake_catalog, ref_store, batch_size=3, workers=2)
    ref_store.close()

    d = tmp_path / "resumed"
    flaky = FakeBackend(fail_on_call=3)  # falha no 3º lote
    s1 = EmbeddingStore(d, _ids(fake_catalog), flaky.name, flaky.dim)
    s1.open()
    with pytest.raises(EmbeddingError):
        encode_catalog(flaky, fake_catalog, s1, batch_size=3, workers=2)
    done = s1.completed_rows
    s1.close()
    assert done == 6  # 2 lotes de 3 gravados antes da falha

    ok = FakeBackend()
    s2 = EmbeddingStore(d, _ids(fake_catalog), ok.name, ok.dim)
    assert s2.open() == 6  # retomou do checkpoint
    encode_catalog(ok, fake_catalog, s2, batch_size=3, workers=2)
    s2.close()

    a, b = StoredEmbeddings.load(tmp_path / "ref"), StoredEmbeddings.load(d)
    assert np.allclose(a.image, b.image) and a.valid.tolist() == b.valid.tolist()


def test_fingerprint_mismatch_requires_force(fake_catalog: pd.DataFrame, tmp_path: Path) -> None:
    d = tmp_path / "emb"
    s = EmbeddingStore(d, _ids(fake_catalog), "model-a", 8)
    s.open()
    s.close()
    other = EmbeddingStore(d, _ids(fake_catalog), "model-b", 8)
    with pytest.raises(EmbeddingStoreError, match="--force"):
        other.open()
    assert other.open(force=True) == 0


def test_incomplete_embeddings_cannot_be_loaded(fake_catalog: pd.DataFrame, tmp_path: Path) -> None:
    s = EmbeddingStore(tmp_path / "emb", _ids(fake_catalog), "m", 8)
    s.open()
    s.close()
    with pytest.raises(EmbeddingStoreError, match="incompletos"):
        StoredEmbeddings.load(tmp_path / "emb")


def test_out_of_order_write_rejected(fake_catalog: pd.DataFrame, tmp_path: Path) -> None:
    s = EmbeddingStore(tmp_path / "emb", _ids(fake_catalog), "m", 8)
    s.open()
    z = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(EmbeddingStoreError, match="fora de ordem"):
        s.write_batch(4, z, z, [])
    s.close()
