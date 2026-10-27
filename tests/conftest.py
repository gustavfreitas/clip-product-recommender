"""Fixtures compartilhadas: backend falso determinístico e catálogo sintético."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.base import EmbeddingBackend, EmbeddingError  # noqa: E402


class FakeBackend(EmbeddingBackend):
    """Backend determinístico: o vetor depende só do conteúdo (cor média / hash do texto)."""

    def __init__(self, dim: int = 8, fail_on_call: int | None = None) -> None:
        self._dim = dim
        self.fail_on_call = fail_on_call
        self.image_calls = 0
        self.max_images_per_call = 0

    @property
    def name(self) -> str:
        return "fake-model"

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        self.max_images_per_call = max(self.max_images_per_call, len(images))
        if self.fail_on_call is not None and self.image_calls == self.fail_on_call:
            raise EmbeddingError("falha simulada")
        rows = []
        for im in images:
            r, g, b = np.asarray(im.resize((1, 1)), dtype=np.float32).reshape(3)
            v = np.zeros(self._dim, dtype=np.float32)
            v[:3] = (r + 1, g + 1, b + 1)
            rows.append(v)
        return self._unit(np.stack(rows))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            rows.append(rng.normal(size=self._dim).astype(np.float32) + 0.1)
        return self._unit(np.stack(rows))


@pytest.fixture
def fake_catalog(tmp_path: Path) -> pd.DataFrame:
    """Catálogo de 10 itens; o item 3 não tem arquivo de imagem."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(10):
        pid = str(100 + i)
        if i != 3:
            Image.new("RGB", (32, 32), (i * 25, 255 - i * 25, 40)).save(img_dir / f"{pid}.jpg")
        rows.append(
            {
                "product_id": pid,
                "name": f"Produto {pid}",
                "image_path": str(img_dir / f"{pid}.jpg"),  # absoluto: resolve_project_path respeita
                "search_text": f"Produto {pid}. Tshirts, Red",
            }
        )
    return pd.DataFrame(rows)
