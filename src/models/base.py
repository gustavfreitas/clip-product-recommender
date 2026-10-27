"""Interface comum dos backends de embedding (CLIP, Gemini, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from PIL import Image


class EmbeddingError(RuntimeError):
    """Falha ao carregar o modelo ou gerar embeddings."""


class EmbeddingBackend(ABC):
    """Contrato de um modelo que projeta imagem e texto no mesmo espaço vetorial.

    Todo backend deve devolver ``float32`` de formato ``(n, dim)`` já
    normalizado em L2. Com vetores unitários, o cosseno vira um produto
    interno, o que simplifica FAISS (``IndexFlatIP``) e a análise de scores.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador do modelo (usado no fingerprint dos embeddings)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensão do vetor de saída."""

    @abstractmethod
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Gera embeddings normalizados para imagens RGB.

        Args:
            images: Imagens PIL já em RGB.

        Returns:
            Array ``float32`` de formato ``(len(images), dim)``.
        """

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Gera embeddings normalizados para textos.

        Args:
            texts: Textos de entrada.

        Returns:
            Array ``float32`` de formato ``(len(texts), dim)``.
        """
