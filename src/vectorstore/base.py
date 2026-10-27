"""Contrato do banco vetorial e tipos de resultado."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class VectorStoreError(RuntimeError):
    """Falha de configuração, escrita ou consulta no banco vetorial."""


@dataclass(frozen=True)
class Hit:
    """Um resultado de busca por similaridade.

    Attributes:
        product_id: ID do produto.
        score: Similaridade de cosseno em [-1, 1] (maior = mais parecido).
        metadata: Metadados do produto (nome, preço, caminho da imagem...).
    """

    product_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Interface mínima de um índice vetorial por similaridade de cosseno."""

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
        documents: Sequence[str] | None = None,
    ) -> None:
        """Insere ou atualiza itens (idempotente por ``id``)."""

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Retorna os ``k`` vizinhos mais próximos, em ordem decrescente de score."""

    @abstractmethod
    def count(self) -> int:
        """Quantidade de itens indexados."""
