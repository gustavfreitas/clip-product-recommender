"""Métricas de ranking e utilitários estatísticos (funções puras)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    """Fração de itens relevantes entre as ``k`` primeiras posições.

    O denominador é sempre ``k``: se o buscador devolver menos de ``k``
    resultados, as posições ausentes contam como não relevantes.

    Args:
        relevant: Relevância de cada resultado, em ordem de ranking.
        k: Profundidade (>= 1).

    Returns:
        Precisão em [0, 1].

    Raises:
        ValueError: Se ``k < 1``.
    """
    if k < 1:
        raise ValueError("k deve ser >= 1")
    return sum(relevant[:k]) / k


def hit_at_k(relevant: Sequence[bool], k: int) -> float:
    """1.0 se há ao menos um item relevante nas ``k`` primeiras posições, senão 0.0."""
    if k < 1:
        raise ValueError("k deve ser >= 1")
    return 1.0 if any(relevant[:k]) else 0.0


def reciprocal_rank(relevant: Sequence[bool]) -> float:
    """Inverso da posição (base 1) do primeiro item relevante; 0.0 se não houver."""
    for rank, rel in enumerate(relevant, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Resume valores por consulta em média e intervalo de confiança de 95%.

    O intervalo usa a aproximação normal (``1,96 · desvio / √n``), adequada
    para centenas de consultas.

    Args:
        values: Métrica de cada consulta.

    Returns:
        ``{"mean", "ci95", "n"}``; com ``n == 0`` tudo é 0.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci95": 0.0, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "ci95": 0.0, "n": 1}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "ci95": 1.96 * math.sqrt(var / n), "n": n}
