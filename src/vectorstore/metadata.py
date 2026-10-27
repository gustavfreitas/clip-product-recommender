"""Conversão de linhas do catálogo em metadados do Chroma e construção de filtros."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Colunas do catálogo que viram metadados (o Chroma só aceita str/int/float/bool).
STRING_FIELDS = (
    "name", "gender", "master_category", "sub_category", "article_type",
    "base_colour", "season", "usage", "image_path",
)  # fmt: skip
CATEGORICAL_FILTERS = ("gender", "master_category", "sub_category", "article_type", "base_colour", "season", "usage")


def row_to_metadata(row: pd.Series) -> dict[str, Any]:
    """Converte uma linha do catálogo em metadados válidos para o Chroma.

    Valores ausentes (``NA``/``NaN``/``None``) são omitidos, pois o Chroma
    rejeita ``None``. Tipos numpy/pandas são convertidos para tipos Python.

    Args:
        row: Linha do catálogo (índice por nome de coluna).

    Returns:
        Dicionário com ``product_id``, campos textuais, ``price``, ``year`` e
        ``price_is_synthetic`` quando disponíveis.
    """
    meta: dict[str, Any] = {"product_id": str(row["product_id"])}
    for col in STRING_FIELDS:
        value = row.get(col)
        if value is not None and not pd.isna(value):
            meta[col] = str(value)
    price = row.get("price")
    if price is not None and not pd.isna(price):
        meta["price"] = float(price)
    year = row.get("year")
    if year is not None and not pd.isna(year):
        meta["year"] = int(year)
    synthetic = row.get("price_is_synthetic")
    if synthetic is not None and not pd.isna(synthetic):
        meta["price_is_synthetic"] = bool(synthetic)
    return meta


def build_where(
    *,
    min_price: float | None = None,
    max_price: float | None = None,
    **categorical: str | list[str] | None,
) -> dict[str, Any] | None:
    """Monta o filtro ``where`` do Chroma a partir de critérios simples.

    Exemplo::

        build_where(gender="Men", article_type=["Shoes", "Sandals"], max_price=300)

    Args:
        min_price: Preço mínimo (inclusivo).
        max_price: Preço máximo (inclusivo).
        **categorical: Filtros de igualdade (str) ou de pertencimento (lista)
            em ``CATEGORICAL_FILTERS``. Valores ``None`` ou lista vazia são ignorados.

    Returns:
        Filtro pronto para o Chroma, ou ``None`` se não houver critérios.

    Raises:
        ValueError: Se um campo categórico desconhecido for informado.
    """
    clauses: list[dict[str, Any]] = []
    for field_name, value in categorical.items():
        if field_name not in CATEGORICAL_FILTERS:
            raise ValueError(f"Filtro desconhecido: {field_name!r}. Válidos: {CATEGORICAL_FILTERS}")
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, list):
            clauses.append({field_name: {"$in": value}})
        else:
            clauses.append({field_name: {"$eq": value}})
    if min_price is not None:
        clauses.append({"price": {"$gte": float(min_price)}})
    if max_price is not None:
        clauses.append({"price": {"$lte": float(max_price)}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
