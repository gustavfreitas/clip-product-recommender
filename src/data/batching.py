"""Leitura do catálogo limpo e geração de lotes (mini-batches)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import resolve_project_path
from src.logger import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("product_id", "name", "image_path", "search_text")


@dataclass(frozen=True)
class CatalogBatch:
    """Um lote contíguo de produtos do catálogo.

    Attributes:
        start: Índice (posição) da primeira linha do lote no catálogo.
        product_ids: IDs dos produtos.
        image_paths: Caminhos absolutos das imagens padronizadas.
        texts: Textos descritivos (``search_text``) para o encoder de texto.
    """

    start: int
    product_ids: list[str]
    image_paths: list[Path]
    texts: list[str]


def load_catalog(path: Path) -> pd.DataFrame:
    """Lê o catálogo limpo gerado no Passo 1.

    Args:
        path: Caminho do ``catalog.parquet``.

    Returns:
        DataFrame com índice posicional 0..N-1.

    Raises:
        FileNotFoundError: Se o catálogo não existir.
        ValueError: Se faltarem colunas obrigatórias ou o catálogo estiver vazio.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Catálogo não encontrado em {path}. Rode antes: python scripts/prepare_data.py")
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no catálogo: {missing}")
    if df.empty:
        raise ValueError("Catálogo vazio.")
    df = df.reset_index(drop=True)
    df["product_id"] = df["product_id"].astype(str)
    log.info("catalog_loaded", rows=len(df), path=str(path))
    return df


def iter_catalog_batches(catalog: pd.DataFrame, batch_size: int, start: int = 0) -> Iterator[CatalogBatch]:
    """Percorre o catálogo em lotes sem materializar imagens ou vetores.

    Args:
        catalog: Catálogo com ``product_id``, ``image_path`` e ``search_text``.
        batch_size: Itens por lote.
        start: Posição inicial (usada para retomar uma execução interrompida).

    Yields:
        ``CatalogBatch`` em ordem. O último lote pode ser menor.

    Raises:
        ValueError: Se ``batch_size < 1`` ou ``start`` for inválido.
    """
    if batch_size < 1:
        raise ValueError("batch_size deve ser >= 1")
    if not 0 <= start <= len(catalog):
        raise ValueError(f"start={start} fora do intervalo [0, {len(catalog)}]")

    ids = catalog["product_id"].astype(str).tolist()
    paths = catalog["image_path"].tolist()
    texts = catalog["search_text"].astype(str).tolist()

    for lo in range(start, len(ids), batch_size):
        hi = min(lo + batch_size, len(ids))
        yield CatalogBatch(
            start=lo,
            product_ids=ids[lo:hi],
            image_paths=[resolve_project_path(p) for p in paths[lo:hi]],
            texts=texts[lo:hi],
        )
