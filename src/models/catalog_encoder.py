"""Geração de embeddings do catálogo em mini-batches (sem estourar memória)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.batching import iter_catalog_batches
from src.logger import get_logger
from src.models.base import EmbeddingBackend, EmbeddingError
from src.models.embedding_io import EmbeddingStore, StoredEmbeddings

log = get_logger(__name__)


@dataclass(frozen=True)
class EncodedBatch:
    """Embeddings de um lote.

    Attributes:
        start: Posição da primeira linha do lote no catálogo.
        product_ids: IDs dos produtos.
        image_embeddings: ``(n, dim)``; linhas de imagens ilegíveis ficam zeradas.
        text_embeddings: ``(n, dim)``.
        valid: Máscara booleana de imagens lidas com sucesso.
    """

    start: int
    product_ids: list[str]
    image_embeddings: np.ndarray
    text_embeddings: np.ndarray
    valid: np.ndarray


def _open_rgb(path: Path) -> Image.Image | None:
    """Abre uma imagem como RGB; devolve None (com log) se falhar."""
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except (OSError, ValueError) as exc:  # UnidentifiedImageError herda de OSError
        log.warning("image_load_failed", path=str(path), error=str(exc))
        return None


def iter_embedding_batches(
    backend: EmbeddingBackend,
    catalog: pd.DataFrame,
    *,
    batch_size: int = 32,
    start: int = 0,
    workers: int = 4,
) -> Iterator[EncodedBatch]:
    """Gerador que embeda o catálogo lote a lote.

    Enquanto o modelo processa o lote ``i``, threads já decodificam as imagens
    do lote ``i+1`` (prefetch), então CPU de I/O e de inferência se sobrepõem.
    No máximo dois lotes de imagens ficam em RAM, independentemente do
    tamanho do catálogo.

    Args:
        backend: Modelo de embedding.
        catalog: Catálogo limpo.
        batch_size: Itens por lote.
        start: Posição inicial (retomada).
        workers: Threads de leitura de imagem.

    Yields:
        ``EncodedBatch`` em ordem de catálogo.

    Raises:
        EmbeddingError: Se o backend falhar (ex.: sem memória no device).
    """
    batches = iter_catalog_batches(catalog, batch_size, start=start)
    current = next(batches, None)
    if current is None:
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit(batch):  # type: ignore[no-untyped-def]
            return [pool.submit(_open_rgb, p) for p in batch.image_paths]

        futures = submit(current)
        while current is not None:
            upcoming = next(batches, None)
            upcoming_futures = submit(upcoming) if upcoming is not None else []

            images = [f.result() for f in futures]
            valid = np.array([im is not None for im in images], dtype=bool)
            image_emb = np.zeros((len(images), backend.dim), dtype=np.float32)
            if valid.any():
                image_emb[valid] = backend.embed_images([im for im in images if im is not None])
            text_emb = backend.embed_texts(current.texts)
            del images

            yield EncodedBatch(current.start, current.product_ids, image_emb, text_emb, valid)
            current, futures = upcoming, upcoming_futures


def encode_catalog(
    backend: EmbeddingBackend,
    catalog: pd.DataFrame,
    store: EmbeddingStore,
    *,
    batch_size: int = 32,
    workers: int = 4,
    checkpoint_every: int = 20,
) -> dict[str, float]:
    """Embeda o catálogo e persiste com checkpoints periódicos.

    Retoma automaticamente de ``store.completed_rows``. Se algo falhar, o
    progresso até o último lote gravado é preservado.

    Args:
        backend: Modelo de embedding.
        catalog: Catálogo limpo.
        store: Store já aberto (``store.open()``).
        batch_size: Itens por lote.
        workers: Threads de leitura de imagem.
        checkpoint_every: Faz flush + meta a cada N lotes.

    Returns:
        Estatísticas: linhas codificadas, imagens inválidas e duração.

    Raises:
        EmbeddingError: Propaga falhas do backend após registrar o progresso.
    """
    start = store.completed_rows
    t0 = time.perf_counter()
    encoded = failed = 0
    try:
        with tqdm(total=len(catalog), initial=start, desc="Embeddings", unit="item") as bar:
            gen = iter_embedding_batches(backend, catalog, batch_size=batch_size, start=start, workers=workers)
            for i, batch in enumerate(gen, 1):
                bad = [pid for pid, ok in zip(batch.product_ids, batch.valid, strict=True) if not ok]
                store.write_batch(batch.start, batch.image_embeddings, batch.text_embeddings, bad)
                encoded += len(batch.product_ids)
                failed += len(bad)
                bar.update(len(batch.product_ids))
                if i % checkpoint_every == 0:
                    store.checkpoint()
    except EmbeddingError as exc:
        log.error("encoding_aborted", error=str(exc), completed=store.completed_rows, total=len(catalog))
        raise
    finally:
        store.checkpoint()

    seconds = time.perf_counter() - t0
    log.info("encoding_finished", rows_encoded=encoded, failed_images=failed, seconds=round(seconds, 1))
    return {"rows_encoded": encoded, "failed_images": failed, "seconds": seconds}


def sanity_search(
    backend: EmbeddingBackend,
    stored: StoredEmbeddings,
    names: dict[str, str],
    query: str,
    k: int = 5,
    chunk: int = 8192,
) -> list[tuple[str, str, float]]:
    """Busca texto→imagem por força bruta, só para validar os embeddings.

    Não substitui o banco vetorial (Passo 3): serve para checar se o espaço
    multimodal está coerente antes de indexar.

    Args:
        backend: Modelo (embeda a query).
        stored: Embeddings persistidos.
        names: Mapa ``product_id -> nome``.
        query: Texto de busca.
        k: Quantidade de resultados.
        chunk: Linhas lidas do disco por vez.

    Returns:
        Lista ``(product_id, nome, cosseno)`` em ordem decrescente de score.
    """
    q = backend.embed_texts([query])[0]
    candidates: list[tuple[str, float]] = []
    for ids, img, _ in stored.iter_batches(chunk):
        scores = img @ q  # vetores unitários: produto interno == cosseno
        top = np.argsort(-scores)[:k]
        candidates.extend((ids[j], float(scores[j])) for j in top)
    candidates.sort(key=lambda t: -t[1])
    return [(pid, names.get(pid, "?"), s) for pid, s in candidates[:k]]
