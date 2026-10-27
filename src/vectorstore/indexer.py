"""Indexação dos embeddings persistidos no banco vetorial, em lotes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.logger import get_logger
from src.models.embedding_io import StoredEmbeddings
from src.vectorstore.base import VectorStore, VectorStoreError
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.metadata import row_to_metadata

log = get_logger(__name__)

Modality = Literal["image", "text"]


@dataclass(frozen=True)
class IndexStats:
    """Resultado de uma indexação.

    Attributes:
        indexed: Itens enviados ao banco (por modalidade).
        skipped_invalid: Itens ignorados por imagem ilegível no Passo 2.
        missing_metadata: Itens sem linha correspondente no catálogo (ignorados).
        seconds: Duração total.
    """

    indexed: int
    skipped_invalid: int
    missing_metadata: int
    seconds: float


def index_embeddings(
    stored: StoredEmbeddings,
    catalog: pd.DataFrame,
    stores: dict[Modality, VectorStore],
    *,
    batch_size: int = 1024,
) -> IndexStats:
    """Envia embeddings + metadados ao banco vetorial, lote a lote.

    Lê os vetores do disco (memmap) em blocos de ``batch_size`` e faz ``upsert``,
    então a RAM usada é proporcional ao lote, não ao catálogo. A operação é
    idempotente: reexecutar atualiza os mesmos IDs sem duplicar.

    Args:
        stored: Embeddings do Passo 2 (completos).
        catalog: Catálogo limpo com os metadados.
        stores: Mapa modalidade -> banco vetorial a preencher.
        batch_size: Itens por ``upsert``.

    Returns:
        Estatísticas da indexação.

    Raises:
        VectorStoreError: Se a contagem final divergir do esperado.
    """
    if not stores:
        raise VectorStoreError("Nenhuma modalidade selecionada para indexar.")
    t0 = time.perf_counter()
    by_id = catalog.drop_duplicates("product_id").set_index("product_id", drop=False)
    n_valid = int(stored.valid.sum())
    indexed = missing = 0

    with tqdm(total=n_valid, desc="Indexando", unit="item") as bar:
        for ids, img, txt in stored.iter_batches(batch_size):
            known = [pid in by_id.index for pid in ids]
            if not all(known):
                keep = np.flatnonzero(known)
                missing += len(ids) - len(keep)
                ids = [ids[i] for i in keep]
                img, txt = img[keep], txt[keep]
            if not ids:
                bar.update(len(known))
                continue

            rows = by_id.loc[ids]
            metadatas = [row_to_metadata(r) for _, r in rows.iterrows()]
            documents = rows["search_text"].astype(str).tolist() if "search_text" in rows else None
            vectors = {"image": img, "text": txt}
            for modality, store in stores.items():
                store.upsert(ids, vectors[modality], metadatas, documents)

            indexed += len(ids)
            bar.update(len(known))

    expected = n_valid - missing
    for modality, store in stores.items():
        if store.count() != expected:
            raise VectorStoreError(
                f"Coleção '{modality}' tem {store.count()} itens, esperado {expected}. "
                "Se havia dados antigos, rode com --recreate."
            )

    seconds = time.perf_counter() - t0
    stats = IndexStats(indexed, len(stored.product_ids) - n_valid, missing, seconds)
    log.info("indexing_finished", modalities=list(stores), **stats.__dict__)
    return stats


def verify_self_retrieval(
    store: ChromaStore,
    stored: StoredEmbeddings,
    *,
    sample: int = 200,
    k: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Mede a saúde do índice HNSW consultando itens pelo próprio vetor.

    Cada item amostrado, usado como consulta, deveria voltar como vizinho nº 1
    com cosseno ≈ 1.0. Se ``recall@1`` cair muito, o índice aproximado está
    perdendo vizinhos (ajuste ``ef_search``) ou o espaço de distância está errado.
    Duplicatas visuais exatas empatam em ~1.0 e também contam como acerto.

    Args:
        store: Coleção a verificar.
        stored: Embeddings de origem (define os IDs válidos).
        sample: Quantidade de itens amostrados.
        k: Vizinhos pedidos por consulta.
        seed: Seed da amostragem.

    Returns:
        Dicionário com ``n``, ``recall_at_1``, ``recall_at_k`` e ``mean_top1_score``.
    """
    valid_ids = [pid for pid, ok in zip(stored.product_ids, stored.valid, strict=True) if ok]
    if not valid_ids:
        return {"n": 0, "recall_at_1": 0.0, "recall_at_k": 0.0, "mean_top1_score": 0.0}
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(valid_ids), size=min(sample, len(valid_ids)), replace=False)
    ids = [valid_ids[i] for i in picked]
    vectors = store.get_embeddings(ids)

    hit1 = hitk = 0
    top1_scores: list[float] = []
    for pid in ids:
        hits = store.query(vectors[pid], k=k)
        if not hits:
            continue
        top1_scores.append(hits[0].score)
        hit1 += int(hits[0].product_id == pid or hits[0].score >= 0.9999)
        hitk += int(any(h.product_id == pid for h in hits))
    n = len(ids)
    return {
        "n": n,
        "recall_at_1": hit1 / n,
        "recall_at_k": hitk / n,
        "mean_top1_score": float(np.mean(top1_scores)) if top1_scores else 0.0,
    }
