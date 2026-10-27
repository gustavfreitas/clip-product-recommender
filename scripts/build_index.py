"""CLI do Passo 3: indexa os embeddings (Passo 2) + metadados no ChromaDB.

Cria duas coleções com o mesmo conjunto de IDs e metadados:
    {prefixo}_image  -> embeddings de imagem (alvo da busca por imagem e texto->imagem)
    {prefixo}_text   -> embeddings de texto  (alvo da busca texto->texto)

Exemplos:
    python scripts/build_index.py                       # indexa tudo (idempotente)
    python scripts/build_index.py --recreate            # apaga e reconstrói do zero
    python scripts/build_index.py --modalities image    # só a coleção de imagens
    python scripts/build_index.py --verify-sample 500   # verificação mais ampla

Este passo não carrega o modelo CLIP: só lê os vetores já gerados.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.data.batching import load_catalog  # noqa: E402
from src.logger import configure_logging, get_logger  # noqa: E402
from src.models.embedding_io import EmbeddingStoreError, StoredEmbeddings  # noqa: E402
from src.vectorstore.base import VectorStoreError  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402
from src.vectorstore.indexer import index_embeddings, verify_self_retrieval  # noqa: E402
from src.vectorstore.metadata import build_where  # noqa: E402

log = get_logger("build_index")
MODALITIES = ("image", "text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="Apaga as coleções existentes e reconstrói.")
    parser.add_argument(
        "--modalities",
        default="image,text",
        help="Coleções a indexar, separadas por vírgula (image,text).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Itens por upsert (padrão: MMREC_INDEX_BATCH_SIZE).")
    parser.add_argument("--verify-sample", type=int, default=200, help="Itens usados na verificação de recall (0 desativa).")
    return parser.parse_args(argv)


def run(settings: Settings, args: argparse.Namespace) -> int:
    """Executa a indexação e a verificação.

    Args:
        settings: Configurações do projeto.
        args: Argumentos da CLI.

    Returns:
        Código de saída (0 = sucesso).
    """
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    invalid = [m for m in modalities if m not in MODALITIES]
    if invalid or not modalities:
        raise ValueError(f"--modalities inválido: {invalid or args.modalities!r}. Use: {','.join(MODALITIES)}")

    catalog = load_catalog(settings.catalog_path)
    stored = StoredEmbeddings.load(settings.embeddings_dir)
    log.info("embeddings_loaded", rows=len(stored.product_ids), valid=int(stored.valid.sum()), dim=stored.dim, model=stored.model_name)

    stores: dict = {}
    try:
        for modality in modalities:
            stores[modality] = ChromaStore(
                settings.vectorstore_dir,
                f"{settings.collection_name}_{modality}",
                dim=stored.dim,
                fingerprint=stored.fingerprint,
                model_name=stored.model_name,
                recreate=args.recreate,
            )

        stats = index_embeddings(stored, catalog, stores, batch_size=args.batch_size or settings.index_batch_size)
        print(f"\nIndexados: {stats.indexed} itens/coleção | ignorados (imagem inválida): {stats.skipped_invalid} | {stats.seconds:.1f}s")

        image_store = stores.get("image") or next(iter(stores.values()))
        if args.verify_sample > 0:
            report = verify_self_retrieval(image_store, stored, sample=args.verify_sample)
            print(
                f"Auto-recuperação ({report['n']} itens): recall@1={report['recall_at_1']:.3f} "
                f"recall@5={report['recall_at_k']:.3f} cosseno_top1_médio={report['mean_top1_score']:.4f}"
            )
            if report["recall_at_1"] < 0.95:
                log.warning("low_self_retrieval_recall", **report, hint="índice HNSW pode estar perdendo vizinhos")

        _demo_filtered_query(image_store, stored, catalog)
    finally:
        for store in stores.values():
            store.close()
    return 0


def _demo_filtered_query(store: ChromaStore, stored: StoredEmbeddings, catalog) -> None:  # type: ignore[no-untyped-def]
    """Mostra uma consulta com filtro de metadados, usando o 1º produto como query."""
    first_valid = next((pid for pid, ok in zip(stored.product_ids, stored.valid, strict=True) if ok), None)
    if first_valid is None:
        return
    row = catalog.loc[catalog["product_id"] == first_valid].iloc[0]
    where = build_where(gender=str(row["gender"]), max_price=float(row["price"]) * 1.5)
    query = store.get_embeddings([first_valid])[first_valid]
    hits = store.query(query, k=5, where=where)
    print(f"\nDemo: vizinhos de [{first_valid}] {row['name']!r} com filtro gender={row['gender']}, preço <= {float(row['price']) * 1.5:.2f}")
    for h in hits:
        m = h.metadata
        print(f"  {h.score:.3f}  [{h.product_id}] {m.get('name')} | {m.get('article_type')} | R$ {m.get('price', float('nan')):.2f}")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return run(settings, args)
    except (VectorStoreError, EmbeddingStoreError, FileNotFoundError, ValueError) as exc:
        log.error("index_failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted_by_user", hint="rode novamente: o upsert é idempotente")
        return 130


if __name__ == "__main__":
    sys.exit(main())
