"""CLI do Passo 2: gera embeddings CLIP (imagem + texto) do catálogo.

Exemplos:
    python scripts/generate_embeddings.py                 # catálogo inteiro
    python scripts/generate_embeddings.py --batch-size 16 # menos memória
    python scripts/generate_embeddings.py --force         # recria do zero
    python scripts/generate_embeddings.py --sanity-query "black running shoes"

Se for interrompido (Ctrl+C, queda de energia), basta rodar de novo: o script
retoma do último checkpoint.
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
from src.models.base import EmbeddingError  # noqa: E402
from src.models.catalog_encoder import encode_catalog, sanity_search  # noqa: E402
from src.models.clip_embedder import ClipEmbedder  # noqa: E402
from src.models.embedding_io import EmbeddingStore, EmbeddingStoreError, StoredEmbeddings  # noqa: E402

log = get_logger("generate_embeddings")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=None, help="Tamanho do mini-batch (padrão: MMREC_EMBED_BATCH_SIZE).")
    parser.add_argument("--limit", type=int, default=None, help="Usa só os N primeiros produtos (teste rápido).")
    parser.add_argument("--force", action="store_true", help="Descarta embeddings existentes e recria.")
    parser.add_argument("--sanity-query", default="red t-shirt", help="Consulta de validação ao final ('' desativa).")
    return parser.parse_args(argv)


def run(settings: Settings, args: argparse.Namespace) -> int:
    """Executa a geração de embeddings.

    Args:
        settings: Configurações do projeto.
        args: Argumentos da CLI.

    Returns:
        Código de saída (0 = sucesso).
    """
    batch_size = args.batch_size or settings.embed_batch_size
    catalog = load_catalog(settings.catalog_path)
    if args.limit is not None:
        catalog = catalog.head(args.limit).reset_index(drop=True)
        log.info("catalog_limited", rows=len(catalog))

    backend = ClipEmbedder(settings.clip_model_name, settings.device)
    store = EmbeddingStore(
        settings.embeddings_dir, catalog["product_id"].tolist(), backend.name, backend.dim
    )
    already = store.open(force=args.force)
    try:
        if already >= store.n_rows:
            log.info("embeddings_already_complete", rows=store.n_rows)
        else:
            encode_catalog(
                backend,
                catalog,
                store,
                batch_size=batch_size,
                workers=settings.num_workers,
            )
    finally:
        store.close()  # libera os memmaps (obrigatório no Windows antes de reabrir)

    stored = StoredEmbeddings.load(settings.embeddings_dir)
    n_valid = int(stored.valid.sum())
    log.info("embeddings_ready", rows=len(stored.product_ids), valid=n_valid, invalid=len(stored.product_ids) - n_valid, dim=stored.dim)

    if args.sanity_query:
        names = dict(zip(catalog["product_id"], catalog["name"], strict=True))
        print(f"\nSanity check texto->imagem: {args.sanity_query!r}")
        for pid, name, score in sanity_search(backend, stored, names, args.sanity_query):
            print(f"  {score:.3f}  [{pid}] {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return run(settings, args)
    except (EmbeddingError, EmbeddingStoreError, FileNotFoundError, ValueError) as exc:
        log.error("embeddings_failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted_by_user", hint="rode novamente para retomar do checkpoint")
        return 130


if __name__ == "__main__":
    sys.exit(main())
