"""CLI do Passo 1: baixa, limpa e padroniza o dataset de produtos.

Exemplos:
    python scripts/prepare_data.py                      # fluxo completo
    python scripts/prepare_data.py --skip-download      # dataset já está em data/raw
    python scripts/prepare_data.py --limit 2000         # amostra para testes rápidos
    python scripts/prepare_data.py --force-images       # reprocessa todas as imagens
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.data.loader import (  # noqa: E402
    DatasetDownloadError,
    RawLayoutError,
    download_dataset,
    load_styles,
    locate_raw_layout,
)
from src.data.preprocessing import (  # noqa: E402
    add_synthetic_price,
    attach_image_paths,
    clean_catalog,
    process_images,
)
from src.logger import configure_logging, get_logger  # noqa: E402

log = get_logger("prepare_data")

OUTPUT_COLUMNS = [
    "product_id", "name", "gender", "master_category", "sub_category",
    "article_type", "base_colour", "season", "year", "usage",
    "price", "price_is_synthetic", "image_path", "search_text",
]  # fmt: skip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true", help="Não baixa; usa o que já está em raw_dir.")
    parser.add_argument("--force-download", action="store_true", help="Baixa novamente mesmo se já existir.")
    parser.add_argument("--force-images", action="store_true", help="Reprocessa imagens já geradas.")
    parser.add_argument("--limit", type=int, default=None, help="Usa apenas N produtos (amostra reprodutível).")
    return parser.parse_args(argv)


def run(settings: Settings, args: argparse.Namespace) -> int:
    """Executa o pipeline completo.

    Args:
        settings: Configurações do projeto.
        args: Argumentos da CLI.

    Returns:
        Código de saída (0 = sucesso).
    """
    # 1) Download / localização do dataset bruto
    if not args.skip_download:
        download_dataset(settings.kaggle_dataset, settings.raw_dir, force=args.force_download)
    styles_csv, raw_images_dir = locate_raw_layout(settings.raw_dir)
    log.info("raw_layout_found", styles=str(styles_csv), images=str(raw_images_dir))

    # 2) Leitura + limpeza
    raw = load_styles(styles_csv)
    catalog, removed = clean_catalog(raw)
    if args.limit is not None:
        catalog = catalog.sample(n=min(args.limit, len(catalog)), random_state=settings.price_seed)
        log.info("catalog_sampled", rows=len(catalog))

    # 3) Padronização das imagens (valida e descarta produtos sem imagem utilizável)
    ok_ids, image_failures = process_images(
        catalog["product_id"].tolist(),
        raw_images_dir,
        settings.processed_images_dir,
        max_side=settings.max_image_side,
        min_side=settings.min_image_side,
        quality=settings.jpeg_quality,
        workers=settings.num_workers,
        force=args.force_images,
    )
    catalog = catalog[catalog["product_id"].isin(ok_ids)]
    if catalog.empty:
        log.error("no_valid_products", hint="Verifique se as imagens estão em raw_dir/images/{id}.jpg")
        return 1

    # 4) Metadados finais e persistência
    catalog = add_synthetic_price(catalog, seed=settings.price_seed)
    catalog = attach_image_paths(catalog, settings.processed_images_dir)[OUTPUT_COLUMNS]

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(settings.catalog_path, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": settings.kaggle_dataset,
        "rows_raw": len(raw),
        "rows_final": len(catalog),
        "removed_by_cleaning": removed,
        "removed_by_image_validation": image_failures,
        "catalog_path": str(settings.catalog_path),
    }
    settings.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("pipeline_finished", **{k: v for k, v in report.items() if k != "generated_at"})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return run(settings, args)
    except (DatasetDownloadError, RawLayoutError, ValueError) as exc:
        log.error("pipeline_failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted_by_user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
