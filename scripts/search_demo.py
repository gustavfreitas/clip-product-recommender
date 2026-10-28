"""CLI do Passo 4: consulta o índice por texto, imagem ou ambos.

Exemplos:
    python scripts/search_demo.py --text "black running shoes"
    python scripts/search_demo.py --text "red t-shirt" --mode text           # texto -> descrições
    python scripts/search_demo.py --image data/processed/images/1530.jpg --exclude-id 1530
    python scripts/search_demo.py --image foto.jpg --text "in black" --image-weight 0.6
    python scripts/search_demo.py --text "sneakers" --gender Men --max-price 200 --grid resultado.jpg

Carrega o CLIP (o mesmo que gerou o índice), então a primeira execução leva alguns segundos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.logger import configure_logging, get_logger  # noqa: E402
from src.models.base import EmbeddingError  # noqa: E402
from src.recommender.search import ProductSearcher, SearchError, load_query_image  # noqa: E402
from src.recommender.visualize import make_contact_sheet  # noqa: E402
from src.vectorstore.base import Hit, VectorStoreError  # noqa: E402
from src.vectorstore.metadata import build_where  # noqa: E402

log = get_logger("search_demo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="Descrição do produto (inglês funciona melhor no CLIP).")
    p.add_argument("--image", help="Caminho de uma imagem de consulta.")
    p.add_argument("--mode", choices=["image", "text"], default="image", help="Coleção alvo da busca só por texto.")
    p.add_argument("--image-weight", type=float, default=0.5, help="Peso da imagem na busca combinada (0-1).")
    p.add_argument("--k", type=int, default=None, help="Resultados (padrão: MMREC_DEFAULT_TOP_K = 5).")
    p.add_argument("--exclude-id", action="append", default=[], help="Omite um product_id (repetível).")
    p.add_argument("--gender", help="Filtro: gênero (ex.: Men).")
    p.add_argument("--article-type", action="append", help="Filtro: tipo de artigo (repetível).")
    p.add_argument("--base-colour", help="Filtro: cor base.")
    p.add_argument("--min-price", type=float)
    p.add_argument("--max-price", type=float)
    p.add_argument("--grid", help="Salva uma folha de contato com os resultados neste caminho (.jpg/.png).")
    args = p.parse_args(argv)
    if not args.text and not args.image:
        p.error("informe --text, --image ou ambos")
    return args


def print_hits(hits: list[Hit]) -> None:
    """Imprime os resultados em formato tabular."""
    if not hits:
        print("Nenhum resultado (os filtros podem estar restritivos demais).")
        return
    print(f"\n{'#':>2}  {'cos':>6}  {'id':>6}  {'preço':>9}  nome | tipo | cor")
    for i, h in enumerate(hits, 1):
        m = h.metadata
        price = m.get("price")
        price_s = f"R$ {price:.2f}" if isinstance(price, int | float) else "-"
        print(f"{i:>2}  {h.score:6.3f}  {h.product_id:>6}  {price_s:>9}  {m.get('name')} | {m.get('article_type')} | {m.get('base_colour')}")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        searcher = ProductSearcher.from_settings(settings)
        where = build_where(
            min_price=args.min_price,
            max_price=args.max_price,
            gender=args.gender,
            article_type=args.article_type,
            base_colour=args.base_colour,
        )
        common = {"k": args.k, "where": where, "exclude_ids": args.exclude_id}
        if args.image and args.text:
            hits = searcher.search_multimodal(args.image, args.text, image_weight=args.image_weight, **common)
        elif args.image:
            hits = searcher.search_by_image(args.image, **common)
        else:
            hits = searcher.search_by_text(args.text, mode=args.mode, **common)

        print_hits(hits)
        if args.grid:
            query_img = load_query_image(args.image) if args.image else None
            sheet = make_contact_sheet(hits, query_img, caption=f"text={args.text!r}" if args.text else "")
            sheet.save(args.grid)
            print(f"\nFolha de contato salva em {args.grid}")
        return 0
    except (SearchError, VectorStoreError, EmbeddingError, OSError) as exc:
        log.error("search_failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
