"""CLI de avaliação offline do índice (P@k, MRR, baseline aleatório e IC 95%).

Experimentos:
    imagem -> imagem       consulta cada produto pelo próprio vetor (não carrega o CLIP)
    texto  -> imagem       "a photo of {cor} {tipo}" contra a coleção de imagens   (carrega o CLIP)
    texto  -> descrição    a mesma consulta contra a coleção de textos            (carrega o CLIP)

Exemplos:
    python scripts/evaluate.py                       # tudo
    python scripts/evaluate.py --skip-text           # só imagem -> imagem (rápido)
    python scripts/evaluate.py --samples 1000 --text-queries 300

Os resultados vão para reports/eval_<modelo>.json e .md. Para comparar modelos, gere
o índice com outro MMREC_CLIP_MODEL_NAME e rode de novo: cada modelo gera seu arquivo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.evaluation.runner import (  # noqa: E402
    EvalReport,
    evaluate_image_to_image,
    evaluate_text_queries,
    format_markdown,
    load_items,
)
from src.logger import configure_logging, get_logger  # noqa: E402
from src.models.base import EmbeddingError  # noqa: E402
from src.recommender.search import ProductSearcher, SearchError  # noqa: E402
from src.vectorstore.base import VectorStoreError  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402

log = get_logger("evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=500, help="Consultas imagem->imagem (padrão: 500).")
    p.add_argument("--text-queries", type=int, default=200, help="Máximo de consultas de texto (padrão: 200).")
    p.add_argument("--min-items", type=int, default=5, help="Mínimo de produtos por par (tipo, cor) para virar consulta.")
    p.add_argument("--k", type=int, default=10, help="Profundidade do ranking (padrão: 10).")
    p.add_argument("--template", default="a photo of {colour} {article_type}", help="Modelo da consulta de texto.")
    p.add_argument("--skip-text", action="store_true", help="Não carrega o CLIP; roda só imagem->imagem.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Prefixo dos arquivos de saída (padrão: reports/eval_<modelo>).")
    args = p.parse_args(argv)
    if args.k < 5:
        p.error("--k deve ser >= 5 (as métricas usam P@5)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    args = parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING", False)
    try:
        store = ChromaStore.open_existing(settings.vectorstore_dir, f"{settings.collection_name}_image")
        items = load_items(store)
        if not items:
            print("[ERRO] O índice está vazio.")
            return 1

        reports: list[EvalReport] = [evaluate_image_to_image(store, items, k=args.k, sample=args.samples, seed=args.seed)]
        if not args.skip_text:
            searcher = ProductSearcher.from_settings(settings)
            modes = ["image"] + (["text"] if searcher.text_store is not None else [])
            for mode in modes:
                reports.append(
                    evaluate_text_queries(
                        searcher,
                        items,
                        mode=mode,
                        k=args.k,
                        max_queries=args.text_queries,
                        min_items=args.min_items,
                        template=args.template,
                        seed=args.seed,
                    )
                )

        markdown = format_markdown(reports, model=store.model_name, n_items=len(items))
        print("\n" + markdown)

        prefix = Path(args.out) if args.out else PROJECT_ROOT / "reports" / f"eval_{re.sub(r'[^A-Za-z0-9._-]+', '_', store.model_name)}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": store.model_name,
            "index_size": len(items),
            "k": args.k,
            "seed": args.seed,
            "reports": [r.to_dict() for r in reports],
        }
        prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        prefix.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
        print(f"\nSalvo em {prefix.with_suffix('.json')} e {prefix.with_suffix('.md')}")
        return 0
    except (VectorStoreError, EmbeddingError, SearchError) as exc:
        log.error("evaluation_failed", error=str(exc))
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
