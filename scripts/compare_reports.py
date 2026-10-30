"""Compara relatórios de avaliação de vários modelos (gerados por scripts/evaluate.py).

Exemplos:
    python scripts/compare_reports.py                       # todos os reports/eval_*.json
    python scripts/compare_reports.py reports/eval_a.json reports/eval_b.json --out reports/comparacao.md

Para cada métrica, mostra a média ± IC95% de cada modelo e se o melhor se separa do
segundo (intervalos de confiança sem sobreposição). A checagem é conservadora: as
consultas são as mesmas nos dois modelos, então um teste pareado poderia detectar
diferenças menores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_reports(paths: list[Path]) -> list[dict]:
    """Lê os JSONs de avaliação.

    Args:
        paths: Caminhos dos arquivos.

    Returns:
        Lista de dicionários (um por modelo).

    Raises:
        ValueError: Se algum arquivo não existir ou não tiver o formato esperado.
    """
    out = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["model"], data["reports"]  # noqa: B018 - valida as chaves
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Relatório inválido: {p} ({exc})") from exc
        out.append(data)
    return out


def build_comparison(reports: list[dict]) -> str:
    """Monta tabelas Markdown comparando os modelos por experimento e métrica.

    Args:
        reports: Saída de ``load_reports`` (mínimo 2).

    Returns:
        Texto Markdown.
    """
    models = [r["model"] for r in reports]
    sizes = {r["index_size"] for r in reports}
    lines = [f"**Modelos:** {', '.join(f'`{m}`' for m in models)} · **Índice:** {', '.join(str(s) for s in sorted(sizes))} produtos"]
    experiments: list[str] = []
    for r in reports:
        for exp in r["reports"]:
            if exp["name"] not in experiments:
                experiments.append(exp["name"])

    for name in experiments:
        per_model = {}
        for r in reports:
            exp = next((e for e in r["reports"] if e["name"] == name), None)
            if exp:
                per_model[r["model"]] = exp
        if len(per_model) < 2:
            continue
        first = next(iter(per_model.values()))
        lines += ["", f"#### {name}", "", "| Métrica | " + " | ".join(per_model) + " | Melhor | Separado? |", "|---|" + "---|" * (len(per_model) + 2)]
        for label in first["metrics"]:
            cells, stats = [], []
            for model, exp in per_model.items():
                m = exp["metrics"].get(label)
                if m is None:
                    cells.append("-")
                    continue
                cells.append(f"{m['mean']:.3f} ± {m['ci95']:.3f}")
                stats.append((model, m["mean"], m["ci95"]))
            if len(stats) < 2:
                continue
            ranked = sorted(stats, key=lambda t: -t[1])
            (best, bm, bc), (_, sm, sc) = ranked[0], ranked[1]
            separated = bm - bc > sm + sc
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {best if bm != sm else 'empate'} | {'sim' if separated else 'não'} |")
    lines += ["", "_\"Separado\" = os intervalos de confiança de 95% do melhor e do segundo não se sobrepõem._"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", type=Path, help="Relatórios JSON (padrão: reports/eval_*.json).")
    p.add_argument("--out", type=Path, default=None, help="Salva a tabela neste arquivo .md.")
    args = p.parse_args(argv)
    files = args.files or sorted((PROJECT_ROOT / "reports").glob("eval_*.json"))
    if len(files) < 2:
        print("[ERRO] São necessários pelo menos 2 relatórios (rode evaluate.py para cada modelo).")
        return 1
    try:
        text = build_comparison(load_reports(list(files)))
    except ValueError as exc:
        print(f"[ERRO] {exc}")
        return 1
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\nSalvo em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
