"""Experimentos de avaliação: imagem→imagem e texto→imagem/texto.

A "verdade" vem dos metadados do catálogo: dois produtos são relevantes entre si
se tiverem o mesmo tipo de artigo (``article_type``), a mesma categoria
(``master_category``) ou o mesmo tipo *e* a mesma cor (critério estrito).
É um proxy: itens de tipos diferentes podem ser, na prática, substitutos
razoáveis, então os números medem coerência com o catálogo, não satisfação real.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.metrics import hit_at_k, precision_at_k, reciprocal_rank, summarize
from src.logger import get_logger
from src.recommender.search import ProductSearcher
from src.vectorstore.base import Hit
from src.vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)

UNKNOWN = "Unknown"
FETCH_CHUNK = 500


@dataclass(frozen=True)
class Item:
    """Rótulos de um produto indexado (``None`` quando ausentes ou ``Unknown``)."""

    product_id: str
    article_type: str | None
    category: str | None
    colour: str | None


@dataclass
class EvalReport:
    """Resultado de um experimento.

    Attributes:
        name: Nome do experimento.
        n_queries: Consultas executadas.
        metrics: Métrica -> ``{"mean", "ci95", "n"}``.
        baselines: Métrica -> precisão esperada de um ranking aleatório (``None`` se não se aplica).
    """

    name: str
    n_queries: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    baselines: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa para JSON."""
        return {"name": self.name, "n_queries": self.n_queries, "metrics": self.metrics, "baselines": self.baselines}


def _label(value: object) -> str | None:
    text = None if value is None else str(value)
    return None if text in (None, "", UNKNOWN) else text


def load_items(store: ChromaStore) -> list[Item]:
    """Lê os rótulos de todos os produtos do índice.

    Args:
        store: Coleção indexada.

    Returns:
        Lista de itens, na ordem do banco.
    """
    items: list[Item] = []
    for page in store.iter_metadatas():
        for m in page:
            items.append(
                Item(
                    str(m.get("product_id")),
                    _label(m.get("article_type")),
                    _label(m.get("master_category")),
                    _label(m.get("base_colour")),
                )
            )
    return items


class _Spec:
    """Define uma métrica: como calcular a partir das relevâncias e qual baseline usar."""

    def __init__(self, label: str, rel: str, fn: Callable[[Sequence[bool]], float], baseline: str | None) -> None:
        self.label, self.rel, self.fn, self.baseline = label, rel, fn, baseline


def _p(k: int) -> Callable[[Sequence[bool]], float]:
    return lambda rels: precision_at_k(rels, k)


def _summarize_specs(
    specs: Sequence[_Spec],
    rels: dict[str, list[Sequence[bool]]],
    base_rates: dict[str, list[float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float | None]]:
    """Calcula média/IC de cada métrica e o baseline aleatório correspondente."""
    metrics: dict[str, dict[str, float]] = {}
    baselines: dict[str, float | None] = {}
    for s in specs:
        metrics[s.label] = summarize([s.fn(r) for r in rels[s.rel]])
        rates = base_rates.get(s.baseline, []) if s.baseline else []
        baselines[s.label] = sum(rates) / len(rates) if rates else None
    return metrics, baselines


def evaluate_image_to_image(
    store: ChromaStore,
    items: Sequence[Item],
    *,
    k: int = 10,
    sample: int = 500,
    seed: int = 0,
) -> EvalReport:
    """Consulta cada produto amostrado pelo próprio vetor de imagem (sem o CLIP).

    O próprio item é removido do ranking. Como a consulta usa o vetor já
    armazenado, o experimento é rápido e determinístico.

    Args:
        store: Coleção de imagens.
        items: Rótulos de todo o índice (``load_items``).
        k: Profundidade do ranking.
        sample: Número de consultas.
        seed: Seed da amostragem.

    Returns:
        Relatório com P@1/5/10 por tipo, MRR, P@5 por categoria e P@5 estrito.
    """
    total = len(items)
    n_type = Counter(i.article_type for i in items if i.article_type)
    n_cat = Counter(i.category for i in items if i.category)
    n_pair = Counter((i.article_type, i.colour) for i in items if i.article_type and i.colour)
    pool = [i for i in items if i.article_type]
    queries = random.Random(seed).sample(pool, min(sample, len(pool)))
    by_id = {i.product_id: i for i in items}

    rels: dict[str, list[Sequence[bool]]] = {"type": [], "cat": [], "strict": []}
    rates: dict[str, list[float]] = {"type": [], "cat": [], "strict": []}
    for lo in range(0, len(queries), FETCH_CHUNK):
        chunk = queries[lo : lo + FETCH_CHUNK]
        vectors = store.get_embeddings([q.product_id for q in chunk])
        for q in chunk:
            vec = vectors.get(q.product_id)
            if vec is None:
                continue
            hits = [h for h in store.query(vec, k=k + 1) if h.product_id != q.product_id][:k]
            hit_items = [by_id.get(h.product_id) for h in hits]
            rels["type"].append([bool(it and it.article_type == q.article_type) for it in hit_items])
            rates["type"].append((n_type[q.article_type] - 1) / max(total - 1, 1))
            if q.category:
                rels["cat"].append([bool(it and it.category == q.category) for it in hit_items])
                rates["cat"].append((n_cat[q.category] - 1) / max(total - 1, 1))
            if q.colour:
                rels["strict"].append([bool(it and it.article_type == q.article_type and it.colour == q.colour) for it in hit_items])
                rates["strict"].append((n_pair[(q.article_type, q.colour)] - 1) / max(total - 1, 1))

    specs = [
        _Spec("P@1 (tipo)", "type", _p(1), "type"),
        _Spec("P@5 (tipo)", "type", _p(5), "type"),
        _Spec(f"P@{k} (tipo)", "type", _p(k), "type"),
        _Spec("MRR (tipo)", "type", reciprocal_rank, None),
        _Spec("P@5 (categoria)", "cat", _p(5), "cat"),
        _Spec("P@5 (tipo+cor)", "strict", _p(5), "strict"),
    ]
    metrics, baselines = _summarize_specs(specs, rels, rates)
    log.info("eval_image_to_image_done", queries=len(rels["type"]))
    return EvalReport("imagem → imagem", len(rels["type"]), metrics, baselines)


def evaluate_text_queries(
    searcher: ProductSearcher,
    items: Sequence[Item],
    *,
    mode: str,
    k: int = 10,
    max_queries: int = 200,
    min_items: int = 5,
    template: str = "a photo of {colour} {article_type}",
    seed: int = 0,
) -> EvalReport:
    """Avalia consultas de texto geradas a partir dos pares (tipo, cor) do catálogo.

    Cada par com pelo menos ``min_items`` produtos vira uma consulta, por
    exemplo ``"a photo of black sports shoes"``. Relevância branda: mesmo tipo;
    estrita: mesmo tipo e mesma cor.

    Args:
        searcher: Buscador (carrega o CLIP).
        items: Rótulos de todo o índice.
        mode: ``"image"`` (texto→imagem) ou ``"text"`` (texto→descrição).
        k: Profundidade do ranking.
        max_queries: Máximo de pares amostrados.
        min_items: Mínimo de produtos por par para gerar consulta.
        template: Modelo da consulta, com ``{colour}`` e ``{article_type}``.
        seed: Seed da amostragem.

    Returns:
        Relatório com P@5/10 (tipo e estrito), MRR e HitRate@10 estritos.
    """
    total = len(items)
    n_type = Counter(i.article_type for i in items if i.article_type)
    n_pair = Counter((i.article_type, i.colour) for i in items if i.article_type and i.colour)
    by_id = {i.product_id: i for i in items}
    pairs = sorted(p for p, n in n_pair.items() if n >= min_items)
    pairs = random.Random(seed).sample(pairs, min(max_queries, len(pairs)))

    rels: dict[str, list[Sequence[bool]]] = {"type": [], "strict": []}
    rates: dict[str, list[float]] = {"type": [], "strict": []}
    for article_type, colour in pairs:
        query = template.format(colour=colour.lower(), article_type=article_type.lower())
        hits: list[Hit] = searcher.search_by_text(query, k=k, mode=mode)  # type: ignore[arg-type]
        hit_items = [by_id.get(h.product_id) for h in hits]
        rels["type"].append([bool(it and it.article_type == article_type) for it in hit_items])
        rels["strict"].append([bool(it and it.article_type == article_type and it.colour == colour) for it in hit_items])
        rates["type"].append(n_type[article_type] / total)
        rates["strict"].append(n_pair[(article_type, colour)] / total)

    specs = [
        _Spec("P@5 (tipo)", "type", _p(5), "type"),
        _Spec(f"P@{k} (tipo)", "type", _p(k), "type"),
        _Spec("P@5 (tipo+cor)", "strict", _p(5), "strict"),
        _Spec(f"P@{k} (tipo+cor)", "strict", _p(k), "strict"),
        _Spec("MRR (tipo+cor)", "strict", reciprocal_rank, None),
        _Spec(f"HitRate@{k} (tipo+cor)", "strict", lambda r: hit_at_k(r, k), None),
    ]
    metrics, baselines = _summarize_specs(specs, rels, rates)
    name = "texto → imagem" if mode == "image" else "texto → descrição"
    log.info("eval_text_done", mode=mode, queries=len(pairs))
    return EvalReport(name, len(pairs), metrics, baselines)


def format_markdown(reports: Sequence[EvalReport], *, model: str, n_items: int) -> str:
    """Formata os relatórios como tabelas Markdown prontas para o README.

    Args:
        reports: Relatórios a exibir.
        model: Checkpoint avaliado.
        n_items: Tamanho do índice.

    Returns:
        Texto Markdown.
    """
    lines = [f"**Modelo:** `{model}` · **Índice:** {n_items:,} produtos".replace(",", ".")]
    for r in reports:
        lines += [
            "",
            f"#### {r.name} ({r.n_queries} consultas)",
            "",
            "| Métrica | Resultado (IC 95%) | Acaso | Ganho |",
            "|---|---|---|---|",
        ]
        for label, m in r.metrics.items():
            base = r.baselines.get(label)
            cell_base = f"{base:.3f}" if base is not None else "-"
            lift = f"{m['mean'] / base:.1f}×" if base else "-"
            lines.append(f"| {label} | {m['mean']:.3f} ± {m['ci95']:.3f} | {cell_base} | {lift} |")
    lines += [
        "",
        "_Relevância definida pelos metadados do catálogo (mesmo tipo de artigo; estrito = tipo e cor). "
        "\"Acaso\" é a precisão esperada de um ranking aleatório._",
    ]
    return "\n".join(lines)
