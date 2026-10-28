"""Opções de filtro (facetas) derivadas do próprio índice vetorial."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.vectorstore.chroma_store import ChromaStore

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Facets:
    """Valores disponíveis para filtrar a busca.

    Attributes:
        total: Itens no índice.
        genders: Gêneros, em ordem alfabética.
        master_categories: Categorias principais, por frequência.
        article_types: Tipos de artigo, por frequência.
        colours: Cores base, por frequência.
        price_min: Menor preço indexado (``0.0`` se não houver preço).
        price_max: Maior preço indexado.
    """

    total: int = 0
    genders: list[str] = field(default_factory=list)
    master_categories: list[str] = field(default_factory=list)
    article_types: list[str] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    price_min: float = 0.0
    price_max: float = 0.0


def _by_frequency(counter: Counter[str]) -> list[str]:
    """Ordena por frequência decrescente (desempate alfabético), sem 'Unknown'."""
    return [v for v, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])) if v != UNKNOWN]


def compute_facets(store: ChromaStore, batch_size: int = 5000) -> Facets:
    """Calcula as facetas lendo só os metadados do índice, em páginas.

    Usar o índice (e não o ``catalog.parquet``) garante que os filtros mostrem
    apenas valores que existem de fato na busca e dispensa o catálogo no deploy.

    Args:
        store: Coleção de imagens já indexada.
        batch_size: Itens por página de leitura.

    Returns:
        Facetas ordenadas para uso em widgets.
    """
    genders: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    types: Counter[str] = Counter()
    colours: Counter[str] = Counter()
    prices: list[float] = []
    total = 0
    for page in store.iter_metadatas(batch_size):
        for m in page:
            total += 1
            if "gender" in m:
                genders[str(m["gender"])] += 1
            if "master_category" in m:
                categories[str(m["master_category"])] += 1
            if "article_type" in m:
                types[str(m["article_type"])] += 1
            if "base_colour" in m:
                colours[str(m["base_colour"])] += 1
            if isinstance(m.get("price"), int | float):
                prices.append(float(m["price"]))
    return Facets(
        total=total,
        genders=sorted(g for g in genders if g != UNKNOWN),
        master_categories=_by_frequency(categories),
        article_types=_by_frequency(types),
        colours=_by_frequency(colours),
        price_min=min(prices) if prices else 0.0,
        price_max=max(prices) if prices else 0.0,
    )
