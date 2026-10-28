"""App Streamlit: recomendação de produtos por imagem, texto ou ambos (CLIP + ChromaDB).

Execução local:
    streamlit run app.py

Pré-requisitos: ``scripts/prepare_data.py``, ``generate_embeddings.py`` e ``build_index.py``
já executados (Passos 1-3). O app só consulta o índice; nunca indexa.
"""

from __future__ import annotations

import html
import io
import re
import time
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from src.config import get_settings, resolve_project_path
from src.logger import configure_logging, get_logger
from src.models.base import EmbeddingError
from src.recommender.facets import Facets, compute_facets
from src.recommender.search import ProductSearcher, SearchError
from src.vectorstore.base import Hit, VectorStoreError
from src.vectorstore.metadata import build_where

log = get_logger("app")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CARD_IMAGE_SIZE = 320
CARDS_PER_ROW = 5
NAME_MAX_CHARS = 46  # cabe em 3 linhas no card
EXAMPLE_QUERIES = ("black running shoes", "red summer dress", "blue denim jacket", "leather handbag")
TEXT_MODES = {"Imagens (visual)": "image", "Descrições (texto)": "text"}

st.set_page_config(page_title="Recomendador Multimodal de Produtos", page_icon="🛍️", layout="wide")


# ------------------------------------------------------------------ recursos em cache
@st.cache_resource(show_spinner=False)
def _init_logging() -> bool:
    """Configura o logging uma única vez por processo."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return True


@st.cache_resource(show_spinner="Carregando o modelo CLIP e o índice vetorial...")
def get_searcher() -> ProductSearcher:
    """Abre o índice e carrega o CLIP (feito uma vez por processo, não por interação)."""
    return ProductSearcher.from_settings(get_settings())


@st.cache_data(show_spinner=False)
def get_facets(_searcher: ProductSearcher, index_size: int) -> Facets:
    """Calcula as opções de filtro a partir do índice (``index_size`` invalida o cache)."""
    return compute_facets(_searcher.image_store)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False, max_entries=512)
def card_image(path: str, size: int = CARD_IMAGE_SIZE) -> bytes:
    """Prepara a imagem do card como JPEG quadrado com fundo branco.

    Redimensionar aqui (e não no navegador) mantém o app independente de
    parâmetros de layout do ``st.image``, que mudam entre versões do Streamlit.

    Args:
        path: Caminho da imagem (relativo à raiz do projeto ou absoluto).
        size: Lado do quadrado em px.

    Returns:
        Bytes JPEG. Se a imagem não puder ser lida, um quadro cinza "sem imagem".
    """
    try:
        with Image.open(resolve_project_path(path)) as im:
            canvas = ImageOps.pad(im.convert("RGB"), (size, size), Image.Resampling.LANCZOS, color=(255, 255, 255))
    except (OSError, ValueError):
        canvas = Image.new("RGB", (size, size), (232, 232, 232))
        ImageDraw.Draw(canvas).text((size // 2 - 28, size // 2), "sem imagem", fill=(120, 120, 120))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=128)
def run_search(
    _searcher: ProductSearcher,
    index_size: int,
    kind: str,
    image_bytes: bytes | None,
    image_path: str | None,
    text: str,
    mode: str,
    image_weight: float,
    k: int,
    where: dict[str, Any] | None,
    exclude: tuple[str, ...],
) -> list[Hit]:
    """Executa a busca; o resultado fica em cache por combinação de parâmetros.

    Mexer só nos filtros ou em ``k`` não reexecuta o encoder de imagem quando a
    combinação já foi vista, e repetir uma consulta é instantâneo.

    Args:
        _searcher: Buscador (ignorado no hash do cache).
        index_size: Itens no índice (invalida o cache se o índice mudar).
        kind: ``similar``, ``multimodal``, ``image`` ou ``text``.
        image_bytes: Bytes da imagem enviada (para ``image``/``multimodal``).
        image_path: Caminho da imagem do catálogo (para ``similar``).
        text: Texto da consulta.
        mode: ``image`` ou ``text`` (coleção alvo da busca só por texto).
        image_weight: Peso da imagem na busca combinada.
        k: Quantidade de resultados.
        where: Filtro de metadados.
        exclude: IDs a omitir.

    Returns:
        Hits em ordem decrescente de cosseno.

    Raises:
        SearchError: Entrada inválida.
    """
    common: dict[str, Any] = {"k": k, "where": where, "exclude_ids": exclude}
    if kind == "similar" and image_path:
        return _searcher.search_by_image(resolve_project_path(image_path), **common)
    if kind == "multimodal" and image_bytes:
        return _searcher.search_multimodal(image_bytes, text, image_weight=image_weight, **common)
    if kind == "image" and image_bytes:
        return _searcher.search_by_image(image_bytes, **common)
    return _searcher.search_by_text(text, mode=mode, **common)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- utilidades
def fmt_brl(value: float) -> str:
    """Formata como moeda brasileira (``R$ 1.234,50``)."""
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def md(text: object) -> str:
    """Escapa caracteres especiais de Markdown em textos vindos dos dados."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~<>$])", r"\\\1", str(text))


def _reset_similar() -> None:
    """Sai do modo 'itens similares' quando o usuário muda a consulta."""
    st.session_state["similar_to"] = None


def _set_similar(product_id: str, name: str, image_path: str) -> None:
    """Ativa o modo 'itens similares' a partir de um card."""
    st.session_state["similar_to"] = {"id": product_id, "name": name, "image_path": image_path}


def _use_example(query: str) -> None:
    """Preenche a caixa de texto com uma consulta de exemplo."""
    st.session_state["query_text"] = query
    st.session_state["similar_to"] = None


# --------------------------------------------------------------------------- interface
def render_sidebar(searcher: ProductSearcher, facets: Facets) -> tuple[int, dict[str, Any] | None]:
    """Desenha a barra lateral e devolve ``(k, where)``."""
    st.sidebar.header("Filtros")
    k = st.sidebar.slider("Quantidade de resultados", 1, 20, searcher.default_k)
    genders = st.sidebar.multiselect("Gênero", facets.genders)
    categories = st.sidebar.multiselect("Categoria", facets.master_categories)
    types = st.sidebar.multiselect("Tipo de artigo", facets.article_types)
    colours = st.sidebar.multiselect("Cor", facets.colours)

    min_price: float | None = None
    max_price: float | None = None
    if facets.price_max > facets.price_min:
        lo, hi = st.sidebar.slider(
            "Faixa de preço (R$)",
            float(int(facets.price_min)),
            float(int(facets.price_max) + 1),
            (float(int(facets.price_min)), float(int(facets.price_max) + 1)),
            step=5.0,
        )
        min_price = lo if lo > int(facets.price_min) else None
        max_price = hi if hi < int(facets.price_max) + 1 else None

    where = build_where(
        min_price=min_price,
        max_price=max_price,
        gender=genders,
        master_category=categories,
        article_type=types,
        base_colour=colours,
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"**Índice:** {facets.total:,} produtos\n\n**Modelo:** {md(searcher.backend.name)}\n\n"
        f"**Dispositivo:** {getattr(searcher.backend, 'device', 'n/d')}".replace(",", ".")
    )
    return k, where


def render_card(hit: Hit, rank: int) -> None:
    """Desenha o card de um produto."""
    m = hit.metadata
    name = str(m.get("name", hit.product_id))
    with st.container(border=True):
        st.image(card_image(str(m.get("image_path", ""))))
        short = name if len(name) <= NAME_MAX_CHARS else name[: NAME_MAX_CHARS - 3] + "..."
        # Alturas mínimas fixas mantêm preço e botão alinhados entre cards com nomes de tamanhos diferentes.
        st.markdown(
            f'<div style="min-height:4.5em;font-weight:600;line-height:1.5">{html.escape(short)}</div>',
            unsafe_allow_html=True,
        )
        details = " · ".join(str(m[f]) for f in ("article_type", "base_colour") if m.get(f) and m[f] != "Unknown")
        st.markdown(
            f'<div style="min-height:2.6rem;opacity:.65;font-size:.85rem;line-height:1.3rem">{html.escape(details)}</div>',
            unsafe_allow_html=True,
        )
        price = m.get("price")
        if isinstance(price, int | float):
            st.markdown(f"**{md(fmt_brl(price))}**")
        st.caption(f"#{rank} · cosseno {hit.score:.3f}")
        st.button(
            "Ver similares",
            key=f"sim_{hit.product_id}_{rank}",
            on_click=_set_similar,
            args=(hit.product_id, name, str(m.get("image_path", ""))),
        )


def render_results(hits: list[Hit], elapsed_ms: float) -> None:
    """Desenha a grade de resultados."""
    if not hits:
        st.info("Nenhum produto encontrado com esses filtros. Tente removê-los ou ampliar a faixa de preço.")
        return
    st.subheader(f"{len(hits)} recomendações")
    st.caption(f"Busca em {elapsed_ms:.0f} ms · ordenado por similaridade de cosseno (CLIP)")
    for start in range(0, len(hits), CARDS_PER_ROW):
        row = hits[start : start + CARDS_PER_ROW]
        for offset, (col, hit) in enumerate(zip(st.columns(CARDS_PER_ROW), row, strict=False)):
            with col:
                render_card(hit, start + offset + 1)
    if any(h.metadata.get("price_is_synthetic") for h in hits):
        st.caption("Preços sintéticos, gerados para demonstração (o dataset original não tem preço).")


def main() -> None:
    """Monta a página."""
    _init_logging()
    st.session_state.setdefault("similar_to", None)

    st.title("Recomendador Multimodal de Produtos")
    st.caption("Busque por **imagem**, por **texto** ou pelos dois juntos. Embeddings CLIP + busca por cosseno no ChromaDB.")

    try:
        searcher = get_searcher()
    except (VectorStoreError, EmbeddingError) as exc:
        st.error(f"Não foi possível iniciar o buscador: {exc}")
        st.markdown("Gere o índice antes de abrir o app:")
        st.code(
            "python scripts/prepare_data.py\npython scripts/generate_embeddings.py\npython scripts/build_index.py",
            language="bash",
        )
        st.stop()

    index_size = searcher.image_store.count()
    facets = get_facets(searcher, index_size)
    k, where = render_sidebar(searcher, facets)

    similar = st.session_state["similar_to"]
    uploaded = None
    text = ""
    mode = "image"
    image_weight = 0.5

    if similar:
        left, right = st.columns([5, 1])
        left.info(f"Mostrando itens parecidos com **{md(similar['name'])}**")
        right.button("Limpar", on_click=_reset_similar)
        st.image(card_image(similar["image_path"]), width=140)
    else:
        col_img, col_txt = st.columns(2)
        with col_img:
            uploaded = st.file_uploader(
                "Arraste uma imagem ou clique para enviar",
                type=["jpg", "jpeg", "png", "webp"],
                key="uploader",
                on_change=_reset_similar,
            )
            if uploaded is not None:
                if uploaded.size > MAX_UPLOAD_BYTES:
                    st.error("A imagem passa de 10 MB. Envie um arquivo menor.")
                    uploaded = None
                else:
                    st.image(uploaded, width=160)
        with col_txt:
            text = st.text_input(
                "Descreva o produto (inglês funciona melhor)",
                key="query_text",
                placeholder="ex.: black leather jacket",
                on_change=_reset_similar,
            ).strip()
            if text and uploaded is None and searcher.text_store is not None:
                label = st.radio("Comparar o texto com", list(TEXT_MODES), horizontal=True)
                mode = TEXT_MODES[label]
            if text and uploaded is not None:
                image_weight = st.slider(
                    "Peso da imagem na busca combinada",
                    0.0,
                    1.0,
                    0.5,
                    0.05,
                    help="0 = só o texto, 1 = só a imagem.",
                )

    if similar:
        kind = "similar"
    elif uploaded is not None and text:
        kind = "multimodal"
    elif uploaded is not None:
        kind = "image"
    elif text:
        kind = "text"
    else:
        st.markdown("#### Comece por aqui")
        st.write("Envie uma foto, escreva uma descrição ou experimente um exemplo:")
        for col, example in zip(st.columns(len(EXAMPLE_QUERIES)), EXAMPLE_QUERIES, strict=True):
            col.button(example, key=f"ex_{example}", on_click=_use_example, args=(example,))
        return

    exclude = (similar["id"],) if similar else ()
    t0 = time.perf_counter()
    try:
        with st.spinner("Buscando..."):
            hits = run_search(
                searcher,
                index_size,
                kind,
                uploaded.getvalue() if uploaded is not None else None,
                similar["image_path"] if similar else None,
                text,
                mode,
                image_weight,
                k,
                where,
                exclude,
            )
    except SearchError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # falha inesperada: registra e mostra mensagem amigável
        log.exception("search_unexpected_error", error=str(exc))
        st.error("Ocorreu um erro inesperado durante a busca. Tente outra consulta.")
        return
    render_results(hits, (time.perf_counter() - t0) * 1000)


main()
