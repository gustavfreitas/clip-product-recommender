"""Buscador de produtos por texto, imagem ou ambos (CLIP + ChromaDB)."""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import Settings, get_settings
from src.logger import get_logger
from src.models.base import EmbeddingBackend, EmbeddingError
from src.vectorstore.base import Hit, VectorStore, VectorStoreError
from src.vectorstore.chroma_store import ChromaStore

if TYPE_CHECKING:
    from src.models.clip_embedder import ClipEmbedder  # noqa: F401

log = get_logger(__name__)

ImageSource = str | Path | bytes | Image.Image | BinaryIO
TextMode = Literal["image", "text"]

MAX_QUERY_SIDE = 1024  # imagens de consulta maiores são reduzidas antes do CLIP
MAX_K = 100
MAX_QUERY_CHARS = 500


class SearchError(ValueError):
    """Entrada inválida ou configuração incompatível na busca."""


def load_query_image(source: ImageSource, max_side: int = MAX_QUERY_SIDE) -> Image.Image:
    """Carrega a imagem de consulta como RGB, corrigindo orientação EXIF.

    Args:
        source: Caminho, bytes, objeto de arquivo (ex.: upload do Streamlit)
            ou ``PIL.Image``.
        max_side: Maior lado permitido; imagens maiores são reduzidas.

    Returns:
        Imagem PIL em RGB.

    Raises:
        SearchError: Se a imagem não existir ou não puder ser decodificada.
    """
    try:
        if isinstance(source, Image.Image):
            im = source.copy()
        elif isinstance(source, bytes):
            im = Image.open(io.BytesIO(source))
        elif isinstance(source, str | Path):
            path = Path(source)
            if not path.is_file():
                raise SearchError(f"Imagem não encontrada: {path}")
            im = Image.open(path)
        else:
            im = Image.open(source)
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    except SearchError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, AttributeError) as exc:
        raise SearchError(f"Não foi possível ler a imagem de consulta: {exc}") from exc
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return im


def _unit(vec: np.ndarray) -> np.ndarray:
    """Normaliza em L2 (devolve float32)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class ProductSearcher:
    """Recomenda produtos por similaridade de cosseno no espaço multimodal do CLIP.

    Três formas de consulta:

    * ``search_by_text``: embeda o texto e busca na coleção de imagens
      (``mode="image"``, texto→imagem, o uso zero-shot do CLIP) ou na de
      textos (``mode="text"``, texto→descrição).
    * ``search_by_image``: embeda a imagem e busca na coleção de imagens.
    * ``search_multimodal``: combina imagem + texto num único vetor
      ("esta jaqueta, mas em preto") e busca na coleção de imagens.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        image_store: VectorStore,
        text_store: VectorStore | None = None,
        *,
        default_k: int = 5,
        text_cache_size: int = 256,
    ) -> None:
        """Valida a compatibilidade entre o modelo de consulta e o índice.

        Args:
            backend: Modelo que embeda as consultas (deve ser o mesmo da indexação).
            image_store: Coleção de embeddings de imagem.
            text_store: Coleção de embeddings de texto (opcional).
            default_k: Resultados retornados quando ``k`` não é informado.
            text_cache_size: Consultas de texto mantidas em cache (LRU).

        Raises:
            SearchError: Se modelo ou dimensão divergirem do índice.
        """
        for label, store in (("imagens", image_store), ("textos", text_store)):
            if store is None:
                continue
            model = getattr(store, "model_name", None)
            if model is not None and model != backend.name:
                raise SearchError(
                    f"A coleção de {label} foi indexada com '{model}', mas o embedder é '{backend.name}'. "
                    "Consultas e índice precisam usar o mesmo modelo."
                )
            dim = getattr(store, "dim", backend.dim)
            if dim != backend.dim:
                raise SearchError(f"Dimensão da coleção de {label} ({dim}) != dimensão do modelo ({backend.dim}).")
        self.backend = backend
        self.image_store = image_store
        self.text_store = text_store
        self.default_k = self._validate_k(default_k)
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._text_cache_size = max(0, text_cache_size)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ProductSearcher:
        """Abre o índice existente e carrega o CLIP que o gerou.

        O modelo é lido dos metadados da coleção (não do ``.env``), garantindo
        que a consulta use sempre o mesmo modelo da indexação.

        Args:
            settings: Configurações (padrão: ``get_settings()``).

        Returns:
            Buscador pronto para uso.

        Raises:
            VectorStoreError: Se o índice não existir.
            EmbeddingError: Se o modelo não puder ser carregado.
        """
        s = settings or get_settings()
        image_store = ChromaStore.open_existing(s.vectorstore_dir, f"{s.collection_name}_image")
        try:
            text_store: VectorStore | None = ChromaStore.open_existing(s.vectorstore_dir, f"{s.collection_name}_text")
        except VectorStoreError:
            log.warning("text_collection_missing", hint="busca texto->texto desativada")
            text_store = None
        try:
            from src.models.clip_embedder import ClipEmbedder  # import tardio: torch é pesado
        except ImportError as exc:
            raise EmbeddingError(f"Dependência ausente ({exc}). Rode: pip install -r requirements.txt") from exc
        backend = ClipEmbedder(image_store.model_name, s.device)
        return cls(backend, image_store, text_store, default_k=s.default_top_k)

    # ------------------------------------------------------------------ consultas
    def search_by_text(
        self,
        query_string: str,
        k: int | None = None,
        *,
        mode: TextMode = "image",
        where: dict[str, Any] | None = None,
        exclude_ids: Collection[str] = (),
    ) -> list[Hit]:
        """Busca produtos a partir de uma descrição em texto livre.

        Args:
            query_string: Descrição (ex.: ``"black running shoes"``). O CLIP
                entende melhor inglês; textos além de 77 tokens são truncados.
            k: Quantidade de resultados (padrão: ``default_k``).
            mode: ``"image"`` compara o texto com as imagens; ``"text"`` compara
                com as descrições dos produtos.
            where: Filtro de metadados (ver ``metadata.build_where``).
            exclude_ids: IDs a omitir do resultado.

        Returns:
            Top-k em ordem decrescente de cosseno.

        Raises:
            SearchError: Texto vazio, ``k`` inválido ou coleção indisponível.
        """
        t0 = time.perf_counter()
        k = self._validate_k(self.default_k if k is None else k)
        if mode == "image":
            store: VectorStore = self.image_store
        elif mode == "text":
            if self.text_store is None:
                raise SearchError("A coleção de textos não está disponível; use mode='image'.")
            store = self.text_store
        else:
            raise SearchError(f"mode inválido: {mode!r} (use 'image' ou 'text')")
        hits = self._search(store, self._embed_text(query_string), k, where, exclude_ids)
        self._log("text", t0, mode=mode, k=k, results=len(hits))
        return hits

    def search_by_image(
        self,
        image_path: ImageSource,
        k: int | None = None,
        *,
        where: dict[str, Any] | None = None,
        exclude_ids: Collection[str] = (),
    ) -> list[Hit]:
        """Busca produtos visualmente parecidos com uma imagem.

        Args:
            image_path: Caminho, bytes, arquivo aberto (upload) ou ``PIL.Image``.
            k: Quantidade de resultados (padrão: ``default_k``).
            where: Filtro de metadados (ver ``metadata.build_where``).
            exclude_ids: IDs a omitir (ex.: o próprio produto, em "itens similares").

        Returns:
            Top-k em ordem decrescente de cosseno.

        Raises:
            SearchError: Imagem ilegível ou ``k`` inválido.
        """
        t0 = time.perf_counter()
        k = self._validate_k(self.default_k if k is None else k)
        hits = self._search(self.image_store, self._embed_image(image_path), k, where, exclude_ids)
        self._log("image", t0, k=k, results=len(hits))
        return hits

    def search_multimodal(
        self,
        image: ImageSource,
        text: str,
        k: int | None = None,
        *,
        image_weight: float = 0.5,
        where: dict[str, Any] | None = None,
        exclude_ids: Collection[str] = (),
    ) -> list[Hit]:
        """Busca combinando imagem e texto num único vetor de consulta.

        O vetor é ``normalize(w·img + (1-w)·txt)`` e a busca roda na coleção de
        imagens. Útil para refinar: foto de uma jaqueta + ``"in black"``.

        Args:
            image: Imagem de referência.
            text: Texto de refinamento.
            k: Quantidade de resultados.
            image_weight: Peso da imagem em [0, 1] (0.5 = pesos iguais).
            where: Filtro de metadados.
            exclude_ids: IDs a omitir.

        Returns:
            Top-k em ordem decrescente de cosseno com o vetor combinado.

        Raises:
            SearchError: Entradas inválidas.
        """
        t0 = time.perf_counter()
        k = self._validate_k(self.default_k if k is None else k)
        if not 0.0 <= image_weight <= 1.0:
            raise SearchError("image_weight deve estar em [0, 1]")
        vec = _unit(image_weight * self._embed_image(image) + (1.0 - image_weight) * self._embed_text(text))
        hits = self._search(self.image_store, vec, k, where, exclude_ids)
        self._log("multimodal", t0, k=k, image_weight=image_weight, results=len(hits))
        return hits

    # ----------------------------------------------------------------- internos
    def _search(
        self,
        store: VectorStore,
        vec: np.ndarray,
        k: int,
        where: dict[str, Any] | None,
        exclude_ids: Collection[str],
    ) -> list[Hit]:
        """Consulta o store pedindo ``k + len(exclude)`` para compensar as exclusões."""
        excluded = set(exclude_ids)
        try:
            hits = store.query(vec, k=k + len(excluded), where=where)
        except VectorStoreError as exc:
            raise SearchError(f"Falha na busca vetorial: {exc}") from exc
        if excluded:
            hits = [h for h in hits if h.product_id not in excluded]
        return hits[:k]

    def _embed_text(self, text: str) -> np.ndarray:
        """Embeda o texto (com cache LRU) e devolve vetor unitário."""
        if not isinstance(text, str) or not text.strip():
            raise SearchError("A consulta de texto está vazia.")
        key = text.strip()[:MAX_QUERY_CHARS]
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        try:
            vec = _unit(self.backend.embed_texts([key])[0])
        except EmbeddingError as exc:
            raise SearchError(f"Falha ao embedar o texto: {exc}") from exc
        if self._text_cache_size:
            self._text_cache[key] = vec
            while len(self._text_cache) > self._text_cache_size:
                self._text_cache.popitem(last=False)
        return vec

    def _embed_image(self, source: ImageSource) -> np.ndarray:
        """Carrega e embeda a imagem, devolvendo vetor unitário."""
        image = load_query_image(source)
        try:
            return _unit(self.backend.embed_images([image])[0])
        except EmbeddingError as exc:
            raise SearchError(f"Falha ao embedar a imagem: {exc}") from exc

    @staticmethod
    def _validate_k(value: int) -> int:
        """Valida ``k`` (inteiro entre 1 e ``MAX_K``)."""
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_K:
            raise SearchError(f"k deve ser um inteiro entre 1 e {MAX_K}")
        return value

    def _log(self, kind: str, t0: float, **fields: object) -> None:
        log.info("search_done", kind=kind, ms=round((time.perf_counter() - t0) * 1000, 1), **fields)
