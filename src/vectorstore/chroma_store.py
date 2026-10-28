"""Implementação do banco vetorial sobre ChromaDB (persistente, HNSW + cosseno)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.logger import get_logger
from src.vectorstore.base import Hit, VectorStore, VectorStoreError

log = get_logger(__name__)

DEFAULT_MAX_BATCH = 5000  # usado se o Chroma não informar seu limite de lote


def _list_collection_names(client: Any) -> set[str]:
    """Lista nomes de coleções (a API retorna objetos ou strings conforme a versão)."""
    return {getattr(c, "name", c) for c in client.list_collections()}


class ChromaStore(VectorStore):
    """Coleção Chroma com distância de cosseno e verificação de compatibilidade.

    Os embeddings são sempre fornecidos por nós (``embedding_function=None``),
    então o Chroma nunca baixa nem executa um modelo próprio. A coleção guarda
    o *fingerprint* dos embeddings de origem; se ele mudar (outro modelo ou
    catálogo), a abertura falha em vez de misturar vetores incompatíveis.
    """

    def __init__(
        self,
        directory: Path,
        collection_name: str,
        *,
        dim: int,
        fingerprint: str,
        model_name: str,
        recreate: bool = False,
    ) -> None:
        """Abre (ou cria) a coleção.

        Args:
            directory: Pasta de persistência do Chroma.
            collection_name: Nome da coleção.
            dim: Dimensão esperada dos vetores.
            fingerprint: Hash dos embeddings de origem (de ``StoredEmbeddings``).
            model_name: Checkpoint que gerou os vetores.
            recreate: Se True, apaga a coleção existente antes de criar.

        Raises:
            VectorStoreError: Se a coleção existente for incompatível e
                ``recreate`` for False, ou se o Chroma falhar ao abrir.
        """
        self.dim = dim
        self.collection_name = collection_name
        self.model_name = model_name
        self._meta = {"embeddings_fingerprint": fingerprint, "model_name": model_name, "dim": dim}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._client = self._make_client(directory)
            exists = collection_name in _list_collection_names(self._client)
            if exists and recreate:
                self._client.delete_collection(collection_name)
                log.info("collection_deleted", name=collection_name)
                exists = False
            if exists:
                self._collection = self._open_existing()
            else:
                self._collection = self._create()
        except VectorStoreError:
            raise
        except Exception as exc:  # o Chroma levanta tipos variados entre versões
            raise VectorStoreError(f"Falha ao abrir o ChromaDB em {directory}: {exc}") from exc

        self._max_batch = self._read_max_batch(self._client)

    @staticmethod
    def _make_client(directory: Path) -> Any:
        """Cria o cliente persistente (telemetria desligada)."""
        return chromadb.PersistentClient(path=str(directory), settings=ChromaSettings(anonymized_telemetry=False))

    @staticmethod
    def _read_max_batch(client: Any) -> int:
        """Lê o limite de lote do Chroma (com fallback para versões antigas)."""
        try:
            return int(client.get_max_batch_size())
        except Exception:  # noqa: BLE001 - recurso ausente em versões antigas
            return DEFAULT_MAX_BATCH

    @classmethod
    def open_existing(cls, directory: Path, collection_name: str) -> ChromaStore:
        """Abre uma coleção já indexada, sem recriá-la (modo de consulta).

        Dimensão, modelo e fingerprint são lidos dos metadados da própria
        coleção, o que permite ao buscador usar o mesmo modelo que indexou.

        Args:
            directory: Pasta de persistência do Chroma (precisa existir).
            collection_name: Nome da coleção.

        Returns:
            Store pronto para ``query``.

        Raises:
            VectorStoreError: Se a pasta ou a coleção não existirem, ou se
                os metadados da coleção estiverem incompletos.
        """
        if not directory.is_dir():
            raise VectorStoreError(f"Banco vetorial não encontrado em {directory}. Rode: python scripts/build_index.py")
        obj = cls.__new__(cls)
        obj.collection_name = collection_name
        try:
            obj._client = cls._make_client(directory)
            if collection_name not in _list_collection_names(obj._client):
                raise VectorStoreError(
                    f"Coleção '{collection_name}' não existe em {directory}. Rode: python scripts/build_index.py"
                )
            obj._collection = obj._client.get_collection(collection_name, embedding_function=None)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Falha ao abrir '{collection_name}': {exc}") from exc

        meta = dict(obj._collection.metadata or {})
        if "dim" not in meta or "model_name" not in meta:
            raise VectorStoreError(f"Coleção '{collection_name}' sem metadados de modelo/dimensão; reconstrua com --recreate.")
        obj.dim = int(meta["dim"])
        obj.model_name = str(meta["model_name"])
        obj._meta = meta
        obj._max_batch = cls._read_max_batch(obj._client)
        log.info("collection_opened_readonly", name=collection_name, items=obj.count(), model=obj.model_name)
        return obj

    def _create(self) -> Any:
        """Cria a coleção com HNSW em cosseno (API nova, com fallback para a legada)."""
        try:
            col = self._client.create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=self._meta,
                embedding_function=None,
            )
        except (TypeError, ValueError):
            col = self._client.create_collection(
                self.collection_name,
                metadata={**self._meta, "hnsw:space": "cosine"},
                embedding_function=None,
            )
        log.info("collection_created", name=self.collection_name, dim=self.dim, space="cosine")
        return col

    def _open_existing(self) -> Any:
        """Abre a coleção existente validando o fingerprint."""
        col = self._client.get_collection(self.collection_name, embedding_function=None)
        stored_fp = (col.metadata or {}).get("embeddings_fingerprint")
        if stored_fp != self._meta["embeddings_fingerprint"]:
            raise VectorStoreError(
                f"A coleção '{self.collection_name}' foi criada com outros embeddings "
                "(modelo ou catálogo diferente). Use --recreate para reconstruí-la."
            )
        log.info("collection_opened", name=self.collection_name, items=col.count())
        return col

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: Sequence[dict[str, Any]],
        documents: Sequence[str] | None = None,
    ) -> None:
        """Insere ou atualiza itens em sub-lotes que respeitam o limite do Chroma.

        Args:
            ids: IDs únicos (``product_id``).
            embeddings: Array ``(n, dim)`` L2-normalizado.
            metadatas: Um dicionário de metadados por item.
            documents: Textos associados (opcional).

        Raises:
            VectorStoreError: Se formatos forem inconsistentes, houver valores
                não finitos ou o Chroma rejeitar a escrita.
        """
        n = len(ids)
        if n == 0:
            return
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.shape != (n, self.dim):
            raise VectorStoreError(f"Formato de embeddings {emb.shape} != esperado {(n, self.dim)}")
        if len(metadatas) != n or (documents is not None and len(documents) != n):
            raise VectorStoreError("ids, metadatas e documents devem ter o mesmo tamanho")
        if len(set(ids)) != n:
            raise VectorStoreError("IDs duplicados dentro do mesmo lote")
        if not np.isfinite(emb).all():
            raise VectorStoreError("Embeddings contêm NaN/Inf")

        try:
            for lo in range(0, n, self._max_batch):
                hi = min(lo + self._max_batch, n)
                self._collection.upsert(
                    ids=list(ids[lo:hi]),
                    embeddings=emb[lo:hi].tolist(),
                    metadatas=list(metadatas[lo:hi]),
                    documents=list(documents[lo:hi]) if documents is not None else None,
                )
        except Exception as exc:
            raise VectorStoreError(f"Falha no upsert: {exc}") from exc

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Busca os ``k`` vizinhos mais próximos por cosseno.

        Args:
            embedding: Vetor de consulta ``(dim,)`` (idealmente já normalizado).
            k: Número de resultados.
            where: Filtro de metadados (ver ``metadata.build_where``).

        Returns:
            Hits em ordem decrescente de similaridade. Pode haver menos de ``k``.

        Raises:
            VectorStoreError: Se o vetor tiver dimensão errada ou a consulta falhar.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise VectorStoreError(f"Dimensão da consulta {vec.shape[0]} != {self.dim}")
        total = self.count()
        if total == 0 or k < 1:
            return []
        try:
            res = self._collection.query(
                query_embeddings=[vec.tolist()],
                n_results=min(k, total),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Falha na consulta: {exc}") from exc

        ids = res["ids"][0]
        distances = res["distances"][0]
        metas = res["metadatas"][0]
        # distância de cosseno = 1 - cos  ->  similaridade = 1 - distância
        return [Hit(pid, float(1.0 - d), dict(m or {})) for pid, d, m in zip(ids, distances, metas, strict=True)]

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Recupera os vetores armazenados (usado para verificar o índice).

        Args:
            ids: IDs desejados.

        Returns:
            Mapa ``id -> vetor float32``. IDs inexistentes ficam de fora.
        """
        try:
            res = self._collection.get(ids=list(ids), include=["embeddings"])
        except Exception as exc:
            raise VectorStoreError(f"Falha ao ler embeddings: {exc}") from exc
        vectors = res["embeddings"]
        return {pid: np.asarray(vectors[i], dtype=np.float32) for i, pid in enumerate(res["ids"])}

    def iter_metadatas(self, batch_size: int = 5000) -> Iterator[list[dict[str, Any]]]:
        """Percorre os metadados de todos os itens em páginas (sem carregar vetores).

        Args:
            batch_size: Itens por página.

        Yields:
            Listas de dicionários de metadados.

        Raises:
            VectorStoreError: Se a leitura falhar.
        """
        offset = 0
        while True:
            try:
                res = self._collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            except Exception as exc:
                raise VectorStoreError(f"Falha ao ler metadados: {exc}") from exc
            metas = res["metadatas"]
            if metas is None or len(metas) == 0:
                return
            yield [dict(m or {}) for m in metas]
            offset += len(metas)

    def count(self) -> int:
        """Quantidade de itens na coleção."""
        return int(self._collection.count())

    def close(self) -> None:
        """Libera os handles do Chroma (evita arquivo travado no Windows)."""
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass
