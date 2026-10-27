"""Persistência dos embeddings em disco (memmap ``.npy``) com retomada.

Layout em ``embeddings_dir``::

    image_embeddings.npy   # (N, D) float32, uma linha por produto do catálogo
    text_embeddings.npy    # (N, D) float32
    product_ids.json       # IDs na mesma ordem das linhas
    meta.json              # fingerprint, progresso e IDs com imagem inválida

Os vetores vivem em memória mapeada: o processo nunca precisa manter a matriz
inteira em RAM, e uma execução interrompida retoma do último checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)

IMAGE_FILE = "image_embeddings.npy"
TEXT_FILE = "text_embeddings.npy"
IDS_FILE = "product_ids.json"
META_FILE = "meta.json"


class EmbeddingStoreError(RuntimeError):
    """Estado inválido ou incompatível do armazenamento de embeddings."""


def compute_fingerprint(product_ids: list[str], model_name: str, dim: int) -> str:
    """Identifica univocamente (catálogo, modelo, dimensão).

    Args:
        product_ids: IDs na ordem do catálogo.
        model_name: Nome do checkpoint.
        dim: Dimensão do embedding.

    Returns:
        Hash SHA-1 hexadecimal.
    """
    h = hashlib.sha1()
    h.update(f"{model_name}|{dim}|".encode())
    h.update("\n".join(product_ids).encode())
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any] | list[str]) -> None:
    """Grava JSON de forma atômica (tmp + replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class EmbeddingStore:
    """Escritor sequencial de embeddings com checkpoints.

    Uso::

        with EmbeddingStore(dir, ids, model_name, dim) as store:
            for batch in ...:
                store.write_batch(batch.start, img, txt, failed_ids)
    """

    def __init__(self, directory: Path, product_ids: list[str], model_name: str, dim: int) -> None:
        """Define o destino; nada é aberto até ``open()``.

        Args:
            directory: Pasta de saída.
            product_ids: IDs do catálogo, na ordem das linhas.
            model_name: Nome do checkpoint (entra no fingerprint).
            dim: Dimensão do embedding.
        """
        if not product_ids:
            raise EmbeddingStoreError("Catálogo vazio: nada a indexar.")
        self.directory = directory
        self.product_ids = product_ids
        self.model_name = model_name
        self.dim = dim
        self.fingerprint = compute_fingerprint(product_ids, model_name, dim)
        self.completed_rows = 0
        self.failed_ids: list[str] = []
        self._image: np.memmap | None = None
        self._text: np.memmap | None = None

    @property
    def n_rows(self) -> int:
        """Total de linhas esperadas."""
        return len(self.product_ids)

    def open(self, force: bool = False) -> int:
        """Cria os arquivos ou retoma uma execução compatível.

        Args:
            force: Se True, descarta qualquer resultado anterior.

        Returns:
            Número de linhas já concluídas (0 em execução nova).

        Raises:
            EmbeddingStoreError: Se existir resultado de outro catálogo/modelo
                e ``force`` for False.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = self._read_meta()
        files_ok = (self.directory / IMAGE_FILE).is_file() and (self.directory / TEXT_FILE).is_file()

        if meta is not None and not force:
            if meta.get("fingerprint") != self.fingerprint:
                raise EmbeddingStoreError(
                    "Já existem embeddings de outro catálogo ou modelo em "
                    f"{self.directory}. Use --force para recriá-los."
                )
            if files_ok:
                self._reopen(meta)
                log.info("embeddings_resume", completed=self.completed_rows, total=self.n_rows)
                return self.completed_rows

        self._create()
        return 0

    def write_batch(
        self,
        start: int,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        failed_ids: list[str],
    ) -> None:
        """Grava um lote na posição ``start`` (a escrita é estritamente sequencial).

        Args:
            start: Posição da primeira linha do lote.
            image_embeddings: Array ``(n, dim)``; linhas de imagens inválidas = 0.
            text_embeddings: Array ``(n, dim)``.
            failed_ids: IDs do lote cuja imagem não pôde ser lida.

        Raises:
            EmbeddingStoreError: Se o store não estiver aberto, a ordem for
                violada ou os formatos forem inconsistentes.
        """
        if self._image is None or self._text is None:
            raise EmbeddingStoreError("Store não está aberto.")
        n = len(image_embeddings)
        if start != self.completed_rows:
            raise EmbeddingStoreError(f"Escrita fora de ordem: start={start}, esperado={self.completed_rows}")
        expected = (n, self.dim)
        if image_embeddings.shape != expected or text_embeddings.shape != expected:
            raise EmbeddingStoreError(f"Formato inválido: esperado {expected}")
        if start + n > self.n_rows:
            raise EmbeddingStoreError("Lote ultrapassa o tamanho do catálogo.")

        self._image[start : start + n] = image_embeddings
        self._text[start : start + n] = text_embeddings
        self.completed_rows += n
        self.failed_ids.extend(failed_ids)

    def checkpoint(self) -> None:
        """Força o flush dos vetores e persiste o progresso em ``meta.json``."""
        if self._image is None or self._text is None:
            return
        self._image.flush()
        self._text.flush()
        _atomic_write_json(
            self.directory / META_FILE,
            {
                "fingerprint": self.fingerprint,
                "model_name": self.model_name,
                "dim": self.dim,
                "n_rows": self.n_rows,
                "completed_rows": self.completed_rows,
                "failed_ids": self.failed_ids,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def close(self) -> None:
        """Faz o checkpoint final e libera os arquivos mapeados (importante no Windows)."""
        self.checkpoint()
        self._image = None
        self._text = None

    def __enter__(self) -> EmbeddingStore:
        """Abre (ou retoma) o store ao entrar no ``with``."""
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Garante checkpoint e fechamento ao sair do ``with``."""
        self.close()

    def _create(self) -> None:
        shape = (self.n_rows, self.dim)
        self._image = np.lib.format.open_memmap(self.directory / IMAGE_FILE, mode="w+", dtype=np.float32, shape=shape)
        self._text = np.lib.format.open_memmap(self.directory / TEXT_FILE, mode="w+", dtype=np.float32, shape=shape)
        _atomic_write_json(self.directory / IDS_FILE, self.product_ids)
        self.completed_rows = 0
        self.failed_ids = []
        self.checkpoint()
        log.info("embeddings_created", rows=self.n_rows, dim=self.dim, dir=str(self.directory))

    def _reopen(self, meta: dict[str, Any]) -> None:
        self._image = np.load(self.directory / IMAGE_FILE, mmap_mode="r+")
        self._text = np.load(self.directory / TEXT_FILE, mmap_mode="r+")
        expected = (self.n_rows, self.dim)
        if self._image.shape != expected or self._text.shape != expected:
            raise EmbeddingStoreError("Arquivos .npy inconsistentes com o catálogo. Use --force.")
        self.completed_rows = int(meta["completed_rows"])
        self.failed_ids = list(meta.get("failed_ids", []))

    def _read_meta(self) -> dict[str, Any] | None:
        path = self.directory / META_FILE
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("meta_unreadable_recreating", path=str(path))
            return None


@dataclass
class StoredEmbeddings:
    """Visão somente leitura dos embeddings persistidos.

    Attributes:
        product_ids: IDs na ordem das linhas.
        image: Matriz memmap ``(N, D)`` de embeddings de imagem.
        text: Matriz memmap ``(N, D)`` de embeddings de texto.
        valid: Máscara booleana; False onde a imagem não pôde ser lida.
        model_name: Checkpoint que gerou os vetores.
        dim: Dimensão dos vetores.
        fingerprint: Hash (catálogo, modelo, dimensão) gravado na geração.
    """

    product_ids: list[str]
    image: np.ndarray
    text: np.ndarray
    valid: np.ndarray
    model_name: str
    dim: int
    fingerprint: str = ""

    @classmethod
    def load(cls, directory: Path) -> StoredEmbeddings:
        """Abre embeddings já concluídos.

        Args:
            directory: Pasta criada por ``EmbeddingStore``.

        Returns:
            Instância somente leitura (arrays em ``mmap_mode='r'``).

        Raises:
            EmbeddingStoreError: Se não existirem ou estiverem incompletos.
        """
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise EmbeddingStoreError(f"Embeddings não encontrados em {directory}. Rode scripts/generate_embeddings.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["completed_rows"] != meta["n_rows"]:
            raise EmbeddingStoreError(f"Embeddings incompletos ({meta['completed_rows']}/{meta['n_rows']}). Reexecute para retomar.")
        ids: list[str] = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
        failed = set(meta.get("failed_ids", []))
        return cls(
            product_ids=ids,
            image=np.load(directory / IMAGE_FILE, mmap_mode="r"),
            text=np.load(directory / TEXT_FILE, mmap_mode="r"),
            valid=np.array([pid not in failed for pid in ids], dtype=bool),
            model_name=meta["model_name"],
            dim=int(meta["dim"]),
            fingerprint=str(meta.get("fingerprint", "")),
        )

    def iter_batches(self, batch_size: int = 1024) -> Iterator[tuple[list[str], np.ndarray, np.ndarray]]:
        """Itera sobre as linhas válidas em lotes (para indexar no banco vetorial).

        Args:
            batch_size: Linhas lidas do disco por vez.

        Yields:
            Tuplas ``(product_ids, image_embeddings, text_embeddings)`` em RAM,
            já sem as linhas com imagem inválida.
        """
        if batch_size < 1:
            raise ValueError("batch_size deve ser >= 1")
        for lo in range(0, len(self.product_ids), batch_size):
            hi = min(lo + batch_size, len(self.product_ids))
            idx = np.flatnonzero(self.valid[lo:hi]) + lo
            if idx.size == 0:
                continue
            yield (
                [self.product_ids[i] for i in idx],
                np.asarray(self.image[idx], dtype=np.float32),
                np.asarray(self.text[idx], dtype=np.float32),
            )
