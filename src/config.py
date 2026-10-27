"""Configuração centralizada do projeto.

Todos os parâmetros podem ser sobrescritos por variáveis de ambiente com
prefixo ``MMREC_`` ou por um arquivo ``.env`` na raiz do projeto.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Parâmetros do pipeline de dados.

    Attributes:
        kaggle_dataset: Slug ``dono/dataset`` no Kaggle.
        raw_dir: Pasta onde o dataset bruto é extraído.
        processed_dir: Pasta de saída (catálogo limpo + imagens padronizadas).
        max_image_side: Maior lado (px) das imagens processadas.
        min_image_side: Menor lado (px) aceito; imagens menores são descartadas.
        jpeg_quality: Qualidade JPEG (1-100) das imagens processadas.
        num_workers: Threads para validar/redimensionar imagens.
        price_seed: Seed do gerador de preços sintéticos (reprodutibilidade).
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR).
        log_json: Se True, emite logs em JSON (ideal para produção).
        clip_model_name: Checkpoint CLIP no Hugging Face Hub.
        device: ``auto`` (cuda > mps > cpu), ``cpu``, ``cuda`` ou ``mps``.
        embed_batch_size: Tamanho do mini-batch na geração de embeddings.
        vectorstore_dir: Pasta de persistência do ChromaDB.
        collection_name: Prefixo das coleções (``{prefixo}_image`` / ``{prefixo}_text``).
        index_batch_size: Itens por ``upsert`` no banco vetorial.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="MMREC_",
        extra="ignore",
    )

    kaggle_dataset: str = "paramaggarwal/fashion-product-images-small"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    max_image_side: int = Field(default=512, ge=64, le=4096)
    min_image_side: int = Field(default=32, ge=1)
    jpeg_quality: int = Field(default=90, ge=1, le=100)
    num_workers: int = Field(default=8, ge=1, le=64)
    price_seed: int = 42
    log_level: str = "INFO"
    log_json: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    embed_batch_size: int = Field(default=32, ge=1, le=1024)
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    collection_name: str = Field(default="products", min_length=3, max_length=40)
    index_batch_size: int = Field(default=1024, ge=1, le=100_000)

    @field_validator("raw_dir", "processed_dir", "vectorstore_dir", mode="after")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        """Resolve caminhos relativos a partir da raiz do projeto."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def processed_images_dir(self) -> Path:
        """Pasta das imagens padronizadas, nomeadas por ``{product_id}.jpg``."""
        return self.processed_dir / "images"

    @property
    def catalog_path(self) -> Path:
        """Caminho do catálogo limpo em Parquet."""
        return self.processed_dir / "catalog.parquet"

    @property
    def embeddings_dir(self) -> Path:
        """Pasta dos embeddings persistidos (memmap ``.npy`` + metadados)."""
        return self.processed_dir / "embeddings"

    @property
    def report_path(self) -> Path:
        """Caminho do relatório JSON de qualidade de dados."""
        return self.processed_dir / "prep_report.json"


def relative_to_project(path: Path) -> str:
    """Converte ``path`` em string POSIX relativa à raiz do projeto.

    Mantém o catálogo portável entre máquinas. Se não houver caminho relativo
    possível (ex.: outro drive no Windows), devolve o caminho absoluto.

    Args:
        path: Caminho a converter.

    Returns:
        Caminho relativo (ou absoluto, no fallback) em formato POSIX.
    """
    try:
        return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_project_path(path: str | Path) -> Path:
    """Transforma um caminho do catálogo em caminho absoluto utilizável.

    Args:
        path: Caminho relativo à raiz do projeto (ou já absoluto).

    Returns:
        Caminho absoluto.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def get_settings() -> Settings:
    """Instancia as configurações (lê ambiente e ``.env``)."""
    return Settings()
