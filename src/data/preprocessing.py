"""Limpeza do catálogo e padronização das imagens por ID."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from src.config import relative_to_project
from src.logger import get_logger

log = get_logger(__name__)

COLUMN_MAP: dict[str, str] = {
    "id": "product_id",
    "gender": "gender",
    "masterCategory": "master_category",
    "subCategory": "sub_category",
    "articleType": "article_type",
    "baseColour": "base_colour",
    "season": "season",
    "year": "year",
    "usage": "usage",
    "productDisplayName": "name",
}
CATEGORICAL_COLS: list[str] = [
    "gender",
    "master_category",
    "sub_category",
    "article_type",
    "base_colour",
    "season",
    "usage",
]
UNKNOWN = "Unknown"

# Preço-base (sintético) por categoria principal.
BASE_PRICE: dict[str, float] = {
    "Apparel": 120.0,
    "Footwear": 220.0,
    "Accessories": 90.0,
    "Personal Care": 60.0,
    "Sporting Goods": 150.0,
    "Home": 100.0,
    "Free Items": 20.0,
}
DEFAULT_BASE_PRICE = 100.0


def clean_catalog(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Padroniza e limpa o catálogo bruto.

    Etapas: seleção/renomeação de colunas, trim de texto, remoção de IDs
    inválidos ou duplicados e de nomes ausentes, normalização de categorias
    (``Title Case``), preenchimento de ausentes e criação de ``search_text``
    (texto descritivo usado depois no embedding de texto).

    Args:
        raw: DataFrame lido de ``styles.csv``.

    Returns:
        Tupla ``(catalogo_limpo, contagem_de_remocoes_por_motivo)``.

    Raises:
        ValueError: Se colunas obrigatórias estiverem ausentes.
    """
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    removed: dict[str, int] = {}
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()

    # Trim + string vazia -> NA
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)

    def _drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            removed[reason] = n
            df = df.loc[~mask].copy()

    _drop(df["product_id"].isna() | ~df["product_id"].fillna("").str.fullmatch(r"\d+"), "invalid_id")
    _drop(df["name"].isna(), "missing_name")
    _drop(df["product_id"].duplicated(keep="first"), "duplicate_id")

    df["name"] = df["name"].str.replace(r"\s+", " ", regex=True)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].str.title().fillna(UNKNOWN)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["search_text"] = df.apply(_build_search_text, axis=1)
    df["product_id"] = df["product_id"].astype(str)

    log.info("catalog_cleaned", rows_in=len(raw), rows_out=len(df), removed=removed)
    return df.reset_index(drop=True), removed


def _build_search_text(row: pd.Series) -> str:
    """Monta um texto descritivo curto a partir dos atributos do produto."""
    parts = [row["name"]]
    for col in ("article_type", "base_colour", "gender", "usage"):
        if row[col] != UNKNOWN:
            parts.append(str(row[col]))
    return ". ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def add_synthetic_price(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Adiciona preço sintético e reprodutível ao catálogo.

    O dataset do Kaggle não traz preço. O valor gerado (base por categoria ×
    ruído log-normal, terminado em ``.90``) serve para demonstrar metadados e
    filtros na UI; a coluna ``price_is_synthetic`` deixa isso explícito.

    Args:
        df: Catálogo limpo.
        seed: Seed do gerador aleatório.

    Returns:
        Cópia do DataFrame com ``price`` e ``price_is_synthetic``.
    """
    out = df.sort_values("product_id", key=lambda s: s.astype("int64")).copy()
    rng = np.random.default_rng(seed)
    base = out["master_category"].map(BASE_PRICE).fillna(DEFAULT_BASE_PRICE).to_numpy()
    noise = rng.lognormal(mean=0.0, sigma=0.45, size=len(out))
    out["price"] = np.maximum(np.round(base * noise / 10) * 10 - 0.10, 9.90).round(2)
    out["price_is_synthetic"] = True
    return out.reset_index(drop=True)


def _process_one(
    product_id: str,
    src_dir: Path,
    dst_dir: Path,
    max_side: int,
    min_side: int,
    quality: int,
    force: bool,
) -> tuple[str, str | None]:
    """Valida e padroniza uma imagem. Retorna ``(id, motivo_da_falha|None)``."""
    src = src_dir / f"{product_id}.jpg"
    dst = dst_dir / f"{product_id}.jpg"
    try:
        if not src.is_file():
            return product_id, "missing_image"
        if dst.is_file() and not force:
            return product_id, None

        with Image.open(src) as im:
            im.load()  # força decodificação completa: pega arquivos truncados
            im = ImageOps.exif_transpose(im)
            if min(im.size) < min_side:
                return product_id, "image_too_small"
            im = im.convert("RGB")  # normaliza RGBA / P / L / CMYK
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, format="JPEG", quality=quality, optimize=True)
        tmp.replace(dst)  # escrita atômica: nunca deixa JPEG pela metade
        return product_id, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        log.debug("image_failed", product_id=product_id, error=str(exc))
        return product_id, "corrupt_image"


def process_images(
    product_ids: list[str],
    src_dir: Path,
    dst_dir: Path,
    *,
    max_side: int = 512,
    min_side: int = 32,
    quality: int = 90,
    workers: int = 8,
    force: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """Padroniza as imagens em ``dst_dir/{product_id}.jpg``.

    Cada imagem é decodificada por completo, corrigida pela orientação EXIF,
    convertida para RGB, limitada a ``max_side`` px e regravada como JPEG.
    O trabalho roda em threads (o decode do Pillow libera o GIL).

    Args:
        product_ids: IDs a processar.
        src_dir: Pasta com as imagens brutas (``{id}.jpg``).
        dst_dir: Pasta de saída (criada se necessário).
        max_side: Maior lado permitido, em px.
        min_side: Menor lado aceito, em px.
        quality: Qualidade JPEG.
        workers: Número de threads.
        force: Reprocessa mesmo se o arquivo de saída já existir.

    Returns:
        Tupla ``(ids_ok, contagem_de_falhas_por_motivo)``.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok: set[str] = set()
    failures: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda pid: _process_one(pid, src_dir, dst_dir, max_side, min_side, quality, force),
            product_ids,
        )
        for pid, reason in tqdm(results, total=len(product_ids), desc="Imagens", unit="img"):
            if reason is None:
                ok.add(pid)
            else:
                failures[reason] = failures.get(reason, 0) + 1

    log.info("images_processed", ok=len(ok), failures=failures)
    return ok, failures


def attach_image_paths(df: pd.DataFrame, dst_dir: Path) -> pd.DataFrame:
    """Cria a coluna ``image_path`` (relativa à raiz do projeto).

    Args:
        df: Catálogo já filtrado para IDs com imagem válida.
        dst_dir: Pasta das imagens processadas.

    Returns:
        Cópia do DataFrame com ``image_path``.
    """
    out = df.copy()
    out["image_path"] = out["product_id"].map(
        lambda pid: relative_to_project(dst_dir / f"{pid}.jpg")
    )
    return out
