"""Download do dataset (Kaggle) e leitura do catálogo bruto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.logger import get_logger

log = get_logger(__name__)

STYLES_FILENAME = "styles.csv"
IMAGES_DIRNAME = "images"


class DatasetDownloadError(RuntimeError):
    """Falha ao baixar o dataset (credenciais, rede ou slug inválido)."""


class RawLayoutError(FileNotFoundError):
    """A estrutura de pastas do dataset bruto não é a esperada."""


def download_dataset(slug: str, dest: Path, force: bool = False) -> None:
    """Baixa e extrai um dataset do Kaggle.

    As credenciais são lidas de ``KAGGLE_USERNAME``/``KAGGLE_KEY`` (inclusive
    via ``.env``) ou de ``~/.kaggle/kaggle.json``.

    Args:
        slug: Identificador ``dono/dataset``.
        dest: Pasta de destino da extração.
        force: Se False e ``dest`` já contém o dataset, pula o download.

    Raises:
        DatasetDownloadError: Se a autenticação ou o download falharem.
    """
    if not force:
        try:
            locate_raw_layout(dest)
            log.info("download_skipped_already_present", dest=str(dest))
            return
        except RawLayoutError:
            pass

    load_dotenv(PROJECT_ROOT / ".env")  # expõe KAGGLE_* ao os.environ
    dest.mkdir(parents=True, exist_ok=True)

    try:
        # Import tardio: `import kaggle` exige credenciais já na importação.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log.info("download_started", dataset=slug, dest=str(dest))
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    except ImportError as exc:
        raise DatasetDownloadError(
            "Pacote 'kaggle' não instalado. Rode: pip install kaggle"
        ) from exc
    except Exception as exc:  # a lib do Kaggle levanta tipos variados
        raise DatasetDownloadError(
            f"Falha ao baixar '{slug}': {exc}. Verifique KAGGLE_USERNAME/KAGGLE_KEY "
            "(ou ~/.kaggle/kaggle.json) e sua conexão."
        ) from exc

    log.info("download_finished", dataset=slug)


def locate_raw_layout(raw_dir: Path) -> tuple[Path, Path]:
    """Localiza ``styles.csv`` e a pasta ``images/`` dentro de ``raw_dir``.

    Aceita a estrutura plana (dataset *small*) e a aninhada (dataset *full*,
    que extrai em ``fashion-dataset/``).

    Args:
        raw_dir: Pasta raiz do dataset bruto.

    Returns:
        Tupla ``(caminho_styles_csv, pasta_images)``.

    Raises:
        RawLayoutError: Se o CSV ou a pasta de imagens não forem encontrados.
    """
    if not raw_dir.is_dir():
        raise RawLayoutError(f"Pasta inexistente: {raw_dir}")

    csv_matches = sorted(raw_dir.rglob(STYLES_FILENAME), key=lambda p: len(p.parts))
    if not csv_matches:
        raise RawLayoutError(f"'{STYLES_FILENAME}' não encontrado em {raw_dir}")

    styles_csv = csv_matches[0]
    images_dir = styles_csv.parent / IMAGES_DIRNAME
    if not images_dir.is_dir():
        raise RawLayoutError(f"Pasta '{IMAGES_DIRNAME}/' não encontrada ao lado de {styles_csv}")

    return styles_csv, images_dir


def load_styles(csv_path: Path) -> pd.DataFrame:
    """Lê o ``styles.csv`` tolerando linhas malformadas.

    O CSV original tem algumas linhas com vírgulas extras no nome do produto;
    elas são descartadas (e contabilizadas no log) em vez de derrubar a leitura.

    Args:
        csv_path: Caminho do ``styles.csv``.

    Returns:
        DataFrame com todas as colunas como ``str``.

    Raises:
        ValueError: Se o arquivo não puder ser lido como CSV.
    """
    bad_lines: list[list[str]] = []

    def _skip_bad_line(line: list[str]) -> None:
        bad_lines.append(line)
        return None

    try:
        df = pd.read_csv(
            csv_path,
            dtype=str,
            engine="python",  # necessário para on_bad_lines com callable
            on_bad_lines=_skip_bad_line,
            encoding="utf-8",
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Não foi possível ler {csv_path}: {exc}") from exc

    log.info("styles_loaded", rows=len(df), malformed_lines_skipped=len(bad_lines))
    return df
