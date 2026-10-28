"""Checagem pré-deploy: o índice e as imagens estão prontos para ir ao repositório?

Valida, sem carregar o CLIP:
    1. o índice existe e abre (coleção de imagens, modelo, quantidade);
    2. toda imagem referenciada nos metadados existe e usa caminho relativo (portável);
    3. nenhum arquivo passa do limite de 100 MB do GitHub e o total é razoável;
    4. (se houver git) as pastas necessárias não estão no .gitignore.

Uso:
    python scripts/check_deploy.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings, resolve_project_path  # noqa: E402
from src.logger import configure_logging, get_logger  # noqa: E402
from src.vectorstore.base import VectorStoreError  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402

log = get_logger("check_deploy")
GITHUB_FILE_LIMIT = 100 * 1024 * 1024
WARN_FILE_SIZE = 90 * 1024 * 1024
WARN_TOTAL_SIZE = 500 * 1024 * 1024


def _is_git_ignored(path: Path) -> bool | None:
    """Retorna True/False se o git souber responder; None se não houver git/repositório."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    return False if result.returncode == 1 else None


def main() -> int:
    """Executa as checagens e imprime um relatório. Retorna 0 se tudo estiver OK."""
    settings = get_settings()
    configure_logging("WARNING", False)
    problems: list[str] = []
    warnings: list[str] = []

    try:
        store = ChromaStore.open_existing(settings.vectorstore_dir, f"{settings.collection_name}_image")
    except VectorStoreError as exc:
        print(f"[ERRO] {exc}")
        return 1
    print(f"Índice: {store.count()} produtos | modelo: {store.model_name} | dim: {store.dim}")

    missing: list[str] = []
    absolute: list[str] = []
    image_files: set[Path] = set()
    for page in store.iter_metadatas():
        for m in page:
            raw = str(m.get("image_path", ""))
            if not raw:
                missing.append(str(m.get("product_id")))
                continue
            if Path(raw).is_absolute():
                absolute.append(raw)
            path = resolve_project_path(raw)
            if path.is_file():
                image_files.add(path)
            else:
                missing.append(raw)
    if absolute:
        problems.append(f"{len(absolute)} imagens com caminho absoluto (não funcionam em outra máquina). Ex.: {absolute[0]}")
    if missing:
        problems.append(f"{len(missing)} imagens referenciadas não existem em disco. Ex.: {missing[0]}")

    index_files = [p for p in settings.vectorstore_dir.rglob("*") if p.is_file()]
    biggest = max(index_files, key=lambda p: p.stat().st_size, default=None)
    total = sum(p.stat().st_size for p in index_files) + sum(p.stat().st_size for p in image_files)
    if biggest is not None and biggest.stat().st_size >= GITHUB_FILE_LIMIT:
        problems.append(f"{biggest.name} tem {biggest.stat().st_size / 1e6:.0f} MB e passa do limite de 100 MB do GitHub. Use menos produtos (--limit).")
    elif biggest is not None and biggest.stat().st_size >= WARN_FILE_SIZE:
        warnings.append(f"{biggest.name} está perto do limite de 100 MB do GitHub ({biggest.stat().st_size / 1e6:.0f} MB).")
    if total >= WARN_TOTAL_SIZE:
        warnings.append(f"Dados a versionar somam {total / 1e6:.0f} MB; considere um subconjunto menor.")
    print(f"Dados a versionar: {len(index_files)} arquivos do índice + {len(image_files)} imagens = {total / 1e6:.1f} MB")

    for label, folder in (("índice", settings.vectorstore_dir), ("imagens", settings.processed_images_dir)):
        probe = next((p for p in folder.rglob("*") if p.is_file()), None)
        if probe is not None and _is_git_ignored(probe) is True:
            problems.append(f"A pasta de {label} está no .gitignore ({folder}); o deploy não a receberá.")

    for w in warnings:
        print(f"[AVISO] {w}")
    for p in problems:
        print(f"[ERRO] {p}")
    if not problems:
        print("OK: pronto para commit e deploy." if not warnings else "OK, com avisos acima.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
