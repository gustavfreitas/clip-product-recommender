"""Montagem visual dos resultados (folha de contato) para inspeção e portfólio."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.config import resolve_project_path
from src.logger import get_logger
from src.vectorstore.base import Hit

log = get_logger(__name__)

_BG = (250, 250, 250)
_INK = (30, 30, 30)
_MUTED = (110, 110, 110)
_ACCENT = (20, 90, 200)


def _load_thumb(path: str | None, size: int) -> Image.Image:
    """Abre a miniatura quadrada de um produto; usa um quadro cinza se falhar."""
    if path:
        try:
            with Image.open(resolve_project_path(path)) as im:
                return ImageOps.fit(im.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
        except (OSError, ValueError) as exc:
            log.warning("thumb_failed", path=path, error=str(exc))
    return Image.new("RGB", (size, size), (225, 225, 225))


def make_contact_sheet(
    hits: Sequence[Hit],
    query_image: Image.Image | None = None,
    *,
    thumb: int = 180,
    caption: str = "",
) -> Image.Image:
    """Gera uma imagem com a consulta (opcional) e os resultados lado a lado.

    Cada célula mostra a miniatura, o score de cosseno, o nome e o preço.

    Args:
        hits: Resultados em ordem de ranking.
        query_image: Imagem de consulta, exibida à esquerda quando fornecida.
        thumb: Lado (px) das miniaturas.
        caption: Texto exibido no topo (ex.: a consulta de texto).

    Returns:
        Imagem PIL em RGB.
    """
    font = ImageFont.load_default()
    cells: list[tuple[Image.Image, list[tuple[str, tuple[int, int, int]]]]] = []
    if query_image is not None:
        cells.append((ImageOps.fit(query_image.convert("RGB"), (thumb, thumb)), [("CONSULTA", _ACCENT)]))
    for rank, h in enumerate(hits, 1):
        m = h.metadata
        price = m.get("price")
        lines = [
            (f"#{rank}  cos={h.score:.3f}", _ACCENT),
            (str(m.get("name", h.product_id))[:26], _INK),
            (f"R$ {price:.2f}" if isinstance(price, int | float) else "", _MUTED),
        ]
        cells.append((_load_thumb(m.get("image_path"), thumb), lines))

    pad, text_h = 10, 44
    top = 26 if caption else 0
    width = pad + len(cells) * (thumb + pad)
    height = top + pad + thumb + text_h + pad
    sheet = Image.new("RGB", (max(width, 200), height), _BG)
    draw = ImageDraw.Draw(sheet)
    if caption:
        draw.text((pad, 8), caption[:120], fill=_INK, font=font)
    for i, (img, lines) in enumerate(cells):
        x = pad + i * (thumb + pad)
        sheet.paste(img, (x, top + pad))
        for j, (text, color) in enumerate(lines):
            draw.text((x, top + pad + thumb + 4 + j * 13), text, fill=color, font=font)
    return sheet
