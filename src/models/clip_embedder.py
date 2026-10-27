"""Backend CLIP (Hugging Face transformers)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.logger import get_logger
from src.models.base import EmbeddingBackend, EmbeddingError

log = get_logger(__name__)

CLIP_CONTEXT_LENGTH = 77  # limite de tokens do encoder de texto do CLIP


def resolve_device(requested: str = "auto") -> torch.device:
    """Escolhe o dispositivo de inferência.

    Args:
        requested: ``auto`` (cuda > mps > cpu) ou um nome explícito de device.

    Returns:
        ``torch.device`` pronto para uso.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClipEmbedder(EmbeddingBackend):
    """Gera embeddings de imagem e texto com um checkpoint CLIP.

    Em GPU CUDA o modelo roda em fp16 (metade da VRAM, ~2x mais rápido); em
    CPU/MPS permanece em fp32. A normalização L2 é sempre feita em fp32.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        """Carrega modelo e processador.

        Args:
            model_name: Checkpoint no Hugging Face Hub.
            device: ``auto``, ``cpu``, ``cuda`` ou ``mps``.

        Raises:
            EmbeddingError: Se o download/carregamento do modelo falhar.
        """
        self._model_name = model_name
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise EmbeddingError(
                f"Não foi possível carregar '{model_name}': {exc}. "
                "Confira o nome do checkpoint e a conexão (o primeiro uso baixa ~600 MB)."
            ) from exc
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self._dim = int(self.model.config.projection_dim)
        log.info("clip_loaded", model=model_name, device=str(self.device), dtype=str(self.dtype), dim=self._dim)

    @property
    def name(self) -> str:
        """Nome do checkpoint carregado."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Dimensão do espaço de embedding (``projection_dim`` do CLIP)."""
        return self._dim

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Gera embeddings normalizados para imagens RGB (ver ``EmbeddingBackend``)."""
        if len(images) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        try:
            inputs = self.processor(images=list(images), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            with torch.inference_mode():
                out = self.model.get_image_features(pixel_values=pixel_values)
            return self._finalize(out, ("image_embeds", "pooler_output"))
        except RuntimeError as exc:
            raise self._wrap_runtime_error(exc) from exc

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Gera embeddings normalizados para textos (ver ``EmbeddingBackend``)."""
        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        try:
            inputs = self.processor(
                text=list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CLIP_CONTEXT_LENGTH,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model.get_text_features(**inputs)
            return self._finalize(out, ("text_embeds", "pooler_output"))
        except RuntimeError as exc:
            raise self._wrap_runtime_error(exc) from exc

    def _finalize(self, output: Any, keys: tuple[str, ...]) -> np.ndarray:
        """Extrai o tensor de features, valida a dimensão e normaliza em L2."""
        feats = output if isinstance(output, torch.Tensor) else next(
            (getattr(output, k) for k in keys if isinstance(getattr(output, k, None), torch.Tensor)),
            None,
        )
        if feats is None:
            raise EmbeddingError(f"Saída inesperada do CLIP: {type(output).__name__}")
        if feats.shape[-1] != self._dim:
            raise EmbeddingError(
                f"Dimensão {feats.shape[-1]} != projection_dim {self._dim}: "
                "a versão do transformers pode ter mudado o retorno de get_*_features."
            )
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _wrap_runtime_error(exc: RuntimeError) -> EmbeddingError:
        """Converte erros de runtime do torch em mensagens acionáveis."""
        if "out of memory" in str(exc).lower():
            return EmbeddingError("Sem memória no dispositivo. Reduza --batch-size e tente novamente.")
        return EmbeddingError(f"Erro durante a inferência do CLIP: {exc}")
