"""Logging estruturado com structlog."""

from __future__ import annotations

import logging

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configura o structlog para todo o processo.

    Args:
        level: Nível mínimo de log (DEBUG, INFO, WARNING, ERROR).
        json_logs: Se True, renderiza JSON (uma linha por evento); caso
            contrário, usa saída colorida legível para desenvolvimento.
    """
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Retorna um logger estruturado nomeado.

    Args:
        name: Nome do módulo (normalmente ``__name__``).

    Returns:
        Logger que aceita pares chave/valor: ``log.info("evento", n=10)``.
    """
    return structlog.get_logger(name)
