"""
统一日志模块 —— rich 格式化，带时间戳和模块名，输出到 stderr。
用法:
    from core.logger import get_logger
    logger = get_logger(__name__)
"""
import logging
import os
from rich.logging import RichHandler
from rich.console import Console

_console = Console(stderr=True)
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level_name, logging.INFO),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=_console,
                    rich_tracebacks=True,
                    show_path=False,
                    markup=True,
                )
            ],
        )
        _configured = True
    return logging.getLogger(name)
