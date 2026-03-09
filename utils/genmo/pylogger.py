"""Minimal logger API compatible with GENMO's Log usage."""
# Derived from: /simurgh/u/askhan1/winter26/Video-as-Action-Prompt/genmo/utils/pylogger.py
# Simplified for local rendering utilities.
from __future__ import annotations

import logging
import time


class _Log:
    _logger = None

    @classmethod
    def _get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger("genmo")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%m/%d %H:%M:%S")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            cls._logger = logger
        return cls._logger

    @staticmethod
    def time() -> float:
        return time.time()

    @classmethod
    def info(cls, msg: str) -> None:
        cls._get_logger().info(msg)

    @classmethod
    def warn(cls, msg: str) -> None:
        cls._get_logger().warning(msg)


Log = _Log
