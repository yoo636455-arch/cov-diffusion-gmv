"""
utils.py
--------
Shared utilities.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def get_device() -> "torch.device":
    """
    Return the best available compute device:
      CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy, random, and torch (if available)."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with a StreamHandler if none exists."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(obj: Any) -> int:
    """MD5-based stable hash of a string representation."""
    key = str(obj).encode()
    return int(hashlib.md5(key).hexdigest(), 16) % (2**31)


def annualized_vol(daily_returns: np.ndarray) -> float:
    """sqrt(252) * std(r)."""
    return float(np.sqrt(252) * np.std(daily_returns, ddof=1))
