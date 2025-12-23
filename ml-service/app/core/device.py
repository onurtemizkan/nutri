"""
Device Detection and Management for ML Models

Supports automatic detection and configuration of:
- NVIDIA CUDA (cloud GPU instances)
- Apple Metal/MPS (local Mac development)
- CPU fallback (Docker on Mac, low-cost instances)

Usage:
    from app.core.device import get_device, get_device_info

    device = get_device()  # Returns "cuda", "mps", or "cpu"
    info = get_device_info()  # Returns detailed device information
"""

import logging
from typing import Dict, Any
from functools import lru_cache

import torch

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> str:
    """
    Get the best available compute device.

    Priority order (when compute_device="auto"):
    1. CUDA (NVIDIA GPU) - Best for production cloud deployments
    2. MPS (Apple Metal) - Best for local Mac development
    3. CPU - Fallback for Docker on Mac, low-cost instances

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    requested_device = settings.compute_device.lower()

    # If specific device requested, validate and return
    if requested_device != "auto":
        if requested_device == "cuda" and torch.cuda.is_available():
            logger.info("device_selected", device="cuda", reason="explicitly requested")
            return "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            logger.info("device_selected", device="mps", reason="explicitly requested")
            return "mps"
        elif requested_device == "cpu":
            logger.info("device_selected", device="cpu", reason="explicitly requested")
            return "cpu"
        else:
            logger.warning(
                "device_not_available",
                requested=requested_device,
                fallback="auto-detect",
            )

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(
            "device_selected",
            device="cuda",
            gpu=gpu_name,
            reason="NVIDIA GPU detected",
        )
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info(
            "device_selected",
            device="mps",
            reason="Apple Metal GPU detected",
        )
    else:
        device = "cpu"
        logger.info(
            "device_selected",
            device="cpu",
            reason="No GPU available, using CPU",
        )

    return device


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about the compute device.

    Returns:
        Dict with device type, name, memory, and capabilities
    """
    device = get_device()
    info: Dict[str, Any] = {
        "device": device,
        "pytorch_version": torch.__version__,
    }

    if device == "cuda":
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "memory_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                ),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated(0) / (1024**3), 2
                ),
            }
        )
    elif device == "mps":
        info.update(
            {
                "gpu_name": "Apple Metal GPU",
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            }
        )
    else:
        import multiprocessing

        info.update(
            {
                "cpu_count": multiprocessing.cpu_count(),
                "note": "GPU not available - using CPU inference (slower)",
            }
        )

    return info


def log_device_info() -> None:
    """Log device information at startup."""
    info = get_device_info()
    device = info["device"]

    if device == "cuda":
        logger.info(
            "compute_device_info",
            device="cuda",
            gpu=info["gpu_name"],
            memory_gb=info["memory_total_gb"],
            cuda=info["cuda_version"],
        )
    elif device == "mps":
        logger.info(
            "compute_device_info",
            device="mps",
            gpu="Apple Metal",
        )
    else:
        logger.info(
            "compute_device_info",
            device="cpu",
            cores=info["cpu_count"],
            warning="Consider GPU for faster inference",
        )
