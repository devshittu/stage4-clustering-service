"""
FAISS GPU/CPU Fallback Utilities.

Provides intelligent fallback between faiss-gpu and faiss-cpu,
with automatic detection and graceful degradation.
"""

import logging
import sys
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global FAISS module reference
_faiss_module: Optional[Any] = None
_faiss_available: bool = False
_using_gpu: bool = False


def _import_faiss():
    """
    Import FAISS with GPU fallback to CPU.

    Attempts to import in this order:
    1. faiss-gpu (if GPU available)
    2. faiss-cpu (fallback)

    Returns:
        faiss module

    Raises:
        ImportError: If neither faiss-gpu nor faiss-cpu available
    """
    global _faiss_module, _faiss_available, _using_gpu

    if _faiss_module is not None:
        return _faiss_module

    # Try faiss-gpu first
    try:
        import faiss

        # Check if GPU is available
        try:
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                logger.info("FAISS-GPU available and GPU detected")
                _using_gpu = True
                _faiss_module = faiss
                _faiss_available = True
                return faiss
            else:
                logger.info("FAISS imported but no GPU detected, using CPU mode")
                _using_gpu = False
                _faiss_module = faiss
                _faiss_available = True
                return faiss
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}. Using CPU mode.")
            _using_gpu = False
            _faiss_module = faiss
            _faiss_available = True
            return faiss

    except ImportError as e:
        logger.warning(f"FAISS import failed: {e}")
        raise ImportError(
            "FAISS not available. Install with: "
            "pip install faiss-gpu (for GPU) or pip install faiss-cpu (for CPU)"
        )


def get_faiss():
    """
    Get FAISS module with GPU/CPU fallback.

    Returns:
        faiss module

    Raises:
        ImportError: If FAISS not available
    """
    return _import_faiss()


def is_faiss_available() -> bool:
    """
    Check if FAISS is available.

    Returns:
        True if FAISS can be imported
    """
    try:
        _import_faiss()
        return _faiss_available
    except ImportError:
        return False


def is_gpu_available() -> bool:
    """
    Check if FAISS GPU is available.

    Returns:
        True if FAISS-GPU is available and working
    """
    if not is_faiss_available():
        return False

    try:
        faiss = get_faiss()
        return _using_gpu and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
    except Exception:
        return False


def get_gpu_resources():
    """
    Get FAISS GPU resources if available.

    Returns:
        StandardGpuResources if GPU available, None otherwise
    """
    if not is_gpu_available():
        return None

    try:
        faiss = get_faiss()
        return faiss.StandardGpuResources()
    except Exception as e:
        logger.warning(f"Failed to create GPU resources: {e}")
        return None


def move_index_to_gpu(index, gpu_id: int = 0):
    """
    Move FAISS index to GPU if available.

    Args:
        index: FAISS index
        gpu_id: GPU device ID

    Returns:
        GPU index if successful, original index otherwise
    """
    if not is_gpu_available():
        logger.info("GPU not available, keeping index on CPU")
        return index

    try:
        faiss = get_faiss()
        gpu_resources = get_gpu_resources()

        if gpu_resources is None:
            logger.warning("Failed to get GPU resources, keeping index on CPU")
            return index

        logger.info(f"Moving index to GPU {gpu_id}")
        return faiss.index_cpu_to_gpu(gpu_resources, gpu_id, index)

    except Exception as e:
        logger.warning(f"Failed to move index to GPU: {e}. Using CPU.")
        return index


def safe_get_vectors_from_index(index, n_vectors: int, dimension: int):
    """
    Safely extract vectors from FAISS index with CPU/GPU compatibility.

    Args:
        index: FAISS index
        n_vectors: Number of vectors
        dimension: Vector dimension

    Returns:
        numpy array of vectors
    """
    import numpy as np

    faiss = get_faiss()

    try:
        # Method 1: Try rev_swig_ptr (works with some FAISS versions)
        if hasattr(faiss, 'rev_swig_ptr'):
            vectors = faiss.rev_swig_ptr(index.get_xb(), n_vectors * dimension)
            return vectors.reshape(n_vectors, dimension)
    except Exception as e:
        logger.debug(f"rev_swig_ptr failed: {e}, trying reconstruct_n")

    try:
        # Method 2: Use reconstruct_n (more compatible)
        vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
        index.reconstruct_n(0, n_vectors, vectors)
        return vectors
    except Exception as e:
        logger.debug(f"reconstruct_n failed: {e}, trying reconstruct loop")

    # Method 3: Fallback to individual reconstruction (slowest but most compatible)
    vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
    for i in range(n_vectors):
        vectors[i] = index.reconstruct(i)

    return vectors


def log_faiss_info():
    """Log FAISS configuration information."""
    if not is_faiss_available():
        logger.warning("FAISS not available")
        return

    faiss = get_faiss()
    gpu_available = is_gpu_available()

    info = {
        "faiss_available": True,
        "gpu_available": gpu_available,
        "using_gpu": _using_gpu,
    }

    if gpu_available:
        try:
            info["num_gpus"] = faiss.get_num_gpus()
        except Exception:
            info["num_gpus"] = 0

    logger.info(f"FAISS configuration: {info}")
    return info
