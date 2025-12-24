"""
resource_manager.py

Resource management utility for Stage 4 Clustering Service.
Monitors GPU/CPU/RAM usage, detects idle periods, and manages cleanup.

Features:
- GPU memory monitoring (nvidia-smi, GPUtil)
- CPU/RAM monitoring (psutil)
- Idle detection with configurable timeout
- Cleanup hooks for resource release
- Context manager for resource allocation
"""

import logging
import time
import subprocess
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from contextlib import contextmanager
import psutil

logger = logging.getLogger(__name__)

# Global singleton instance
_resource_manager_instance: Optional["ResourceManager"] = None


class ResourceManager:
    """
    Manages system resources (CPU, RAM, GPU).

    Features:
    - Resource utilization monitoring
    - Idle mode to conserve resources
    - GPU memory threshold monitoring
    - Cleanup hooks
    """

    def __init__(
        self,
        idle_timeout_seconds: int = 300,
        gpu_memory_threshold_mb: int = 14000,
        enable_idle_mode: bool = True,
    ):
        """
        Initialize resource manager.

        Args:
            idle_timeout_seconds: Seconds before entering idle mode
            gpu_memory_threshold_mb: GPU memory threshold for warnings
            enable_idle_mode: Enable idle mode optimization
        """
        self.idle_timeout = idle_timeout_seconds
        self.gpu_memory_threshold = gpu_memory_threshold_mb
        self.enable_idle_mode = enable_idle_mode
        self.last_activity = time.time()
        self.gpu_available = self._check_gpu_availability()

        logger.info(
            f"ResourceManager initialized (idle_timeout={idle_timeout_seconds}s, "
            f"gpu_threshold={gpu_memory_threshold_mb}MB, gpu_available={self.gpu_available})"
        )

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def record_activity(self) -> None:
        """Record activity to prevent idle mode."""
        self.last_activity = time.time()

    def is_idle(self) -> bool:
        """Check if system is idle."""
        if not self.enable_idle_mode:
            return False

        idle_duration = time.time() - self.last_activity
        return idle_duration > self.idle_timeout

    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get current resource utilization.

        Returns:
            Dictionary with CPU, RAM, and GPU stats
        """
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": self.gpu_available,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if self.gpu_available:
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**2)  # MB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)

                    stats["gpu_utilization_percent"] = (gpu_mem_allocated / gpu_mem_total) * 100
                    stats["gpu_memory_used_mb"] = gpu_mem_allocated
                    stats["gpu_memory_reserved_mb"] = gpu_mem_reserved
                    stats["gpu_memory_total_mb"] = gpu_mem_total
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
                stats["gpu_utilization_percent"] = None
                stats["gpu_memory_used_mb"] = None
                stats["gpu_memory_total_mb"] = None

        return stats

    def check_gpu_memory(self) -> bool:
        """
        Check GPU memory usage and clear cache if needed.

        Returns:
            True if memory is within threshold
        """
        if not self.gpu_available:
            return True

        try:
            import torch

            if not torch.cuda.is_available():
                return True

            mem_allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB

            if mem_allocated > self.gpu_memory_threshold:
                logger.warning(
                    f"GPU memory usage ({mem_allocated:.0f}MB) exceeds threshold "
                    f"({self.gpu_memory_threshold}MB). Clearing cache..."
                )
                torch.cuda.empty_cache()
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check GPU memory: {e}")
            return True

    def cleanup_on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("ResourceManager cleanup on shutdown")

        if self.gpu_available:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU cache")
            except Exception as e:
                logger.error(f"Failed to clear GPU cache: {e}")


def get_resource_manager(
    idle_timeout_seconds: int = 300,
    gpu_memory_threshold_mb: int = 14000,
    enable_idle_mode: bool = True,
) -> ResourceManager:
    """
    Get or create singleton ResourceManager instance.

    Args:
        idle_timeout_seconds: Seconds before entering idle mode
        gpu_memory_threshold_mb: GPU memory threshold
        enable_idle_mode: Enable idle mode

    Returns:
        ResourceManager singleton instance
    """
    global _resource_manager_instance

    if _resource_manager_instance is None:
        _resource_manager_instance = ResourceManager(
            idle_timeout_seconds=idle_timeout_seconds,
            gpu_memory_threshold_mb=gpu_memory_threshold_mb,
            enable_idle_mode=enable_idle_mode,
        )

    return _resource_manager_instance


def cleanup_resources() -> None:
    """Cleanup all resources (for worker shutdown)."""
    global _resource_manager_instance

    if _resource_manager_instance:
        _resource_manager_instance.cleanup_on_shutdown()
        _resource_manager_instance = None
        logger.info("Resource manager cleaned up")
