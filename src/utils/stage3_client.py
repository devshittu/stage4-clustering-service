"""
Stage 3 Integration Client.

Provides utilities for checking Stage 3 availability and health.
"""

import logging
import requests
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class Stage3Client:
    """Client for communicating with Stage 3 Embedding Service."""

    def __init__(
        self,
        api_url: str = "http://embeddings-orchestrator:8000",
        timeout: int = 10,
    ):
        """
        Initialize Stage 3 client.

        Args:
            api_url: Base URL for Stage 3 API
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Check Stage 3 service health.

        Returns:
            Health check response dictionary

        Raises:
            Exception: If health check fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Stage 3 health check returned {response.status_code}")
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                }

        except requests.ConnectionError:
            logger.error(f"Failed to connect to Stage 3 at {self.api_url}")
            return {
                "status": "unavailable",
                "error": "Connection failed",
            }
        except requests.Timeout:
            logger.error(f"Stage 3 health check timeout after {self.timeout}s")
            return {
                "status": "timeout",
                "error": f"Timeout after {self.timeout}s",
            }
        except Exception as e:
            logger.error(f"Stage 3 health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def is_available(self) -> bool:
        """
        Quick check if Stage 3 is available.

        Returns:
            True if Stage 3 is healthy, False otherwise
        """
        try:
            health = self.check_health()
            return health.get("status") == "healthy"
        except Exception:
            return False

    def get_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get Stage 3 statistics.

        Returns:
            Statistics dictionary or None if unavailable
        """
        try:
            response = requests.get(
                f"{self.api_url}/statistics",
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Stage 3 statistics returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to get Stage 3 statistics: {e}")
            return None

    def check_indices_ready(self) -> bool:
        """
        Check if FAISS indices are ready.

        Returns:
            True if indices are ready, False otherwise
        """
        try:
            stats = self.get_statistics()
            if stats:
                # Check if Stage 3 has vectors indexed
                total_vectors = stats.get("total_vectors", {})
                for etype, count in total_vectors.items():
                    if count > 0:
                        return True
            return False
        except Exception:
            return False


def get_stage3_client(api_url: Optional[str] = None) -> Stage3Client:
    """
    Factory function to get Stage 3 client.

    Args:
        api_url: Optional override for API URL

    Returns:
        Stage3Client instance
    """
    from src.config.config import get_config

    config = get_config()
    stage3_config = config.get_section("stage3_integration")

    if api_url is None:
        api_url = stage3_config.get("api_url", "http://embeddings-orchestrator:8000")

    timeout = stage3_config.get("health_check_timeout", 10)

    return Stage3Client(api_url=api_url, timeout=timeout)
