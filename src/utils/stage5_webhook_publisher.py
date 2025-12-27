"""
Stage 5 Webhook Publisher.

Sends HTTP webhook notifications to Stage 5 (Graph Construction Service)
when clustering jobs complete, enabling downstream automation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
import structlog

from src.config.config import get_config

logger = structlog.get_logger(__name__)


class Stage5WebhookPublisher:
    """
    Webhook publisher for notifying Stage 5 of clustering completion.

    Sends HTTP POST requests to Stage 5's webhook endpoint with cluster
    metadata and output file paths, enabling automatic graph construction.

    Features:
    - Multiple webhook URLs for redundancy
    - Retry logic with exponential backoff
    - Fail-safe mode (don't block on failure)
    - Optional authentication
    - Async/non-blocking HTTP requests
    """

    def __init__(self):
        """Initialize the webhook publisher."""
        self.config = settings.downstream_automation.webhook_publisher

        # Webhook URLs
        self.webhook_urls = self.config.stage5_urls
        self.enabled = self.config.enabled

        # Retry configuration
        self.max_attempts = self.config.retry.max_attempts
        self.backoff_seconds = self.config.retry.backoff_seconds
        self.timeout_seconds = self.config.retry.timeout_seconds

        # Fail silently (don't raise errors if Stage 5 is unavailable)
        self.fail_silently = self.config.fail_silently

        # Auth token
        self.auth_token = self.config.auth_token
        if self.auth_token and self.auth_token.startswith("${"):
            # Environment variable not expanded
            self.auth_token = None

        # HTTP client (reusable)
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=10),
            )
        return self.client

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def publish_clustering_completed(
        self,
        job_id: str,
        embedding_type: str,
        algorithm: str,
        clusters_created: int,
        outliers: int,
        output_files: List[str],
        quality_metrics: Optional[Dict[str, float]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        sample_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Publish clustering completion event to Stage 5 webhook.

        Args:
            job_id: Clustering job ID
            embedding_type: Type of embeddings (document/event/entity/storyline)
            algorithm: Clustering algorithm used
            clusters_created: Number of clusters created
            outliers: Number of outliers
            output_files: List of output file paths (JSONL)
            quality_metrics: Quality metrics (silhouette, etc.)
            statistics: Additional statistics
            sample_clusters: Sample cluster data

        Returns:
            True if webhook succeeded (or disabled), False if failed
        """
        if not self.enabled:
            logger.debug("stage5_webhook_disabled")
            return True  # Not an error

        if not self.webhook_urls:
            logger.warning("stage5_webhook_no_urls_configured")
            return True  # Not an error

        # Build payload
        payload = {
            "event_type": "clustering.job.completed",
            "job_id": job_id,
            "embedding_type": embedding_type,
            "algorithm": algorithm,
            "clusters_created": clusters_created,
            "outliers": outliers,
            "output_files": output_files,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Add optional fields
        if quality_metrics:
            payload["quality_metrics"] = quality_metrics

        if statistics:
            payload["statistics"] = statistics

        if sample_clusters:
            payload["sample_clusters"] = sample_clusters

        # Try each webhook URL
        success_count = 0
        for url in self.webhook_urls:
            try:
                success = await self._send_webhook(url, payload)
                if success:
                    success_count += 1

            except Exception as e:
                logger.error(
                    "stage5_webhook_unexpected_error",
                    url=url,
                    job_id=job_id,
                    error=str(e),
                    exc_info=True,
                )

        # Consider successful if at least one webhook succeeded
        if success_count > 0:
            logger.info(
                "stage5_webhook_success",
                job_id=job_id,
                successful_webhooks=success_count,
                total_webhooks=len(self.webhook_urls),
            )
            return True
        else:
            if self.fail_silently:
                logger.warning(
                    "stage5_webhook_all_failed_but_continuing",
                    job_id=job_id,
                    total_webhooks=len(self.webhook_urls),
                )
                return True  # Don't block job completion
            else:
                logger.error(
                    "stage5_webhook_all_failed",
                    job_id=job_id,
                    total_webhooks=len(self.webhook_urls),
                )
                return False

    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """
        Send webhook to a single URL with retries.

        Args:
            url: Webhook URL
            payload: Event payload

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()

        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Retry loop
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(
                    "stage5_webhook_sending",
                    url=url,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                )

                # Send POST request
                response = await client.post(url, json=payload, headers=headers)

                # Check response status
                if response.status_code in [200, 201, 202, 204]:
                    logger.info(
                        "stage5_webhook_sent",
                        url=url,
                        status_code=response.status_code,
                        job_id=payload["job_id"],
                    )
                    return True

                elif response.status_code == 401:
                    logger.error(
                        "stage5_webhook_unauthorized",
                        url=url,
                        status_code=response.status_code,
                    )
                    return False  # Don't retry auth failures

                elif response.status_code >= 500:
                    # Server error, retry
                    logger.warning(
                        "stage5_webhook_server_error_retrying",
                        url=url,
                        status_code=response.status_code,
                        attempt=attempt,
                    )

                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.backoff_seconds * attempt)
                        continue
                    else:
                        logger.error(
                            "stage5_webhook_failed_after_retries",
                            url=url,
                            status_code=response.status_code,
                            max_attempts=self.max_attempts,
                        )
                        return False

                else:
                    # Client error (4xx), don't retry
                    logger.error(
                        "stage5_webhook_client_error",
                        url=url,
                        status_code=response.status_code,
                        response_text=response.text,
                    )
                    return False

            except httpx.TimeoutException:
                logger.warning(
                    "stage5_webhook_timeout",
                    url=url,
                    attempt=attempt,
                    timeout_seconds=self.timeout_seconds,
                )

                if attempt < self.max_attempts:
                    await asyncio.sleep(self.backoff_seconds * attempt)
                    continue
                else:
                    logger.error(
                        "stage5_webhook_timeout_after_retries",
                        url=url,
                        max_attempts=self.max_attempts,
                    )
                    return False

            except httpx.ConnectError as e:
                logger.warning(
                    "stage5_webhook_connection_error",
                    url=url,
                    attempt=attempt,
                    error=str(e),
                )

                if attempt < self.max_attempts:
                    await asyncio.sleep(self.backoff_seconds * attempt)
                    continue
                else:
                    logger.error(
                        "stage5_webhook_connection_failed_after_retries",
                        url=url,
                        error=str(e),
                        max_attempts=self.max_attempts,
                    )
                    return False

            except Exception as e:
                logger.error(
                    "stage5_webhook_unexpected_error",
                    url=url,
                    attempt=attempt,
                    error=str(e),
                    exc_info=True,
                )

                if attempt < self.max_attempts:
                    await asyncio.sleep(self.backoff_seconds * attempt)
                    continue
                else:
                    return False

        return False  # All retries exhausted


# Global instance
_publisher: Optional[Stage5WebhookPublisher] = None


def get_stage5_webhook_publisher() -> Stage5WebhookPublisher:
    """
    Get global Stage 5 webhook publisher instance.

    Returns:
        Stage5WebhookPublisher instance
    """
    global _publisher
    if _publisher is None:
        _publisher = Stage5WebhookPublisher()
    return _publisher


async def publish_to_stage5(
    job_id: str,
    embedding_type: str,
    algorithm: str,
    clusters_created: int,
    outliers: int,
    output_files: List[str],
    quality_metrics: Optional[Dict[str, float]] = None,
    statistics: Optional[Dict[str, Any]] = None,
    sample_clusters: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Convenience function to publish clustering completion to Stage 5.

    Args:
        job_id: Clustering job ID
        embedding_type: Embedding type
        algorithm: Algorithm used
        clusters_created: Number of clusters
        outliers: Number of outliers
        output_files: Output file paths
        quality_metrics: Quality metrics
        statistics: Statistics
        sample_clusters: Sample clusters

    Returns:
        True if successful
    """
    publisher = get_stage5_webhook_publisher()
    return await publisher.publish_clustering_completed(
        job_id=job_id,
        embedding_type=embedding_type,
        algorithm=algorithm,
        clusters_created=clusters_created,
        outliers=outliers,
        output_files=output_files,
        quality_metrics=quality_metrics,
        statistics=statistics,
        sample_clusters=sample_clusters,
    )
