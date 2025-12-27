"""
Event Publisher for Job Status Updates.

Publishes clustering job events to Redis Streams, webhooks, and other
event channels for real-time monitoring and integration.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import redis
import httpx


logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Publishes job lifecycle events to multiple channels.

    Supports:
    - Redis Streams (for inter-service communication)
    - Webhooks (for external integrations)
    - NATS (future support)
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        stream_name: str = "stage4:clustering:events",
        webhook_urls: Optional[List[str]] = None,
    ):
        """
        Initialize event publisher.

        Args:
            redis_client: Redis client for stream publishing
            stream_name: Redis stream name
            webhook_urls: List of webhook URLs to notify
        """
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.webhook_urls = webhook_urls or []

    def publish_job_created(
        self,
        job_id: str,
        embedding_type: str,
        algorithm: str,
        **kwargs,
    ):
        """Publish job created event."""
        event = {
            "event_type": "job.created",
            "job_id": job_id,
            "embedding_type": embedding_type,
            "algorithm": algorithm,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_started(
        self,
        job_id: str,
        total_items: int,
        **kwargs,
    ):
        """Publish job started event."""
        event = {
            "event_type": "job.started",
            "job_id": job_id,
            "total_items": total_items,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_progress(
        self,
        job_id: str,
        processed_items: int,
        total_items: int,
        clusters_created: int,
        **kwargs,
    ):
        """Publish job progress event."""
        progress_percent = (processed_items / total_items * 100) if total_items > 0 else 0

        event = {
            "event_type": "job.progress",
            "job_id": job_id,
            "processed_items": processed_items,
            "total_items": total_items,
            "clusters_created": clusters_created,
            "progress_percent": round(progress_percent, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_completed(
        self,
        job_id: str,
        clusters_created: int,
        outliers: int,
        quality_metrics: Dict[str, float],
        processing_time_ms: float,
        output_files: Optional[List[str]] = None,
        embedding_type: Optional[str] = None,
        algorithm: Optional[str] = None,
        statistics: Optional[Dict[str, Any]] = None,
        sample_clusters: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Publish job completed event.

        Args:
            job_id: Clustering job ID
            clusters_created: Number of clusters created
            outliers: Number of outlier points
            quality_metrics: Quality metrics (silhouette, etc.)
            processing_time_ms: Total processing time
            output_files: List of output file paths (JSONL, etc.)
            embedding_type: Type of embeddings (document/event/entity/storyline)
            algorithm: Clustering algorithm used
            statistics: Additional statistics
            sample_clusters: Sample of cluster data for preview
            **kwargs: Additional metadata
        """
        event = {
            "event_type": "job.completed",
            "job_id": job_id,
            "clusters_created": clusters_created,
            "outliers": outliers,
            "quality_metrics": quality_metrics,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }

        # Add output files if provided (for Stage 5 consumption)
        if output_files:
            event["output_files"] = output_files

        # Add embedding type if provided
        if embedding_type:
            event["embedding_type"] = embedding_type

        # Add algorithm if provided
        if algorithm:
            event["algorithm"] = algorithm

        # Add statistics if provided
        if statistics:
            event["statistics"] = statistics

        # Add sample clusters if provided
        if sample_clusters:
            event["sample_clusters"] = sample_clusters

        self._publish(event)

    def publish_job_failed(
        self,
        job_id: str,
        error_message: str,
        **kwargs,
    ):
        """Publish job failed event."""
        event = {
            "event_type": "job.failed",
            "job_id": job_id,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_paused(
        self,
        job_id: str,
        processed_items: int,
        **kwargs,
    ):
        """Publish job paused event."""
        event = {
            "event_type": "job.paused",
            "job_id": job_id,
            "processed_items": processed_items,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_resumed(
        self,
        job_id: str,
        **kwargs,
    ):
        """Publish job resumed event."""
        event = {
            "event_type": "job.resumed",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def publish_job_canceled(
        self,
        job_id: str,
        **kwargs,
    ):
        """Publish job canceled event."""
        event = {
            "event_type": "job.canceled",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self._publish(event)

    def _publish(self, event: Dict[str, Any]):
        """
        Publish event to all configured channels.

        Args:
            event: Event dictionary
        """
        # Publish to Redis Stream
        if self.redis_client:
            self._publish_redis_stream(event)

        # Publish to Webhooks
        if self.webhook_urls:
            self._publish_webhooks(event)

    def _publish_redis_stream(self, event: Dict[str, Any]):
        """Publish to Redis Stream."""
        try:
            # Convert all values to strings for Redis
            event_data = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in event.items()}

            self.redis_client.xadd(
                self.stream_name,
                event_data,
                maxlen=10000,  # Keep last 10K events
                approximate=True,
            )

            logger.debug(
                f"Published event to Redis stream: {event['event_type']} for job {event.get('job_id')}"
            )

        except Exception as e:
            logger.error(f"Failed to publish to Redis stream: {e}")

    def _publish_webhooks(self, event: Dict[str, Any]):
        """Publish to webhooks (non-blocking)."""
        for webhook_url in self.webhook_urls:
            try:
                # Webhook call with timeout
                with httpx.Client(timeout=5.0) as client:
                    client.post(
                        webhook_url,
                        json=event,
                        headers={"Content-Type": "application/json"},
                    )

                logger.debug(f"Published event to webhook: {webhook_url}")

            except httpx.TimeoutException:
                logger.warning(f"Webhook timeout: {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to publish to webhook {webhook_url}: {e}")


def get_redis_client(decode_responses: bool = False) -> redis.Redis:
    """
    Get Redis client for event publishing.

    Args:
        decode_responses: Whether to decode responses

    Returns:
        Redis client instance
    """
    from src.utils.redis_client import get_broker_redis_client

    return get_broker_redis_client(decode_responses=decode_responses)


def get_event_publisher(
    redis_client: Optional[redis.Redis] = None,
) -> EventPublisher:
    """
    Factory function to get event publisher.

    Args:
        redis_client: Redis client (optional, will create if None)

    Returns:
        EventPublisher instance
    """
    from src.config.config import get_config

    config = get_config()

    # Get stream configuration
    try:
        event_config = config.get_section("event_streaming")
    except (KeyError, AttributeError):
        event_config = {}

    stream_name = event_config.get("redis_stream_name", "stage4:clustering:events")
    webhook_urls = event_config.get("webhook_urls", [])

    # Use provided client or create new one
    if redis_client is None and event_config.get("enabled", True):
        redis_client = get_redis_client(decode_responses=False)

    return EventPublisher(
        redis_client=redis_client,
        stream_name=stream_name,
        webhook_urls=webhook_urls,
    )
