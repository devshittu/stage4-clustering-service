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
import requests


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
        **kwargs,
    ):
        """Publish job completed event."""
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
                # Async webhook call with timeout
                requests.post(
                    webhook_url,
                    json=event,
                    timeout=5,
                    headers={"Content-Type": "application/json"},
                )

                logger.debug(f"Published event to webhook: {webhook_url}")

            except requests.Timeout:
                logger.warning(f"Webhook timeout: {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to publish to webhook {webhook_url}: {e}")


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
    event_config = config.get_section("event_streaming", {})
    stream_name = event_config.get("redis_stream_name", "stage4:clustering:events")
    webhook_urls = event_config.get("webhook_urls", [])

    # Use provided client or create new one
    if redis_client is None and event_config.get("enabled", True):
        redis_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
        redis_client = redis.from_url(redis_url, decode_responses=False)

    return EventPublisher(
        redis_client=redis_client,
        stream_name=stream_name,
        webhook_urls=webhook_urls,
    )
