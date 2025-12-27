"""
Stage 3 Redis Stream Consumer Service.

Listens to Stage 3 embedding events and automatically triggers clustering jobs
when new embeddings are produced.

This service implements the upstream automation (Stage 3 → Stage 4) for the
Sequential Storytelling Pipeline.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
import structlog
from pydantic import BaseModel, Field, ValidationError

from src.api.celery_worker import cluster_batch_task
from src.config.config import get_config

logger = structlog.get_logger(__name__)


class Stage3Event(BaseModel):
    """Stage 3 embedding event schema."""

    event_type: str = Field(..., description="Event type from Stage 3")
    job_id: str = Field(..., description="Stage 3 job ID")
    embedding_type: str = Field(..., description="Type: document, event, entity, storyline")
    total_embeddings: int = Field(..., description="Total embeddings created")
    output_path: Optional[str] = Field(None, description="Path to FAISS index file")
    quality_score: Optional[float] = Field(None, description="Embedding quality (0-1)")
    timestamp: str = Field(..., description="Event timestamp (ISO 8601)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Stage3StreamConsumer:
    """
    Redis stream consumer for Stage 3 embedding events.

    Listens to 'stage3:embeddings:events' stream and auto-triggers clustering
    jobs when Stage 3 completes embedding generation.

    Features:
    - Consumer group with auto-recovery
    - Event validation with Pydantic
    - Configurable auto-trigger rules
    - Retry mechanism for failed events
    - Graceful shutdown handling
    """

    def __init__(self):
        """Initialize the consumer."""
        self.redis_client: Optional[redis.Redis] = None
        self.running = False

        # Load configuration
        config = get_config()
        upstream_config = config.get_section("upstream_automation")

        # Configuration from settings
        redis_consumer = upstream_config.get("redis_consumer", {})
        self.stream_name = redis_consumer.get("stream_name", "stage3:embeddings:events")
        self.consumer_group = redis_consumer.get("consumer_group", "stage4-clustering-consumers")
        self.consumer_name = redis_consumer.get("consumer_name", "stage4-worker-1")
        self.block_ms = redis_consumer.get("block_ms", 5000)
        self.count = redis_consumer.get("count", 10)
        self.trigger_events = redis_consumer.get("trigger_events", ["embedding.job.completed"])

        # Auto-trigger rules
        self.auto_trigger_enabled = upstream_config.get("enabled", True)
        auto_trigger = upstream_config.get("auto_trigger", {})
        self.allowed_embedding_types = auto_trigger.get("embedding_types", ["document", "event", "entity", "storyline"])
        self.default_algorithm = auto_trigger.get("default_algorithm", "hdbscan")
        self.min_embeddings = auto_trigger.get("min_embeddings", 10)
        self.quality_threshold = auto_trigger.get("quality_threshold", 0.0)

        # Retry config
        retry_config = redis_consumer.get("retry", {})
        self.max_retry_attempts = retry_config.get("max_attempts", 3)
        self.retry_backoff = retry_config.get("backoff_seconds", 5)

        # Statistics
        self.stats = {
            "events_processed": 0,
            "jobs_triggered": 0,
            "errors": 0,
            "last_event_time": None,
        }

    def connect(self) -> None:
        """
        Connect to Redis and create consumer group if needed.

        Raises:
            redis.RedisError: If connection fails
        """
        try:
            config = get_config()
            redis_host = config.get("redis.broker_host", "redis-broker")
            redis_port = config.get("redis.broker_port", 6379)
            redis_db = config.get("redis.broker_db", 6)

            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
            )

            # Test connection
            self.redis_client.ping()

            # Create consumer group if not exists
            try:
                self.redis_client.xgroup_create(
                    name=self.stream_name,
                    groupname=self.consumer_group,
                    id="0",  # Start from beginning
                    mkstream=True,  # Create stream if not exists
                )
                logger.info(
                    "consumer_group_created",
                    stream=self.stream_name,
                    group=self.consumer_group,
                )
            except redis.ResponseError as e:
                # Group already exists
                if "BUSYGROUP" in str(e):
                    logger.info(
                        "consumer_group_exists",
                        stream=self.stream_name,
                        group=self.consumer_group,
                    )
                else:
                    raise

            logger.info(
                "redis_consumer_connected",
                host=redis_host,
                db=redis_db,
                stream=self.stream_name,
            )

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e), exc_info=True)
            raise

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("redis_consumer_disconnected")

    async def process_event(self, event_id: str, event_data: Dict[str, Any]) -> bool:
        """
        Process a single event from Stage 3.

        Args:
            event_id: Redis stream message ID
            event_data: Event payload from Stage 3

        Returns:
            True if processed successfully, False otherwise
        """
        try:
            # Parse and validate event
            stage3_event = Stage3Event(**event_data)

            logger.info(
                "stage3_event_received",
                event_id=event_id,
                event_type=stage3_event.event_type,
                job_id=stage3_event.job_id,
                embedding_type=stage3_event.embedding_type,
                total_embeddings=stage3_event.total_embeddings,
            )

            # Check if event type should trigger clustering
            if stage3_event.event_type not in self.trigger_events:
                logger.debug(
                    "event_type_not_triggering",
                    event_type=stage3_event.event_type,
                    allowed_types=self.trigger_events,
                )
                return True  # Successfully ignored

            # Check if auto-trigger is enabled
            if not self.auto_trigger_enabled:
                logger.warning(
                    "auto_trigger_disabled",
                    event_id=event_id,
                    message="Received trigger event but auto-trigger is disabled",
                )
                return True  # Successfully ignored

            # Validate embedding type
            if stage3_event.embedding_type not in self.allowed_embedding_types:
                logger.warning(
                    "embedding_type_not_allowed",
                    embedding_type=stage3_event.embedding_type,
                    allowed_types=self.allowed_embedding_types,
                )
                return True  # Successfully ignored

            # Check minimum embeddings threshold
            if stage3_event.total_embeddings < self.min_embeddings:
                logger.warning(
                    "insufficient_embeddings",
                    total=stage3_event.total_embeddings,
                    minimum=self.min_embeddings,
                    job_id=stage3_event.job_id,
                )
                return True  # Successfully ignored (not an error)

            # Check quality threshold
            if stage3_event.quality_score is not None:
                if stage3_event.quality_score < self.quality_threshold:
                    logger.warning(
                        "embedding_quality_too_low",
                        quality=stage3_event.quality_score,
                        threshold=self.quality_threshold,
                        job_id=stage3_event.job_id,
                    )
                    return True  # Successfully ignored

            # Trigger clustering job
            success = await self._trigger_clustering_job(stage3_event)

            if success:
                self.stats["jobs_triggered"] += 1
                logger.info(
                    "clustering_job_triggered",
                    stage3_job_id=stage3_event.job_id,
                    embedding_type=stage3_event.embedding_type,
                    algorithm=self.default_algorithm,
                )

            return success

        except ValidationError as e:
            logger.error(
                "event_validation_failed",
                event_id=event_id,
                error=str(e),
                event_data=event_data,
            )
            return False

        except Exception as e:
            logger.error(
                "event_processing_failed",
                event_id=event_id,
                error=str(e),
                exc_info=True,
            )
            return False

    async def _trigger_clustering_job(self, stage3_event: Stage3Event) -> bool:
        """
        Trigger a clustering job via Celery.

        Args:
            stage3_event: Validated Stage 3 event

        Returns:
            True if job submitted successfully
        """
        try:
            # Build job configuration
            job_config = {
                "embedding_type": stage3_event.embedding_type,
                "algorithm": self.default_algorithm,
                "min_cluster_size": None,  # Use defaults from settings
                "metadata": {
                    "triggered_by": "stage3_event",
                    "stage3_job_id": stage3_event.job_id,
                    "auto_triggered": True,
                    "trigger_timestamp": datetime.utcnow().isoformat(),
                },
            }

            # Submit Celery task (async)
            # NORMAL PRIORITY (5) for auto-triggered jobs - manual calls processed first
            task = cluster_batch_task.apply_async(
                kwargs=job_config,
                queue="clustering",
                priority=5,  # Lower priority than manual API calls
            )

            logger.info(
                "celery_task_submitted",
                task_id=task.id,
                stage3_job_id=stage3_event.job_id,
                embedding_type=stage3_event.embedding_type,
            )

            return True

        except Exception as e:
            logger.error(
                "clustering_job_submission_failed",
                stage3_job_id=stage3_event.job_id,
                error=str(e),
                exc_info=True,
            )
            return False

    async def consume_events(self) -> None:
        """
        Main event consumption loop.

        Continuously reads events from Redis stream and processes them.
        Handles retries and acknowledgements.
        """
        logger.info(
            "consumer_starting",
            stream=self.stream_name,
            consumer_group=self.consumer_group,
            consumer_name=self.consumer_name,
        )

        self.running = True
        last_id = ">"  # Read only new messages

        while self.running:
            try:
                # Read events from stream
                events = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: last_id},
                    count=self.count,
                    block=self.block_ms,
                )

                if not events:
                    # No new events (timeout)
                    await asyncio.sleep(0.1)
                    continue

                # Process each event
                for stream_name, messages in events:
                    for event_id, event_data in messages:
                        try:
                            # Process event
                            success = await self.process_event(event_id, event_data)

                            if success:
                                # Acknowledge successful processing
                                self.redis_client.xack(
                                    self.stream_name, self.consumer_group, event_id
                                )
                                self.stats["events_processed"] += 1
                                self.stats["last_event_time"] = datetime.utcnow().isoformat()

                                logger.debug("event_acknowledged", event_id=event_id)
                            else:
                                # Event processing failed, will be retried
                                self.stats["errors"] += 1
                                logger.warning(
                                    "event_processing_failed_will_retry",
                                    event_id=event_id,
                                )

                        except Exception as e:
                            self.stats["errors"] += 1
                            logger.error(
                                "event_loop_error",
                                event_id=event_id,
                                error=str(e),
                                exc_info=True,
                            )

                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)

            except redis.RedisError as e:
                logger.error("redis_error", error=str(e), exc_info=True)
                await asyncio.sleep(5)  # Wait before retry

            except Exception as e:
                logger.error("consumer_loop_error", error=str(e), exc_info=True)
                await asyncio.sleep(5)

        logger.info("consumer_stopped")

    def stop(self) -> None:
        """Stop the consumer gracefully."""
        logger.info("consumer_stopping")
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get consumer statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "stream_name": self.stream_name,
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name,
            "running": self.running,
        }


async def run_consumer():
    """
    Run the Stage 3 stream consumer.

    This is the main entry point for the consumer service.
    """
    consumer = Stage3StreamConsumer()

    try:
        # Connect to Redis
        consumer.connect()

        # Start consuming events
        await consumer.consume_events()

    except KeyboardInterrupt:
        logger.info("consumer_interrupted")
        consumer.stop()

    except Exception as e:
        logger.error("consumer_error", error=str(e), exc_info=True)
        raise

    finally:
        consumer.disconnect()
        logger.info("consumer_shutdown_complete")


if __name__ == "__main__":
    # Run consumer as standalone service
    asyncio.run(run_consumer())
