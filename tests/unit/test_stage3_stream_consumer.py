"""
Unit tests for Stage 3 Redis Stream Consumer.

Tests the upstream automation (Stage 3 → Stage 4) consumer service.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.services.stage3_stream_consumer import (
    Stage3StreamConsumer,
    Stage3Event,
)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = Mock()
    client.ping = Mock(return_value=True)
    client.xgroup_create = Mock()
    client.xreadgroup = Mock(return_value=[])
    client.xack = Mock()
    return client


@pytest.fixture
def mock_settings():
    """Mock settings configuration."""
    with patch("src.services.stage3_stream_consumer.settings") as mock_config:
        # Upstream automation config
        upstream = Mock()
        upstream.enabled = True
        upstream.redis_consumer = Mock()
        upstream.redis_consumer.stream_name = "stage3:embeddings:events"
        upstream.redis_consumer.consumer_group = "stage4-test-consumers"
        upstream.redis_consumer.consumer_name = "test-worker-1"
        upstream.redis_consumer.block_ms = 1000
        upstream.redis_consumer.count = 10
        upstream.redis_consumer.trigger_events = ["embedding.job.completed"]
        upstream.redis_consumer.retry = Mock()
        upstream.redis_consumer.retry.max_attempts = 3
        upstream.redis_consumer.retry.backoff_seconds = 1

        # Auto-trigger config
        upstream.auto_trigger = Mock()
        upstream.auto_trigger.embedding_types = ["document", "event", "entity", "storyline"]
        upstream.auto_trigger.default_algorithm = "hdbscan"
        upstream.auto_trigger.min_embeddings = 10
        upstream.auto_trigger.quality_threshold = 0.0

        mock_config.upstream_automation = upstream

        # Redis broker config
        mock_config.redis_broker_host = "localhost"
        mock_config.redis_broker_port = 6379
        mock_config.redis_broker_db = 6

        yield mock_config


@pytest.fixture
def consumer(mock_settings):
    """Create Stage3StreamConsumer instance."""
    return Stage3StreamConsumer()


class TestStage3Event:
    """Test Stage3Event Pydantic model."""

    def test_valid_event(self):
        """Test creating valid event."""
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 5000,
            "timestamp": "2025-12-27T10:00:00Z",
        }
        event = Stage3Event(**event_data)

        assert event.event_type == "embedding.job.completed"
        assert event.job_id == "stage3_job_123"
        assert event.embedding_type == "event"
        assert event.total_embeddings == 5000

    def test_event_with_optional_fields(self):
        """Test event with optional fields."""
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 5000,
            "output_path": "/path/to/faiss/index.bin",
            "quality_score": 0.85,
            "timestamp": "2025-12-27T10:00:00Z",
            "metadata": {"domain": "diplomatic_relations"},
        }
        event = Stage3Event(**event_data)

        assert event.output_path == "/path/to/faiss/index.bin"
        assert event.quality_score == 0.85
        assert event.metadata["domain"] == "diplomatic_relations"

    def test_invalid_event_missing_field(self):
        """Test validation fails for missing required field."""
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            # Missing embedding_type
            "total_embeddings": 5000,
            "timestamp": "2025-12-27T10:00:00Z",
        }

        with pytest.raises(Exception):  # Pydantic ValidationError
            Stage3Event(**event_data)


class TestStage3StreamConsumer:
    """Test Stage3StreamConsumer class."""

    def test_initialization(self, consumer):
        """Test consumer initializes with correct configuration."""
        assert consumer.stream_name == "stage3:embeddings:events"
        assert consumer.consumer_group == "stage4-test-consumers"
        assert consumer.consumer_name == "test-worker-1"
        assert consumer.auto_trigger_enabled is True
        assert "event" in consumer.allowed_embedding_types
        assert consumer.default_algorithm == "hdbscan"

    def test_connect_creates_consumer_group(self, consumer, mock_redis_client):
        """Test connect creates Redis consumer group."""
        with patch("src.services.stage3_stream_consumer.redis.Redis", return_value=mock_redis_client):
            consumer.connect()

            # Should call xgroup_create
            mock_redis_client.xgroup_create.assert_called_once_with(
                name="stage3:embeddings:events",
                groupname="stage4-test-consumers",
                id="0",
                mkstream=True,
            )

    def test_connect_handles_existing_group(self, consumer, mock_redis_client):
        """Test connect handles existing consumer group gracefully."""
        import redis as redis_lib

        mock_redis_client.xgroup_create.side_effect = redis_lib.ResponseError("BUSYGROUP Consumer Group name already exists")

        with patch("src.services.stage3_stream_consumer.redis.Redis", return_value=mock_redis_client):
            # Should not raise exception
            consumer.connect()

    @pytest.mark.asyncio
    async def test_process_event_valid_trigger(self, consumer):
        """Test processing valid trigger event."""
        event_id = "1234567890-0"
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 100,
            "timestamp": "2025-12-27T10:00:00Z",
        }

        with patch.object(consumer, "_trigger_clustering_job", return_value=True) as mock_trigger:
            result = await consumer.process_event(event_id, event_data)

            assert result is True
            mock_trigger.assert_called_once()

            # Verify passed event
            call_args = mock_trigger.call_args
            stage3_event = call_args[0][0]
            assert stage3_event.job_id == "stage3_job_123"
            assert stage3_event.embedding_type == "event"

    @pytest.mark.asyncio
    async def test_process_event_non_trigger_type(self, consumer):
        """Test event with non-triggering event type is ignored."""
        event_id = "1234567890-0"
        event_data = {
            "event_type": "embedding.job.started",  # Not in trigger_events
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 100,
            "timestamp": "2025-12-27T10:00:00Z",
        }

        with patch.object(consumer, "_trigger_clustering_job") as mock_trigger:
            result = await consumer.process_event(event_id, event_data)

            # Should return True (successfully ignored)
            assert result is True
            # Should not trigger job
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_insufficient_embeddings(self, consumer):
        """Test event with insufficient embeddings is ignored."""
        event_id = "1234567890-0"
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 5,  # Below min_embeddings (10)
            "timestamp": "2025-12-27T10:00:00Z",
        }

        with patch.object(consumer, "_trigger_clustering_job") as mock_trigger:
            result = await consumer.process_event(event_id, event_data)

            assert result is True  # Successfully ignored
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_low_quality(self, consumer):
        """Test event with low quality score is ignored."""
        consumer.quality_threshold = 0.5  # Set threshold

        event_id = "1234567890-0"
        event_data = {
            "event_type": "embedding.job.completed",
            "job_id": "stage3_job_123",
            "embedding_type": "event",
            "total_embeddings": 100,
            "quality_score": 0.3,  # Below threshold
            "timestamp": "2025-12-27T10:00:00Z",
        }

        with patch.object(consumer, "_trigger_clustering_job") as mock_trigger:
            result = await consumer.process_event(event_id, event_data)

            assert result is True
            mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_event_invalid_data(self, consumer):
        """Test processing invalid event data returns False."""
        event_id = "1234567890-0"
        event_data = {
            "invalid": "data",
            # Missing required fields
        }

        result = await consumer.process_event(event_id, event_data)

        # Should return False for validation error
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_clustering_job_success(self, consumer):
        """Test successful job triggering."""
        stage3_event = Stage3Event(
            event_type="embedding.job.completed",
            job_id="stage3_job_123",
            embedding_type="event",
            total_embeddings=100,
            timestamp="2025-12-27T10:00:00Z",
        )

        mock_task = Mock()
        mock_task.id = "celery-task-uuid"

        with patch("src.services.stage3_stream_consumer.cluster_batch_task") as mock_celery_task:
            mock_celery_task.apply_async.return_value = mock_task

            result = await consumer._trigger_clustering_job(stage3_event)

            assert result is True

            # Verify Celery task called with correct priority
            mock_celery_task.apply_async.assert_called_once()
            call_kwargs = mock_celery_task.apply_async.call_args.kwargs
            assert call_kwargs["priority"] == 5  # Normal priority for auto-triggered

            # Verify job config
            job_config = call_kwargs["kwargs"]
            assert job_config["embedding_type"] == "event"
            assert job_config["algorithm"] == "hdbscan"
            assert job_config["metadata"]["triggered_by"] == "stage3_event"
            assert job_config["metadata"]["stage3_job_id"] == "stage3_job_123"

    @pytest.mark.asyncio
    async def test_trigger_clustering_job_failure(self, consumer):
        """Test job triggering handles failures."""
        stage3_event = Stage3Event(
            event_type="embedding.job.completed",
            job_id="stage3_job_123",
            embedding_type="event",
            total_embeddings=100,
            timestamp="2025-12-27T10:00:00Z",
        )

        with patch("src.services.stage3_stream_consumer.cluster_batch_task") as mock_celery_task:
            mock_celery_task.apply_async.side_effect = Exception("Celery error")

            result = await consumer._trigger_clustering_job(stage3_event)

            assert result is False

    def test_get_stats(self, consumer):
        """Test getting consumer statistics."""
        consumer.stats["events_processed"] = 100
        consumer.stats["jobs_triggered"] = 50
        consumer.stats["errors"] = 5
        consumer.running = True

        stats = consumer.get_stats()

        assert stats["events_processed"] == 100
        assert stats["jobs_triggered"] == 50
        assert stats["errors"] == 5
        assert stats["running"] is True
        assert stats["stream_name"] == "stage3:embeddings:events"

    def test_stop(self, consumer):
        """Test stopping consumer."""
        consumer.running = True
        consumer.stop()

        assert consumer.running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
