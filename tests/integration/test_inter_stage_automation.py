"""
Integration tests for inter-stage automation.

Tests the full automation flow:
- Stage 3 → Stage 4 (upstream)
- Stage 4 → Stage 5 (downstream)
- Priority queue handling
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
import redis
from fastapi.testclient import TestClient

from src.api.orchestrator import app


@pytest.fixture
def redis_client():
    """Create real Redis client for integration tests."""
    try:
        client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        client.ping()
        # Clean up test stream
        try:
            client.delete("stage3:embeddings:events:test")
            client.delete("stage4:clustering:events:test")
        except:
            pass
        yield client
        # Cleanup
        try:
            client.delete("stage3:embeddings:events:test")
            client.delete("stage4:clustering:events:test")
        except:
            pass
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestUpstreamAutomation:
    """Test Stage 3 → Stage 4 automation."""

    def test_webhook_triggers_clustering(self, client):
        """Test webhook from Stage 3 triggers clustering job."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "default_algorithm": "hdbscan",
                    "min_embeddings": 10,
                    "quality_threshold": 0.0,
                },
            }
            mock_cfg.get_section.return_value = upstream

            with patch("src.api.orchestrator.cluster_batch_task") as mock_task:
                mock_task.apply_async.return_value = Mock(id="triggered-task-123")

                # Simulate Stage 3 webhook call
                response = client.post(
                    "/webhooks/embeddings-completed",
                    params={
                        "event_type": "embedding.job.completed",
                        "job_id": "stage3_integration_test_001",
                        "embedding_type": "event",
                        "total_embeddings": 500,
                        "quality_score": 0.85,
                    },
                )

                assert response.status_code == 202
                data = response.json()
                assert data["status"] == "accepted"
                assert "stage4_task_id" in data

                # Verify job was triggered with correct parameters
                mock_task.apply_async.assert_called_once()
                call_kwargs = mock_task.apply_async.call_args.kwargs

                # Verify auto-triggered job has normal priority
                assert call_kwargs["priority"] == 5

                # Verify job configuration
                job_config = call_kwargs["kwargs"]
                assert job_config["embedding_type"] == "event"
                assert job_config["algorithm"] == "hdbscan"
                assert job_config["metadata"]["triggered_by"] == "stage3_webhook"
                assert job_config["metadata"]["stage3_job_id"] == "stage3_integration_test_001"


class TestDownstreamAutomation:
    """Test Stage 4 → Stage 5 automation."""

    @pytest.mark.asyncio
    async def test_job_completion_publishes_to_redis_stream(self, redis_client):
        """Test job completion publishes event to Redis stream."""
        from src.utils.event_publisher import EventPublisher

        stream_name = "stage4:clustering:events:test"
        publisher = EventPublisher(
            redis_client=redis_client,
            stream_name=stream_name,
            webhook_urls=[],
        )

        # Publish job completion event
        publisher.publish_job_completed(
            job_id="integration_test_job_001",
            clusters_created=50,
            outliers=5,
            quality_metrics={"silhouette_score": 0.75},
            processing_time_ms=120000,
            output_files=["/app/data/clusters/test_output.jsonl"],
            embedding_type="event",
            algorithm="hdbscan",
            statistics={"total_items": 500},
        )

        # Verify event was published to stream
        events = redis_client.xread({stream_name: "0"}, count=10)

        assert len(events) > 0
        stream_events = events[0][1]
        assert len(stream_events) > 0

        # Verify event content
        event_id, event_data = stream_events[0]
        assert event_data["event_type"] == "job.completed"
        assert event_data["job_id"] == "integration_test_job_001"
        assert event_data["embedding_type"] == "event"
        assert event_data["algorithm"] == "hdbscan"
        assert "output_files" in event_data

    @pytest.mark.asyncio
    async def test_job_completion_calls_stage5_webhook(self):
        """Test job completion calls Stage 5 webhook."""
        from src.utils.stage5_webhook_publisher import publish_to_stage5
        from unittest.mock import AsyncMock

        with patch("src.utils.stage5_webhook_publisher.settings") as mock_settings:
            # Configure mock settings
            downstream = Mock()
            downstream.webhook_publisher = Mock()
            downstream.webhook_publisher.enabled = True
            downstream.webhook_publisher.stage5_urls = ["http://localhost:9999/webhooks/test"]
            downstream.webhook_publisher.retry = Mock()
            downstream.webhook_publisher.retry.max_attempts = 1
            downstream.webhook_publisher.retry.backoff_seconds = 0
            downstream.webhook_publisher.retry.timeout_seconds = 1
            downstream.webhook_publisher.fail_silently = True
            downstream.webhook_publisher.auth_token = None

            mock_settings.downstream_automation = downstream

            # Mock HTTP client to avoid actual network call
            with patch("src.utils.stage5_webhook_publisher.httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 202
                mock_client.post.return_value = mock_response
                MockClient.return_value.__aenter__.return_value = mock_client

                # Call webhook publisher
                result = await publish_to_stage5(
                    job_id="integration_test_job_002",
                    embedding_type="event",
                    algorithm="hdbscan",
                    clusters_created=50,
                    outliers=5,
                    output_files=["/app/data/clusters/output.jsonl"],
                    quality_metrics={"silhouette_score": 0.75},
                )

                assert result is True

                # Verify HTTP POST was called
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args

                # Verify URL
                assert "localhost:9999" in str(call_args.args[0])

                # Verify payload
                payload = call_args.kwargs["json"]
                assert payload["event_type"] == "clustering.job.completed"
                assert payload["job_id"] == "integration_test_job_002"
                assert payload["embedding_type"] == "event"
                assert payload["clusters_created"] == 50
                assert "/app/data/clusters/output.jsonl" in payload["output_files"]


class TestPriorityQueueHandling:
    """Test priority queue handling for manual vs auto-triggered jobs."""

    def test_manual_job_higher_priority_than_auto(self, client):
        """Test manual API jobs have higher priority than auto-triggered jobs."""
        with patch("src.api.orchestrator.job_manager") as mock_jm:
            mock_jm.create_job.return_value = {"job_id": "manual_job"}

            with patch("src.api.orchestrator.celery_app") as mock_celery:
                mock_celery.send_task.return_value = None

                # Submit manual job
                response_manual = client.post(
                    "/api/v1/batch",
                    json={"embedding_type": "event", "algorithm": "hdbscan"},
                )

                assert response_manual.status_code == 202

                # Verify manual job uses priority=9
                manual_call_kwargs = mock_celery.send_task.call_args.kwargs
                assert manual_call_kwargs["priority"] == 9

        # Now test auto-triggered job
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "default_algorithm": "hdbscan",
                    "min_embeddings": 10,
                },
            }
            mock_cfg.get_section.return_value = upstream

            with patch("src.api.orchestrator.cluster_batch_task") as mock_task:
                mock_task.apply_async.return_value = Mock(id="auto-task")

                # Submit auto-triggered job via webhook
                response_auto = client.post(
                    "/webhooks/embeddings-completed",
                    params={
                        "event_type": "embedding.job.completed",
                        "job_id": "stage3_job",
                        "embedding_type": "event",
                        "total_embeddings": 100,
                    },
                )

                assert response_auto.status_code == 202

                # Verify auto job uses priority=5
                auto_call_kwargs = mock_task.apply_async.call_args.kwargs
                assert auto_call_kwargs["priority"] == 5

                # Assert manual priority (9) > auto priority (5)
                assert manual_call_kwargs["priority"] > auto_call_kwargs["priority"]


class TestEndToEndAutomation:
    """Test end-to-end automation flow."""

    def test_stage3_to_stage4_to_stage5_flow(self, client):
        """
        Test full automation flow:
        1. Stage 3 webhook → Stage 4 clusters
        2. Stage 4 completes → Stage 5 webhook
        """
        # Step 1: Stage 3 webhook triggers Stage 4
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "default_algorithm": "hdbscan",
                    "min_embeddings": 10,
                },
            }
            mock_cfg.get_section.return_value = upstream

            with patch("src.api.orchestrator.cluster_batch_task") as mock_clustering_task:
                # Mock clustering task completion
                mock_task_result = Mock()
                mock_task_result.id = "clustering-task-e2e-001"
                mock_clustering_task.apply_async.return_value = mock_task_result

                # Stage 3 calls webhook
                response_webhook = client.post(
                    "/webhooks/embeddings-completed",
                    params={
                        "event_type": "embedding.job.completed",
                        "job_id": "stage3_e2e_job_001",
                        "embedding_type": "event",
                        "total_embeddings": 500,
                    },
                )

                assert response_webhook.status_code == 202
                assert response_webhook.json()["status"] == "accepted"

                # Verify clustering was triggered
                mock_clustering_task.apply_async.assert_called_once()

        # Step 2: Simulate clustering completion calling Stage 5 webhook
        with patch("src.utils.stage5_webhook_publisher.settings") as mock_settings:
            from src.utils.stage5_webhook_publisher import publish_to_stage5
            from unittest.mock import AsyncMock

            downstream = Mock()
            downstream.webhook_publisher = Mock()
            downstream.webhook_publisher.enabled = True
            downstream.webhook_publisher.stage5_urls = ["http://graph-orchestrator:8000/webhooks/clustering-completed"]
            downstream.webhook_publisher.retry = Mock()
            downstream.webhook_publisher.retry.max_attempts = 1
            downstream.webhook_publisher.retry.backoff_seconds = 0
            downstream.webhook_publisher.retry.timeout_seconds = 1
            downstream.webhook_publisher.fail_silently = True
            downstream.webhook_publisher.auth_token = None

            mock_settings.downstream_automation = downstream

            with patch("src.utils.stage5_webhook_publisher.httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 202
                mock_client.post.return_value = mock_response
                MockClient.return_value.__aenter__.return_value = mock_client

                # Clustering completes and calls Stage 5
                import asyncio

                async def run_webhook():
                    return await publish_to_stage5(
                        job_id="e2e_clustering_job_001",
                        embedding_type="event",
                        algorithm="hdbscan",
                        clusters_created=75,
                        outliers=10,
                        output_files=["/app/data/clusters/e2e_output.jsonl"],
                    )

                result = asyncio.run(run_webhook())
                assert result is True

                # Verify Stage 5 was called
                mock_client.post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
