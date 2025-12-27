"""
Unit tests for automation webhook endpoints and publishers.

Tests:
- Stage 3 webhook receiver (upstream)
- Stage 5 webhook publisher (downstream)
- Priority queue handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from src.api.orchestrator import app
from src.utils.stage5_webhook_publisher import Stage5WebhookPublisher


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("src.api.orchestrator.config") as mock_cfg:
        # Upstream automation config
        upstream = {
            "enabled": True,
            "webhook_receiver": {
                "enabled": True,
                "auth_token": None,  # No auth for testing
            },
            "auto_trigger": {
                "embedding_types": ["document", "event", "entity", "storyline"],
                "default_algorithm": "hdbscan",
                "min_embeddings": 10,
                "quality_threshold": 0.0,
            },
        }
        mock_cfg.get_section.side_effect = lambda key: {
            "upstream_automation": upstream,
        }.get(key, {})

        yield mock_cfg


class TestStage3WebhookReceiver:
    """Test POST /webhooks/embeddings-completed endpoint."""

    def test_webhook_receiver_success(self, client, mock_config):
        """Test successful webhook reception."""
        with patch("src.api.orchestrator.cluster_batch_task") as mock_task:
            mock_task.apply_async.return_value = Mock(id="celery-task-123")

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "accepted"
            assert data["stage4_task_id"] == "celery-task-123"
            assert data["stage3_job_id"] == "stage3_job_123"
            assert data["embedding_type"] == "event"

            # Verify Celery task called with NORMAL priority (5)
            mock_task.apply_async.assert_called_once()
            call_kwargs = mock_task.apply_async.call_args.kwargs
            assert call_kwargs["priority"] == 5

    def test_webhook_receiver_auto_trigger_disabled(self, client):
        """Test webhook rejected when auto-trigger disabled."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            mock_cfg.get_section.return_value = {"enabled": False}

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                },
            )

            assert response.status_code == 503
            assert "Auto-trigger disabled" in response.json()["detail"]

    def test_webhook_receiver_webhook_disabled(self, client, mock_config):
        """Test webhook rejected when receiver disabled."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": False},
                "auto_trigger": {"embedding_types": ["event"]},
            }
            mock_cfg.get_section.return_value = upstream

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                },
            )

            assert response.status_code == 503

    def test_webhook_receiver_invalid_embedding_type(self, client, mock_config):
        """Test webhook rejected for invalid embedding type."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],  # Only event allowed
                    "min_embeddings": 10,
                },
            }
            mock_cfg.get_section.return_value = upstream

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "invalid_type",
                    "total_embeddings": 100,
                },
            )

            assert response.status_code == 400
            assert "not allowed" in response.json()["detail"]

    def test_webhook_receiver_insufficient_embeddings(self, client, mock_config):
        """Test webhook rejected for insufficient embeddings."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "min_embeddings": 100,  # Require 100
                },
            }
            mock_cfg.get_section.return_value = upstream

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 50,  # Only 50
                },
            )

            assert response.status_code == 400
            assert "Insufficient embeddings" in response.json()["detail"]

    def test_webhook_receiver_low_quality(self, client, mock_config):
        """Test webhook rejected for low quality embeddings."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {"enabled": True, "auth_token": None},
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "min_embeddings": 10,
                    "quality_threshold": 0.7,  # Require 0.7
                },
            }
            mock_cfg.get_section.return_value = upstream

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                    "quality_score": 0.5,  # Only 0.5
                },
            )

            assert response.status_code == 400
            assert "quality too low" in response.json()["detail"]

    def test_webhook_receiver_with_auth_token(self, client):
        """Test webhook with authentication token."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {
                    "enabled": True,
                    "auth_token": "secret_token_123",
                },
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "min_embeddings": 10,
                },
            }
            mock_cfg.get_section.return_value = upstream

            with patch("src.api.orchestrator.cluster_batch_task") as mock_task:
                mock_task.apply_async.return_value = Mock(id="task-123")

                # With correct token
                response = client.post(
                    "/webhooks/embeddings-completed",
                    params={
                        "event_type": "embedding.job.completed",
                        "job_id": "stage3_job_123",
                        "embedding_type": "event",
                        "total_embeddings": 100,
                        "auth_token": "secret_token_123",
                    },
                )

                assert response.status_code == 202

    def test_webhook_receiver_invalid_auth_token(self, client):
        """Test webhook rejected with invalid auth token."""
        with patch("src.api.orchestrator.config") as mock_cfg:
            upstream = {
                "enabled": True,
                "webhook_receiver": {
                    "enabled": True,
                    "auth_token": "secret_token_123",
                },
                "auto_trigger": {
                    "embedding_types": ["event"],
                    "min_embeddings": 10,
                },
            }
            mock_cfg.get_section.return_value = upstream

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                    "auth_token": "wrong_token",
                },
            )

            assert response.status_code == 401


class TestStage5WebhookPublisher:
    """Test Stage5WebhookPublisher class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.utils.stage5_webhook_publisher.settings") as mock_cfg:
            downstream = Mock()
            downstream.webhook_publisher = Mock()
            downstream.webhook_publisher.enabled = True
            downstream.webhook_publisher.stage5_urls = ["http://graph-orchestrator:8000/webhooks/clustering-completed"]
            downstream.webhook_publisher.retry = Mock()
            downstream.webhook_publisher.retry.max_attempts = 3
            downstream.webhook_publisher.retry.backoff_seconds = 1
            downstream.webhook_publisher.retry.timeout_seconds = 5
            downstream.webhook_publisher.fail_silently = True
            downstream.webhook_publisher.auth_token = None

            mock_cfg.downstream_automation = downstream

            yield mock_cfg

    @pytest.fixture
    def publisher(self, mock_settings):
        """Create publisher instance."""
        return Stage5WebhookPublisher()

    @pytest.mark.asyncio
    async def test_publish_success(self, publisher):
        """Test successful webhook publishing."""
        mock_response = Mock()
        mock_response.status_code = 202

        with patch.object(publisher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await publisher.publish_clustering_completed(
                job_id="job_123",
                embedding_type="event",
                algorithm="hdbscan",
                clusters_created=50,
                outliers=5,
                output_files=["/app/data/clusters/output.jsonl"],
            )

            assert result is True
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_disabled(self, publisher):
        """Test publishing when disabled."""
        publisher.enabled = False

        result = await publisher.publish_clustering_completed(
            job_id="job_123",
            embedding_type="event",
            algorithm="hdbscan",
            clusters_created=50,
            outliers=5,
            output_files=[],
        )

        assert result is True  # Returns true when disabled

    @pytest.mark.asyncio
    async def test_publish_no_urls(self, publisher):
        """Test publishing with no configured URLs."""
        publisher.webhook_urls = []

        result = await publisher.publish_clustering_completed(
            job_id="job_123",
            embedding_type="event",
            algorithm="hdbscan",
            clusters_created=50,
            outliers=5,
            output_files=[],
        )

        assert result is True  # Returns true when no URLs

    @pytest.mark.asyncio
    async def test_publish_retry_on_server_error(self, publisher):
        """Test retry logic on server error."""
        mock_response = Mock()
        mock_response.status_code = 503

        with patch.object(publisher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            publisher.max_attempts = 2  # Limit retries for test
            publisher.fail_silently = True

            result = await publisher.publish_clustering_completed(
                job_id="job_123",
                embedding_type="event",
                algorithm="hdbscan",
                clusters_created=50,
                outliers=5,
                output_files=[],
            )

            # Should fail silently
            assert result is True
            # Should retry
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_no_retry_on_client_error(self, publisher):
        """Test no retry on client error."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        with patch.object(publisher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            publisher.fail_silently = True

            result = await publisher.publish_clustering_completed(
                job_id="job_123",
                embedding_type="event",
                algorithm="hdbscan",
                clusters_created=50,
                outliers=5,
                output_files=[],
            )

            # Should fail silently
            assert result is True
            # Should NOT retry on 4xx
            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_publish_timeout(self, publisher):
        """Test handling of timeout."""
        with patch.object(publisher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client

            publisher.max_attempts = 2
            publisher.fail_silently = True

            result = await publisher.publish_clustering_completed(
                job_id="job_123",
                embedding_type="event",
                algorithm="hdbscan",
                clusters_created=50,
                outliers=5,
                output_files=[],
            )

            # Should retry and fail silently
            assert result is True
            assert mock_client.post.call_count == 2


class TestManualVsAutoJobPriority:
    """Test that manual API calls have higher priority than auto-triggered jobs."""

    def test_manual_api_uses_high_priority(self, client):
        """Test manual API call uses priority=9."""
        with patch("src.api.orchestrator.job_manager") as mock_jm:
            mock_jm.create_job.return_value = {"job_id": "job_123"}

            with patch("src.api.orchestrator.celery_app") as mock_celery:
                mock_celery.send_task.return_value = None

                response = client.post(
                    "/api/v1/batch",
                    json={
                        "embedding_type": "event",
                        "algorithm": "hdbscan",
                    },
                )

                assert response.status_code == 202

                # Verify priority=9 was used
                mock_celery.send_task.assert_called_once()
                call_kwargs = mock_celery.send_task.call_args.kwargs
                assert call_kwargs["priority"] == 9

    def test_webhook_auto_trigger_uses_normal_priority(self, client, mock_config):
        """Test webhook auto-trigger uses priority=5."""
        with patch("src.api.orchestrator.cluster_batch_task") as mock_task:
            mock_task.apply_async.return_value = Mock(id="task-123")

            response = client.post(
                "/webhooks/embeddings-completed",
                params={
                    "event_type": "embedding.job.completed",
                    "job_id": "stage3_job_123",
                    "embedding_type": "event",
                    "total_embeddings": 100,
                },
            )

            assert response.status_code == 202

            # Verify priority=5 was used
            call_kwargs = mock_task.apply_async.call_args.kwargs
            assert call_kwargs["priority"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
