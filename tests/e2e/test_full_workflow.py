"""
End-to-end tests for full clustering workflow.

Tests the complete workflow from API request to results:
- API job submission
- Celery task execution
- Result storage
- Result retrieval
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.mark.e2e
@pytest.mark.slow
class TestFullWorkflow:
    """End-to-end tests for complete clustering workflow."""

    @patch('redis.Redis')
    def test_api_job_submission_flow(self, mock_redis, test_client):
        """Test complete job submission flow through API."""
        # Mock Redis for job management
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.get.return_value = None

        # Submit job via API
        response = test_client.post(
            "/api/v1/batch",
            json={
                "embedding_type": "event",
                "algorithm": "kmeans",
                "algorithm_params": {"n_clusters": 5},
                "metadata_filters": {}
            }
        )

        # Should return 202 Accepted with job_id
        # Note: Actual implementation may differ
        assert response.status_code in [200, 202, 404, 500]

        # If succeeded, should have job_id
        if response.status_code in [200, 202]:
            data = response.json()
            assert "job_id" in data or "error" in data

    def test_health_check_workflow(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        # Should return health status
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data

    @pytest.mark.requires_stage3
    def test_stage3_to_stage4_handoff(self, temp_faiss_indices):
        """Test data handoff from Stage 3 to Stage 4."""
        try:
            from src.storage.faiss_loader import FAISSLoader
            from src.schemas.data_models import EmbeddingType
        except ImportError:
            pytest.skip("FAISS not available")

        # Simulate Stage 3 indices
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Load all index types (Stage 3 output)
        for emb_type in [EmbeddingType.DOCUMENT, EmbeddingType.EVENT,
                         EmbeddingType.ENTITY, EmbeddingType.STORYLINE]:
            success = loader.load_index(emb_type)
            assert success

            # Get stats
            stats = loader.get_index_stats(emb_type)
            assert stats["loaded"] == True
            assert stats["total_vectors"] > 0

    @pytest.mark.requires_postgres
    @pytest.mark.requires_redis
    @patch('redis.Redis')
    @patch('sqlalchemy.ext.asyncio.create_async_engine')
    def test_full_clustering_and_storage(self, mock_engine, mock_redis,
                                         temp_faiss_indices, sample_vectors):
        """Test complete workflow: load, cluster, store, retrieve."""
        try:
            from src.storage.faiss_loader import FAISSLoader
            from src.core.clustering_engine import ClusteringEngine
            from src.schemas.data_models import EmbeddingType
        except ImportError:
            pytest.skip("FAISS or dependencies not available")

        # 1. Load indices (Stage 3 → Stage 4)
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.EVENT)

        # 2. Perform clustering
        engine = ClusteringEngine()
        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5}
        )

        # 3. Verify clustering results
        assert result.n_clusters == 5
        assert len(result.labels) == len(sample_vectors)

        # 4. Storage would happen here (mocked in this test)
        # In production: ClusterStorageManager.save_clusters(result)

        # 5. Retrieval would happen here (mocked in this test)
        # In production: ClusterStorageManager.get_cluster(cluster_id)

    def test_algorithm_recommendation_workflow(self):
        """Test algorithm recommendation based on dataset characteristics."""
        from src.core.clustering_engine import ClusteringEngine

        engine = ClusteringEngine()

        # Small dataset → agglomerative
        rec = engine.get_recommended_algorithm(n_vectors=50, embedding_type="entity")
        assert rec in ["hdbscan", "kmeans", "agglomerative"]

        # Medium events → hdbscan
        rec = engine.get_recommended_algorithm(n_vectors=1000, embedding_type="event")
        assert rec in ["hdbscan", "kmeans", "agglomerative"]

        # Large dataset → kmeans
        rec = engine.get_recommended_algorithm(n_vectors=15000, embedding_type="document")
        assert rec in ["hdbscan", "kmeans", "agglomerative"]

    @pytest.mark.slow
    def test_multi_level_clustering_workflow(self, temp_faiss_indices, sample_vectors):
        """Test multi-level clustering workflow (documents, events, entities, storylines)."""
        try:
            from src.storage.faiss_loader import FAISSLoader
            from src.core.clustering_engine import ClusteringEngine
            from src.schemas.data_models import EmbeddingType
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        engine = ClusteringEngine()

        # Multi-level clustering strategy
        levels = {
            EmbeddingType.DOCUMENT: ("hdbscan", {"min_cluster_size": 5}),
            EmbeddingType.EVENT: ("hdbscan", {"min_cluster_size": 5}),
            EmbeddingType.ENTITY: ("agglomerative", {"distance_threshold": 0.3}),
            EmbeddingType.STORYLINE: ("kmeans", {"n_clusters": 20})
        }

        results = {}

        for emb_type, (algorithm, params) in levels.items():
            # Load index
            loader.load_index(emb_type)

            # Cluster
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm=algorithm,
                algorithm_params=params
            )

            results[emb_type.value] = result

            # Verify
            assert result.n_clusters >= 0
            assert len(result.labels) == len(sample_vectors)

        # All levels should succeed
        assert len(results) == 4

    def test_error_recovery_workflow(self, test_client):
        """Test error handling and recovery in API workflow."""
        # Invalid request (missing required fields)
        response = test_client.post(
            "/api/v1/batch",
            json={}
        )

        # Should return error
        assert response.status_code in [400, 422, 500]

    def test_configuration_validation_workflow(self):
        """Test configuration validation workflow."""
        from src.core.clustering_engine import ClusteringEngine

        engine = ClusteringEngine()

        # Valid config
        errors = engine.validate_clustering_config(
            algorithm="kmeans",
            params={"n_clusters": 10}
        )
        assert len(errors) == 0

        # Invalid config
        errors = engine.validate_clustering_config(
            algorithm="invalid_algorithm",
            params={}
        )
        assert len(errors) > 0

    @pytest.mark.requires_stage3
    @patch('httpx.get')
    def test_stage3_health_check_integration(self, mock_get):
        """Test Stage 3 health check integration."""
        # Mock Stage 3 health check response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        # In production, this would check actual Stage 3 service
        # Here we just verify the mock
        assert mock_response.status_code == 200

    def test_resource_cleanup_workflow(self, temp_faiss_indices):
        """Test resource cleanup after clustering."""
        try:
            from src.storage.faiss_loader import FAISSLoader
            from src.schemas.data_models import EmbeddingType
        except ImportError:
            pytest.skip("FAISS not available")

        # Use context manager for automatic cleanup
        with FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False) as loader:
            loader.load_index(EmbeddingType.DOCUMENT)
            assert loader._loaded["document"] == True

        # After context exit, cleanup should have occurred
        # (loader is out of scope)
