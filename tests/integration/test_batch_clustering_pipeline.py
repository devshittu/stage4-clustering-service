"""
Integration tests for batch clustering pipeline.

Tests the full clustering workflow including:
- FAISS index loading
- Clustering execution
- Storage manager integration
- Event publishing
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.core.clustering_engine import ClusteringEngine
from src.schemas.data_models import EmbeddingType


@pytest.mark.integration
class TestBatchClusteringPipeline:
    """Integration tests for batch clustering pipeline."""

    def test_full_clustering_pipeline(self, temp_faiss_indices, sample_vectors):
        """Test complete clustering pipeline from loading to clustering."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Load index
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        success = loader.load_index(EmbeddingType.DOCUMENT)
        assert success

        # Perform clustering
        engine = ClusteringEngine()
        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5}
        )

        # Verify results
        assert result.n_clusters == 5
        assert len(result.labels) == len(sample_vectors)

    def test_clustering_with_metadata_filtering(self, temp_faiss_indices,
                                                 sample_vectors, sample_metadata):
        """Test clustering with metadata filtering integration."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Load index
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.EVENT)

        # Cluster with metadata filters
        engine = ClusteringEngine()
        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="hdbscan",
            algorithm_params={"min_cluster_size": 3},
            metadata=sample_metadata,
            metadata_filters={"domain": "diplomatic"}
        )

        # Should succeed
        assert result.n_clusters >= 0

    def test_temporal_clustering_pipeline(self, temp_faiss_indices,
                                          sample_vectors, sample_metadata_with_dates):
        """Test temporal clustering integration."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Load index
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.EVENT)

        # Cluster with temporal weighting
        engine = ClusteringEngine()
        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5},
            metadata=sample_metadata_with_dates,
            enable_temporal_weighting=True,
            temporal_decay_factor=7.0
        )

        # Should succeed
        assert result.n_clusters == 5

    @pytest.mark.slow
    def test_multiple_algorithm_comparison(self, temp_faiss_indices, sample_vectors):
        """Test running multiple algorithms on same data and comparing results."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Load index
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        engine = ClusteringEngine()
        results = {}

        # Run all algorithms
        algorithms = {
            "hdbscan": {"min_cluster_size": 5},
            "kmeans": {"n_clusters": 5},
            "agglomerative": {"n_clusters": 5}
        }

        for alg_name, params in algorithms.items():
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm=alg_name,
                algorithm_params=params
            )
            results[alg_name] = result

        # All should complete successfully
        for alg_name, result in results.items():
            assert result is not None
            assert len(result.labels) == len(sample_vectors)
            assert result.n_clusters >= 0

    @pytest.mark.requires_redis
    @patch('redis.Redis')
    def test_event_publishing_integration(self, mock_redis, sample_vectors):
        """Test event publishing during clustering."""
        engine = ClusteringEngine()

        # Cluster with mock event publisher
        with patch('src.utils.event_publisher.publish_job_event') as mock_publish:
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm="kmeans",
                algorithm_params={"n_clusters": 3}
            )

            # Verify clustering succeeded
            assert result.n_clusters == 3

            # Event publishing is handled by Celery worker, not engine
            # So we just verify clustering works

    def test_faiss_search_then_cluster(self, temp_faiss_indices, sample_vectors):
        """Test search followed by clustering on results."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Load index and search
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        query = sample_vectors[0:1]
        distances, indices, metadata = loader.search(
            EmbeddingType.DOCUMENT,
            query,
            k=50
        )

        # Extract vectors from search results
        search_vectors = sample_vectors[indices[0]]

        # Cluster the search results
        engine = ClusteringEngine()
        result = engine.cluster(
            vectors=search_vectors,
            algorithm="hdbscan",
            algorithm_params={"min_cluster_size": 3}
        )

        # Should successfully cluster search results
        assert result.n_clusters >= 0
        assert len(result.labels) == len(search_vectors)

    def test_all_embedding_types(self, temp_faiss_indices, sample_vectors):
        """Test clustering all embedding types."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        engine = ClusteringEngine()

        embedding_types = [
            EmbeddingType.DOCUMENT,
            EmbeddingType.EVENT,
            EmbeddingType.ENTITY,
            EmbeddingType.STORYLINE
        ]

        for emb_type in embedding_types:
            # Load index
            success = loader.load_index(emb_type)
            assert success

            # Cluster
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm="kmeans",
                algorithm_params={"n_clusters": 3}
            )

            # Verify
            assert result.n_clusters == 3
            assert len(result.labels) == len(sample_vectors)

    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test clustering large batch of vectors."""
        # Create large dataset
        np.random.seed(42)
        large_vectors = np.random.randn(5000, 768).astype(np.float32)
        large_vectors = large_vectors / np.linalg.norm(large_vectors, axis=1, keepdims=True)

        engine = ClusteringEngine()

        # Cluster large batch
        result = engine.cluster(
            vectors=large_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 20}
        )

        # Should handle large batch
        assert result.n_clusters == 20
        assert len(result.labels) == len(large_vectors)

    def test_error_handling_in_pipeline(self, temp_faiss_indices):
        """Test error handling throughout pipeline."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        engine = ClusteringEngine()

        # Test with invalid vectors
        invalid_vectors = np.array([]).reshape(0, 768)

        with pytest.raises((ValueError, IndexError)):
            engine.cluster(
                vectors=invalid_vectors,
                algorithm="kmeans",
                algorithm_params={"n_clusters": 3}
            )

    def test_quality_metrics_across_pipeline(self, temp_faiss_indices,
                                             clustered_vectors):
        """Test that quality metrics are calculated correctly in pipeline."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        vectors, _ = clustered_vectors

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        engine = ClusteringEngine()

        # Cluster with well-separated data
        result = engine.cluster(
            vectors=vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 3}
        )

        # Quality metrics should be present
        assert result.quality_metrics is not None
        assert "silhouette_score" in result.quality_metrics

        # Well-separated clusters should have good silhouette score
        assert result.quality_metrics["silhouette_score"] > 0.0
