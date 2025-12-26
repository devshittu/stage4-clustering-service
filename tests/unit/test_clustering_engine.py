"""
Unit tests for ClusteringEngine orchestration layer.

Tests the ClusteringEngine class including:
- Algorithm selection and instantiation
- Parameter validation
- Algorithm recommendation
- Optimal k estimation
- Integration with all clustering algorithms
"""

import pytest
import numpy as np
from src.core.clustering_engine import ClusteringEngine


@pytest.mark.unit
class TestClusteringEngine:
    """Test suite for ClusteringEngine."""

    def test_init(self):
        """Test ClusteringEngine initialization."""
        engine = ClusteringEngine()
        assert engine is not None

    def test_algorithm_registry(self):
        """Test that all algorithms are registered."""
        engine = ClusteringEngine()

        expected_algorithms = ["hdbscan", "kmeans", "agglomerative"]
        for alg in expected_algorithms:
            assert alg in engine.ALGORITHMS

    def test_cluster_hdbscan(self, clustered_vectors):
        """Test clustering with HDBSCAN algorithm."""
        vectors, _ = clustered_vectors
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=vectors,
            algorithm="hdbscan",
            algorithm_params={"min_cluster_size": 5, "min_samples": 3}
        )

        assert result.n_clusters >= 0
        assert len(result.labels) == len(vectors)

    def test_cluster_kmeans(self, clustered_vectors):
        """Test clustering with K-Means algorithm."""
        vectors, _ = clustered_vectors
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 3}
        )

        assert result.n_clusters == 3
        assert len(result.labels) == len(vectors)

    def test_cluster_agglomerative(self, clustered_vectors):
        """Test clustering with Agglomerative algorithm."""
        vectors, _ = clustered_vectors
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=vectors,
            algorithm="agglomerative",
            algorithm_params={"n_clusters": 3}
        )

        assert result.n_clusters == 3
        assert len(result.labels) == len(vectors)

    def test_case_insensitive_algorithm_name(self, sample_vectors):
        """Test that algorithm names are case-insensitive."""
        engine = ClusteringEngine()

        for alg_name in ["HDBSCAN", "HDBScan", "hdbscan"]:
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm=alg_name,
                algorithm_params={"min_cluster_size": 5}
            )
            assert result is not None

    def test_unsupported_algorithm(self, sample_vectors):
        """Test error handling for unsupported algorithm."""
        engine = ClusteringEngine()

        with pytest.raises(ValueError, match="Unsupported algorithm"):
            engine.cluster(
                vectors=sample_vectors,
                algorithm="invalid_algorithm",
                algorithm_params={}
            )

    def test_temporal_weighting(self, sample_vectors, sample_metadata_with_dates):
        """Test clustering with temporal weighting enabled."""
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5},
            metadata=sample_metadata_with_dates,
            enable_temporal_weighting=True,
            temporal_decay_factor=7.0
        )

        assert result.n_clusters == 5
        assert len(result.labels) == len(sample_vectors)

    def test_metadata_filtering(self, sample_vectors, sample_metadata):
        """Test clustering with metadata filtering."""
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="hdbscan",
            algorithm_params={"min_cluster_size": 3},
            metadata=sample_metadata,
            metadata_filters={"domain": "diplomatic"}
        )

        assert result.n_clusters >= 0

    def test_get_recommended_algorithm_small(self):
        """Test algorithm recommendation for small datasets."""
        engine = ClusteringEngine()

        # <100 vectors -> agglomerative
        rec = engine.get_recommended_algorithm(n_vectors=50, embedding_type="event")
        assert rec == "agglomerative"

    def test_get_recommended_algorithm_medium_event(self):
        """Test algorithm recommendation for medium event datasets."""
        engine = ClusteringEngine()

        # 100-10000 events -> hdbscan
        rec = engine.get_recommended_algorithm(n_vectors=1000, embedding_type="event")
        assert rec == "hdbscan"

    def test_get_recommended_algorithm_medium_document(self):
        """Test algorithm recommendation for medium document datasets."""
        engine = ClusteringEngine()

        # 100-10000 documents -> kmeans
        rec = engine.get_recommended_algorithm(n_vectors=1000, embedding_type="document")
        assert rec == "kmeans"

    def test_get_recommended_algorithm_large(self):
        """Test algorithm recommendation for large datasets."""
        engine = ClusteringEngine()

        # >10000 vectors -> kmeans
        rec = engine.get_recommended_algorithm(n_vectors=15000, embedding_type="event")
        assert rec == "kmeans"

    def test_estimate_optimal_k_small(self, sample_vectors):
        """Test optimal k estimation for small dataset."""
        engine = ClusteringEngine()

        optimal_k = engine.estimate_optimal_k(sample_vectors, min_k=2, max_k=20)

        # Should return reasonable value
        assert 2 <= optimal_k <= 20

    def test_estimate_optimal_k_large(self):
        """Test optimal k estimation for large dataset (uses sampling)."""
        engine = ClusteringEngine()

        # Create large dataset
        np.random.seed(42)
        vectors = np.random.randn(15000, 768).astype(np.float32)

        optimal_k = engine.estimate_optimal_k(vectors, min_k=10, max_k=100)

        # Should sample and return reasonable value
        assert 10 <= optimal_k <= 100

    def test_validate_clustering_config_valid(self):
        """Test config validation with valid parameters."""
        engine = ClusteringEngine()

        # Valid HDBSCAN config
        errors = engine.validate_clustering_config(
            algorithm="hdbscan",
            params={"min_cluster_size": 5, "min_samples": 3}
        )
        assert len(errors) == 0

        # Valid K-Means config
        errors = engine.validate_clustering_config(
            algorithm="kmeans",
            params={"n_clusters": 10}
        )
        assert len(errors) == 0

        # Valid Agglomerative config
        errors = engine.validate_clustering_config(
            algorithm="agglomerative",
            params={"n_clusters": 5}
        )
        assert len(errors) == 0

    def test_validate_clustering_config_invalid_algorithm(self):
        """Test config validation with invalid algorithm."""
        engine = ClusteringEngine()

        errors = engine.validate_clustering_config(
            algorithm="invalid",
            params={}
        )

        assert "algorithm" in errors

    def test_validate_clustering_config_invalid_hdbscan_params(self):
        """Test config validation with invalid HDBSCAN parameters."""
        engine = ClusteringEngine()

        # min_cluster_size too small
        errors = engine.validate_clustering_config(
            algorithm="hdbscan",
            params={"min_cluster_size": 1, "min_samples": 0}
        )

        assert "min_cluster_size" in errors or "min_samples" in errors

    def test_validate_clustering_config_invalid_kmeans_params(self):
        """Test config validation with invalid K-Means parameters."""
        engine = ClusteringEngine()

        # n_clusters too small
        errors = engine.validate_clustering_config(
            algorithm="kmeans",
            params={"n_clusters": 1}
        )

        assert "n_clusters" in errors

    def test_validate_clustering_config_invalid_agglomerative_params(self):
        """Test config validation with invalid Agglomerative parameters."""
        engine = ClusteringEngine()

        # Neither n_clusters nor distance_threshold specified
        errors = engine.validate_clustering_config(
            algorithm="agglomerative",
            params={}
        )

        assert "config" in errors or "n_clusters" in errors

    def test_all_algorithms_produce_results(self, sample_vectors):
        """Test that all registered algorithms produce valid results."""
        engine = ClusteringEngine()

        for algorithm_name in engine.ALGORITHMS.keys():
            # Use appropriate parameters for each algorithm
            if algorithm_name == "hdbscan":
                params = {"min_cluster_size": 5}
            elif algorithm_name == "kmeans":
                params = {"n_clusters": 5}
            elif algorithm_name == "agglomerative":
                params = {"n_clusters": 5}
            else:
                params = {}

            result = engine.cluster(
                vectors=sample_vectors,
                algorithm=algorithm_name,
                algorithm_params=params
            )

            # All should produce valid results
            assert result is not None
            assert len(result.labels) == len(sample_vectors)
            assert result.n_clusters >= 0

    def test_cluster_with_metadata_none(self, sample_vectors):
        """Test clustering when metadata is None."""
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5},
            metadata=None
        )

        assert result.n_clusters == 5

    def test_cluster_with_empty_metadata_filters(self, sample_vectors):
        """Test clustering with empty metadata filters."""
        engine = ClusteringEngine()

        result = engine.cluster(
            vectors=sample_vectors,
            algorithm="kmeans",
            algorithm_params={"n_clusters": 5},
            metadata_filters={}
        )

        assert result.n_clusters == 5

    @pytest.mark.slow
    def test_performance_multiple_algorithms(self, sample_vectors):
        """Test performance of running multiple algorithms on same data."""
        import time

        engine = ClusteringEngine()

        timings = {}

        for algorithm in ["hdbscan", "kmeans", "agglomerative"]:
            if algorithm == "hdbscan":
                params = {"min_cluster_size": 5}
            else:
                params = {"n_clusters": 5}

            start = time.time()
            result = engine.cluster(
                vectors=sample_vectors,
                algorithm=algorithm,
                algorithm_params=params
            )
            elapsed = time.time() - start
            timings[algorithm] = elapsed

            assert result is not None

        # All should complete in reasonable time
        for alg, timing in timings.items():
            assert timing < 10.0, f"{alg} took too long: {timing}s"
