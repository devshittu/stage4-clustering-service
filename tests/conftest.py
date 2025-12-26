"""
Pytest configuration and shared fixtures for Stage 4 Clustering Service tests.

This module provides:
- Shared test fixtures
- Mock data generators
- Database and Redis fixtures
- FAISS index fixtures
- Clustering algorithm fixtures
"""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import redis

# Set test environment variables
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


# =============================================================================
# Test Data Generators
# =============================================================================

@pytest.fixture
def sample_vectors():
    """Generate sample embedding vectors for testing."""
    np.random.seed(42)
    # Generate 100 768-dimensional vectors (same as all-mpnet-base-v2)
    vectors = np.random.randn(100, 768).astype(np.float32)
    # L2 normalize (cosine similarity ready)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


@pytest.fixture
def small_vectors():
    """Generate small set of vectors for quick tests."""
    np.random.seed(42)
    vectors = np.random.randn(10, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


@pytest.fixture
def clustered_vectors():
    """
    Generate vectors with clear cluster structure.

    Creates 3 distinct clusters:
    - Cluster 0: centered at [1, 0, 0, ...]
    - Cluster 1: centered at [0, 1, 0, ...]
    - Cluster 2: centered at [0, 0, 1, ...]
    """
    np.random.seed(42)
    n_per_cluster = 30
    dim = 768

    vectors = []
    labels = []

    # Cluster 0
    center0 = np.zeros(dim)
    center0[0] = 1.0
    cluster0 = center0 + np.random.randn(n_per_cluster, dim) * 0.1
    vectors.append(cluster0)
    labels.extend([0] * n_per_cluster)

    # Cluster 1
    center1 = np.zeros(dim)
    center1[1] = 1.0
    cluster1 = center1 + np.random.randn(n_per_cluster, dim) * 0.1
    vectors.append(cluster1)
    labels.extend([1] * n_per_cluster)

    # Cluster 2
    center2 = np.zeros(dim)
    center2[2] = 1.0
    cluster2 = center2 + np.random.randn(n_per_cluster, dim) * 0.1
    vectors.append(cluster2)
    labels.extend([2] * n_per_cluster)

    vectors = np.vstack(vectors).astype(np.float32)
    # L2 normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors, np.array(labels)


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for vectors."""
    metadata = []
    for i in range(100):
        metadata.append({
            "source_id": f"doc_{i:03d}",
            "embedding_type": "event",
            "domain": "diplomatic" if i < 50 else "military",
            "event_type": "contact_meet" if i % 2 == 0 else "conflict_attack",
            "temporal_reference": f"2025-12-{(i % 30) + 1:02d}T00:00:00Z",
            "confidence": 0.8 + (i % 20) / 100.0
        })
    return metadata


@pytest.fixture
def sample_metadata_with_dates():
    """Generate metadata with temporal information for temporal clustering tests."""
    from datetime import datetime, timedelta

    metadata = []
    base_date = datetime(2025, 12, 1)

    for i in range(100):
        date = base_date + timedelta(days=i % 30)
        metadata.append({
            "source_id": f"doc_{i:03d}",
            "embedding_type": "event",
            "temporal_reference": date.isoformat(),
            "normalized_date": date.isoformat(),
        })
    return metadata


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def clustering_config():
    """Sample clustering configuration."""
    from src.core.base_clustering import ClusteringConfig

    return ClusteringConfig(
        algorithm_name="hdbscan",
        params={
            "min_cluster_size": 5,
            "min_samples": 3
        },
        enable_temporal_weighting=False,
        temporal_decay_factor=7.0,
        metadata_filters=None
    )


@pytest.fixture
def clustering_config_with_temporal():
    """Clustering config with temporal weighting enabled."""
    from src.core.base_clustering import ClusteringConfig

    return ClusteringConfig(
        algorithm_name="hdbscan",
        params={
            "min_cluster_size": 5,
            "min_samples": 3
        },
        enable_temporal_weighting=True,
        temporal_decay_factor=7.0,
        metadata_filters=None
    )


@pytest.fixture
def clustering_config_with_filters():
    """Clustering config with metadata filters."""
    from src.core.base_clustering import ClusteringConfig

    return ClusteringConfig(
        algorithm_name="hdbscan",
        params={
            "min_cluster_size": 5,
            "min_samples": 3
        },
        enable_temporal_weighting=False,
        temporal_decay_factor=7.0,
        metadata_filters={
            "domain": "diplomatic",
            "event_type": "contact_meet"
        }
    )


# =============================================================================
# FAISS Fixtures
# =============================================================================

@pytest.fixture
def mock_faiss_index(sample_vectors):
    """Create mock FAISS index for testing."""
    try:
        import faiss
    except ImportError:
        pytest.skip("FAISS not available")

    # Create flat index
    dimension = sample_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(sample_vectors)

    return index


@pytest.fixture
def temp_faiss_indices(sample_vectors, sample_metadata):
    """Create temporary FAISS indices for testing."""
    import pickle
    try:
        import faiss
    except ImportError:
        pytest.skip("FAISS not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create index for each embedding type
        for emb_type in ["document", "event", "entity", "storyline"]:
            # Create FAISS index
            dimension = sample_vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(sample_vectors)

            # Save index
            index_path = tmpdir / f"{emb_type}s.index"
            faiss.write_index(index, str(index_path))

            # Save metadata
            metadata_path = tmpdir / f"{emb_type}s_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(sample_metadata, f)

        yield tmpdir


# =============================================================================
# Storage Fixtures
# =============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = MagicMock(spec=redis.Redis)
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = False
    return mock


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL connection for testing."""
    mock = MagicMock()
    mock.execute.return_value = None
    mock.fetch.return_value = []
    mock.fetchrow.return_value = None
    return mock


# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.orchestrator import app

    return TestClient(app)


# =============================================================================
# Cleanup
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup code here if needed
    pass


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for full workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >1 second"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_stage3: Tests requiring Stage 3 indices"
    )
    config.addinivalue_line(
        "markers", "requires_redis: Tests requiring Redis connection"
    )
    config.addinivalue_line(
        "markers", "requires_postgres: Tests requiring PostgreSQL connection"
    )
