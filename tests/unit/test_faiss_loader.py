"""
Unit tests for FAISS index loader.

Tests the FAISSLoader class including:
- Index loading from disk
- Metadata loading
- GPU/CPU mode handling
- Search functionality
- Metadata filtering
- Error handling
"""

import pytest
import numpy as np
from pathlib import Path
from src.schemas.data_models import EmbeddingType


@pytest.mark.unit
class TestFAISSLoader:
    """Test suite for FAISS index loader."""

    def test_init(self):
        """Test FAISSLoader initialization."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(use_gpu=False)
        assert loader is not None
        assert loader.use_gpu == False

    def test_load_index_success(self, temp_faiss_indices):
        """Test successful index loading."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Load document index
        success = loader.load_index(EmbeddingType.DOCUMENT)
        assert success == True

        # Check that index is cached
        assert "document" in loader._loaded
        assert loader._loaded["document"] == True

    def test_load_all_index_types(self, temp_faiss_indices):
        """Test loading all index types."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        for emb_type in [EmbeddingType.DOCUMENT, EmbeddingType.EVENT,
                         EmbeddingType.ENTITY, EmbeddingType.STORYLINE]:
            success = loader.load_index(emb_type)
            assert success == True

    def test_load_index_caching(self, temp_faiss_indices):
        """Test that indices are cached and not reloaded."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Load once
        loader.load_index(EmbeddingType.DOCUMENT)

        # Load again (should use cache)
        success = loader.load_index(EmbeddingType.DOCUMENT, force_reload=False)
        assert success == True

    def test_load_index_force_reload(self, temp_faiss_indices):
        """Test force reload of index."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Load once
        loader.load_index(EmbeddingType.DOCUMENT)

        # Force reload
        success = loader.load_index(EmbeddingType.DOCUMENT, force_reload=True)
        assert success == True

    def test_load_index_not_found(self, tmp_path):
        """Test handling of missing index file."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(tmp_path), use_gpu=False)

        # Try to load non-existent index
        success = loader.load_index(EmbeddingType.DOCUMENT)
        assert success == False

    def test_search_basic(self, temp_faiss_indices, sample_vectors):
        """Test basic similarity search."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        # Search with first vector
        query = sample_vectors[0:1]
        distances, indices, metadata = loader.search(
            EmbeddingType.DOCUMENT,
            query,
            k=5
        )

        # Should return 5 results
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert len(metadata) == 1
        assert len(metadata[0]) == 5

    def test_search_with_metadata_filters(self, temp_faiss_indices, sample_vectors):
        """Test search with metadata filtering."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.EVENT)

        # Search with domain filter
        query = sample_vectors[0:1]
        distances, indices, metadata = loader.search(
            EmbeddingType.EVENT,
            query,
            k=5,
            metadata_filters={"domain": "diplomatic"}
        )

        # Should return results (possibly fewer than k if filtered)
        assert distances.shape[0] == 1
        assert len(metadata) == 1

    def test_get_index_stats(self, temp_faiss_indices):
        """Test retrieving index statistics."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        stats = loader.get_index_stats(EmbeddingType.DOCUMENT)

        # Check stats structure
        assert stats["loaded"] == True
        assert "total_vectors" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 768
        assert "on_gpu" in stats

    def test_get_index_stats_not_loaded(self):
        """Test stats for unloaded index."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(use_gpu=False)

        stats = loader.get_index_stats(EmbeddingType.DOCUMENT)

        # Should indicate not loaded or attempt to load
        assert "loaded" in stats

    def test_unload_index(self, temp_faiss_indices):
        """Test index unloading."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        # Unload index
        loader.unload_index(EmbeddingType.DOCUMENT)

        # Should no longer be loaded
        assert loader._loaded.get("document", False) == False

    def test_unload_all(self, temp_faiss_indices):
        """Test unloading all indices."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Load multiple indices
        loader.load_index(EmbeddingType.DOCUMENT)
        loader.load_index(EmbeddingType.EVENT)

        # Unload all
        loader.unload_all()

        # All should be unloaded
        assert len(loader._indices) == 0

    def test_context_manager(self, temp_faiss_indices):
        """Test FAISSLoader as context manager."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        with FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False) as loader:
            loader.load_index(EmbeddingType.DOCUMENT)
            assert loader._loaded["document"] == True

        # After exiting context, indices should be unloaded
        # (loader object is out of scope, but unload_all should have been called)

    def test_gpu_fallback_to_cpu(self, temp_faiss_indices):
        """Test automatic fallback to CPU when GPU not available."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        # Request GPU even if not available
        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=True)

        # Should fall back to CPU gracefully
        # (use_gpu may be set to False if GPU unavailable)
        success = loader.load_index(EmbeddingType.DOCUMENT)
        assert success == True

    def test_search_invalid_dimension(self, temp_faiss_indices):
        """Test error handling for query vectors with wrong dimension."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        # Query with wrong dimension
        query = np.random.randn(1, 512).astype(np.float32)  # Should be 768

        with pytest.raises(ValueError, match="dimension"):
            loader.search(EmbeddingType.DOCUMENT, query, k=5)

    def test_search_before_loading(self, temp_faiss_indices):
        """Test search on unloaded index (should auto-load)."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)

        # Search without loading first (should auto-load)
        query = np.random.randn(1, 768).astype(np.float32)
        distances, indices, metadata = loader.search(
            EmbeddingType.DOCUMENT,
            query,
            k=5
        )

        # Should succeed (auto-loaded)
        assert distances.shape == (1, 5)

    @pytest.mark.slow
    def test_search_large_k(self, temp_faiss_indices):
        """Test search with large k value."""
        try:
            from src.storage.faiss_loader import FAISSLoader
        except ImportError:
            pytest.skip("FAISS not available")

        loader = FAISSLoader(indices_path=str(temp_faiss_indices), use_gpu=False)
        loader.load_index(EmbeddingType.DOCUMENT)

        # Search with k=50
        query = np.random.randn(1, 768).astype(np.float32)
        distances, indices, metadata = loader.search(
            EmbeddingType.DOCUMENT,
            query,
            k=50
        )

        # Should return up to 50 results (or total_vectors if less)
        assert distances.shape[1] <= 50

    def test_singleton_pattern(self):
        """Test singleton get_faiss_loader function."""
        try:
            from src.storage.faiss_loader import get_faiss_loader
        except ImportError:
            pytest.skip("FAISS not available")

        loader1 = get_faiss_loader()
        loader2 = get_faiss_loader()

        # Should return same instance
        assert loader1 is loader2
