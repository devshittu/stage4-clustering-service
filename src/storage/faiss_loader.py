"""
faiss_loader.py

FAISS index loader for Stage 4 Clustering Service.
Loads vector indices from Stage 3 and provides similarity search with metadata filtering.

Features:
- Load indices from /shared/stage3/data/vector_indices
- GPU support with CPU fallback
- Metadata loading and filtering
- Similarity search with metadata constraints
- Index caching for performance
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-gpu or faiss-cpu")

from src.config.settings_loader import get_settings
from src.schemas.data_models import EmbeddingType

logger = logging.getLogger(__name__)


# =============================================================================
# FAISS Loader
# =============================================================================

class FAISSLoader:
    """
    FAISS index loader with metadata support.

    Loads vector indices from Stage 3 and provides search functionality
    with metadata filtering capabilities.
    """

    def __init__(self, indices_path: Optional[str] = None, use_gpu: Optional[bool] = None):
        """
        Initialize FAISS loader.

        Args:
            indices_path: Path to FAISS indices (uses config if None)
            use_gpu: Whether to use GPU (uses config if None)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-gpu or faiss-cpu")

        settings = get_settings()
        self.indices_path = Path(indices_path or settings.faiss.indices_path)
        self.use_gpu = use_gpu if use_gpu is not None else settings.faiss.use_gpu

        # Index cache
        self._indices: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded: Dict[str, bool] = {}

        # GPU resources
        self._gpu_resources = None
        if self.use_gpu:
            self._initialize_gpu()

        logger.info(
            f"FAISSLoader initialized (path={self.indices_path}, gpu={self.use_gpu})"
        )

    def _initialize_gpu(self) -> None:
        """Initialize GPU resources if available."""
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_resources = faiss.StandardGpuResources()
                logger.info("FAISS GPU resources initialized")
            else:
                logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
                self.use_gpu = False
        except Exception as e:
            logger.warning(f"Failed to initialize GPU resources: {e}. Falling back to CPU.")
            self.use_gpu = False

    def load_index(self, embedding_type: EmbeddingType, force_reload: bool = False) -> bool:
        """
        Load FAISS index for a specific embedding type.

        Args:
            embedding_type: Type of embedding (document, event, entity, storyline)
            force_reload: Force reload even if cached

        Returns:
            True if loaded successfully, False otherwise
        """
        type_key = embedding_type.value

        # Return cached index if available
        if type_key in self._loaded and self._loaded[type_key] and not force_reload:
            logger.debug(f"Using cached index for {type_key}")
            return True

        try:
            # Construct index path
            index_filename = f"{type_key}s.index"  # documents.index, events.index, etc.
            index_path = self.indices_path / index_filename
            metadata_path = self.indices_path / f"{type_key}s_metadata.pkl"

            if not index_path.exists():
                logger.error(f"Index file not found: {index_path}")
                return False

            # Load FAISS index
            logger.info(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))

            # Move to GPU if enabled
            if self.use_gpu and self._gpu_resources:
                logger.info(f"Moving index to GPU: {type_key}")
                index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)

            self._indices[type_key] = index

            # Load metadata if available
            if metadata_path.exists():
                logger.info(f"Loading metadata from {metadata_path}")
                with open(metadata_path, 'rb') as f:
                    self._metadata[type_key] = pickle.load(f)
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                self._metadata[type_key] = []

            self._loaded[type_key] = True
            logger.info(
                f"Successfully loaded {type_key} index "
                f"(vectors={index.ntotal}, dim={index.d})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load index for {type_key}: {e}", exc_info=True)
            return False

    def search(
        self,
        embedding_type: EmbeddingType,
        query_vectors: np.ndarray,
        k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        """
        Search for similar vectors with optional metadata filtering.

        Args:
            embedding_type: Type of embedding to search
            query_vectors: Query vectors (shape: [n_queries, dim])
            k: Number of neighbors to return
            metadata_filters: Optional metadata filters (e.g., {"domain": "politics"})

        Returns:
            Tuple of (distances, indices, metadata_list)
            - distances: shape [n_queries, k]
            - indices: shape [n_queries, k]
            - metadata_list: List of metadata dicts for each result

        Raises:
            ValueError: If index not loaded or invalid query
        """
        type_key = embedding_type.value

        # Ensure index is loaded
        if type_key not in self._loaded or not self._loaded[type_key]:
            if not self.load_index(embedding_type):
                raise ValueError(f"Failed to load index for {type_key}")

        index = self._indices[type_key]
        metadata = self._metadata.get(type_key, [])

        # Validate query vectors
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        if query_vectors.shape[1] != index.d:
            raise ValueError(
                f"Query vector dimension {query_vectors.shape[1]} "
                f"does not match index dimension {index.d}"
            )

        # Convert to float32 (FAISS requirement)
        query_vectors = query_vectors.astype(np.float32)

        # Perform search
        logger.debug(
            f"Searching {type_key} index: queries={query_vectors.shape[0]}, k={k}"
        )

        if metadata_filters:
            # Filter by metadata
            distances, indices, filtered_metadata = self._search_with_filters(
                index, metadata, query_vectors, k, metadata_filters
            )
        else:
            # Direct search
            distances, indices = index.search(query_vectors, k)
            # Gather metadata
            filtered_metadata = []
            for query_indices in indices:
                query_metadata = []
                for idx in query_indices:
                    if idx >= 0 and idx < len(metadata):
                        query_metadata.append(metadata[idx])
                    else:
                        query_metadata.append({})
                filtered_metadata.append(query_metadata)

        return distances, indices, filtered_metadata

    def _search_with_filters(
        self,
        index: faiss.Index,
        metadata: List[Dict[str, Any]],
        query_vectors: np.ndarray,
        k: int,
        filters: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        """
        Search with metadata filtering (post-filtering approach).

        Args:
            index: FAISS index
            metadata: Metadata list
            query_vectors: Query vectors
            k: Number of results
            filters: Metadata filters

        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        # Search for more candidates (k * 10) to allow for filtering
        k_candidates = min(k * 10, index.ntotal)
        distances, indices = index.search(query_vectors, k_candidates)

        # Filter results
        filtered_distances = []
        filtered_indices = []
        filtered_metadata = []

        for query_idx, (query_dists, query_indices) in enumerate(zip(distances, indices)):
            query_filtered_dists = []
            query_filtered_indices = []
            query_filtered_metadata = []

            for dist, idx in zip(query_dists, query_indices):
                if idx < 0 or idx >= len(metadata):
                    continue

                item_metadata = metadata[idx]

                # Apply filters
                if self._matches_filters(item_metadata, filters):
                    query_filtered_dists.append(dist)
                    query_filtered_indices.append(idx)
                    query_filtered_metadata.append(item_metadata)

                    if len(query_filtered_dists) >= k:
                        break

            # Pad if needed
            while len(query_filtered_dists) < k:
                query_filtered_dists.append(float('inf'))
                query_filtered_indices.append(-1)
                query_filtered_metadata.append({})

            filtered_distances.append(query_filtered_dists)
            filtered_indices.append(query_filtered_indices)
            filtered_metadata.append(query_filtered_metadata)

        return (
            np.array(filtered_distances, dtype=np.float32),
            np.array(filtered_indices, dtype=np.int64),
            filtered_metadata
        )

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filters.

        Args:
            metadata: Item metadata
            filters: Filter criteria

        Returns:
            True if matches all filters
        """
        for key, value in filters.items():
            if key not in metadata:
                return False

            meta_value = metadata[key]

            # List matching (item in list)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            # Exact matching
            elif meta_value != value:
                return False

        return True

    def get_vectors_by_ids(
        self,
        embedding_type: EmbeddingType,
        ids: List[str]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieve vectors and metadata by IDs.

        Args:
            embedding_type: Type of embedding
            ids: List of IDs to retrieve

        Returns:
            Tuple of (vectors, metadata_list)

        Raises:
            ValueError: If index not loaded
        """
        type_key = embedding_type.value

        # Ensure index is loaded
        if type_key not in self._loaded or not self._loaded[type_key]:
            if not self.load_index(embedding_type):
                raise ValueError(f"Failed to load index for {type_key}")

        index = self._indices[type_key]
        metadata = self._metadata.get(type_key, [])

        # Find indices by ID
        id_to_index = {
            meta.get('source_id', ''): idx
            for idx, meta in enumerate(metadata)
        }

        indices = []
        found_metadata = []
        for item_id in ids:
            if item_id in id_to_index:
                idx = id_to_index[item_id]
                indices.append(idx)
                found_metadata.append(metadata[idx])
            else:
                logger.warning(f"ID not found in index: {item_id}")

        if not indices:
            return np.array([]), []

        # Reconstruct vectors from index
        vectors = np.zeros((len(indices), index.d), dtype=np.float32)
        for i, idx in enumerate(indices):
            vectors[i] = index.reconstruct(int(idx))

        return vectors, found_metadata

    def get_index_stats(self, embedding_type: EmbeddingType) -> Dict[str, Any]:
        """
        Get statistics about an index.

        Args:
            embedding_type: Type of embedding

        Returns:
            Dictionary with index statistics
        """
        type_key = embedding_type.value

        if type_key not in self._loaded or not self._loaded[type_key]:
            if not self.load_index(embedding_type):
                return {"loaded": False}

        index = self._indices[type_key]
        metadata = self._metadata.get(type_key, [])

        return {
            "loaded": True,
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "metadata_count": len(metadata),
            "on_gpu": self.use_gpu,
            "index_type": type(index).__name__
        }

    def unload_index(self, embedding_type: EmbeddingType) -> None:
        """
        Unload index from memory.

        Args:
            embedding_type: Type of embedding to unload
        """
        type_key = embedding_type.value

        if type_key in self._indices:
            del self._indices[type_key]
            del self._metadata[type_key]
            self._loaded[type_key] = False
            logger.info(f"Unloaded index: {type_key}")

    def unload_all(self) -> None:
        """Unload all indices from memory."""
        for type_key in list(self._indices.keys()):
            embedding_type = EmbeddingType(type_key)
            self.unload_index(embedding_type)
        logger.info("Unloaded all indices")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.unload_all()


# =============================================================================
# Singleton Instance
# =============================================================================

_faiss_loader_instance: Optional[FAISSLoader] = None


def get_faiss_loader() -> FAISSLoader:
    """
    Get singleton FAISS loader instance.

    Returns:
        FAISSLoader instance
    """
    global _faiss_loader_instance

    if _faiss_loader_instance is None:
        _faiss_loader_instance = FAISSLoader()

    return _faiss_loader_instance


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test FAISS loader
    try:
        loader = FAISSLoader()

        # Load document index
        success = loader.load_index(EmbeddingType.DOCUMENT)
        print(f"Document index loaded: {success}")

        if success:
            stats = loader.get_index_stats(EmbeddingType.DOCUMENT)
            print(f"Index stats: {stats}")

            # Test search
            query = np.random.randn(1, 768).astype(np.float32)
            distances, indices, metadata = loader.search(
                EmbeddingType.DOCUMENT,
                query,
                k=5
            )
            print(f"Search results: distances shape={distances.shape}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
