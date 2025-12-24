# Stage 4: Clustering Service

**Version:** 1.0.0
**Last Updated:** December 24, 2025

## Overview

The Stage 4 Clustering Service is a production-ready microservice within the Sequential Storytelling Pipeline that clusters semantically similar embeddings at multiple granularity levels to identify patterns, themes, and narrative threads across document corpora.

### Key Features

- **Multi-Level Clustering**: Document, event, entity, and storyline clustering
- **GPU-Accelerated**: FAISS-GPU for fast similarity search
- **Advanced Algorithms**: HDBSCAN, K-Means, Agglomerative clustering
- **Temporal Awareness**: Time-aware clustering with decay weighting
- **Metadata Filtering**: Domain and event-type aware clustering
- **RESTful API**: FastAPI-based interface with OpenAPI documentation
- **Batch Processing**: Celery-based distributed processing
- **Production-Ready**: Docker Compose deployment with health checks and monitoring

## Position in Pipeline

```
Stage 2 (NLP Processing)  →  Stage 3 (Embeddings)  →  Stage 4 (Clustering)  →  Stage 5 (Graph)
   ↓ ProcessedDocument         ↓ 768D Vectors            ↓ Clustered Data       ↓ Knowledge Graph
```

**Input**: FAISS vector indices from Stage 3 (documents, events, entities, storylines)
**Output**: Clustered entities/events with labels and metadata for Stage 5
**Hardware**: AMD Threadripper (48 cores), 160GB RAM, RTX A4000 GPU

## Quick Start

### Prerequisites

- Docker Engine 29.1.2+
- Docker Compose 5.0.0+
- NVIDIA GPU with 8GB+ VRAM (16GB recommended)
- CUDA 12.1+ compatible GPU
- Access to centralized infrastructure (Redis, PostgreSQL, Traefik)
- Stage 3 running and producing vector indices

### Installation

1. **Read the startup guide** (CRITICAL):
   ```bash
   cat STARTUP_GUIDE.md
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env and set your passwords
   ```

3. **Ensure infrastructure is running**:
   ```bash
   cd ../infrastructure
   ./scripts/start.sh
   ```

4. **Ensure Stage 3 is running**:
   ```bash
   cd ../stage3_embedding_service
   ./run-with-infrastructure.sh start
   ```

5. **Start Stage 4 with infrastructure**:
   ```bash
   cd ../stage4-clustering-service
   ./run-with-infrastructure.sh start
   ```

6. **Verify services are running**:
   ```bash
   docker compose -f docker-compose.infrastructure.yml ps
   docker logs clustering-orchestrator -f
   ```

7. **Check health**:
   ```bash
   curl http://localhost/api/v1/clustering/health
   ```

### First Clustering Job

```bash
# Submit event clustering job
curl -X POST "http://localhost/api/v1/clustering/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_type": "events",
    "algorithm": "hdbscan",
    "min_cluster_size": 5
  }'

# Check job status
curl "http://localhost/api/v1/clustering/jobs/{job_id}"

# Get results
curl "http://localhost/api/v1/clustering/clusters?embedding_type=event"
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  - /api/v1/cluster/batch       - /api/v1/clusters          │
│  - /api/v1/cluster/status      - /health                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              FAISS Index Loader                              │
│  Reads Stage 3 indices: docs, events, entities, storylines  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Clustering Engine                                 │
│  Algorithms: HDBSCAN, K-Means, Agglomerative               │
│  Features: Temporal weighting, metadata filtering           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Cluster Storage                                   │
│  Backends: PostgreSQL, JSONL, Redis Cache                   │
└─────────────────────────────────────────────────────────────┘
```

### Service Architecture

- **Orchestrator**: FastAPI server (accessed via Traefik)
- **Celery Worker**: Batch clustering with GPU (22 workers)
- **Celery Beat**: Periodic tasks (cleanup, statistics)
- **Redis**: Job queue and result cache
- **PostgreSQL**: Cluster metadata and results

## Configuration

### Main Configuration File

Create `config/settings.yaml`:

```yaml
clustering:
  default_algorithm: "hdbscan"
  algorithms:
    hdbscan:
      min_cluster_size: 5
      min_samples: 3
      cluster_selection_epsilon: 0.0
    kmeans:
      n_clusters: 50
      n_init: 10
      max_iter: 300

  temporal_clustering:
    enabled: true
    decay_factor: 7  # days
    max_temporal_gap: 30  # days

  metadata_filtering:
    enabled: true
    filter_by_domain: true
    filter_by_event_type: true

faiss:
  use_gpu: true
  indices_path: "/shared/stage3/data/vector_indices"

storage:
  enabled_backends: ["postgresql", "jsonl"]
  output_dir: "data/clusters"
```

## API Reference

### Base URL

```
http://localhost/api/v1/clustering
```

### Endpoints

#### 1. Submit Batch Clustering Job

**POST** `/batch`

```json
// Request
{
  "embedding_type": "events",
  "algorithm": "hdbscan",
  "min_cluster_size": 5,
  "metadata_filters": {
    "domain": "diplomatic_relations"
  }
}

// Response (202 Accepted)
{
  "job_id": "job_abc123",
  "status": "queued",
  "embedding_type": "events",
  "algorithm": "hdbscan",
  "created_at": "2025-12-24T10:00:00Z"
}
```

#### 2. Get Job Status

**GET** `/jobs/{job_id}`

```json
{
  "job_id": "job_abc123",
  "status": "running",
  "progress_percent": 45.0,
  "clusters_found": 12,
  "items_clustered": 2250,
  "total_items": 5000,
  "started_at": "2025-12-24T10:00:05Z"
}
```

#### 3. List Clusters

**GET** `/clusters?embedding_type=event&limit=50`

```json
{
  "clusters": [
    {
      "cluster_id": "event_cluster_001",
      "cluster_label": "G7 Summit Diplomatic Meetings",
      "cluster_size": 45,
      "member_ids": ["doc_001_event_0", ...],
      "temporal_span": ["2025-12-10", "2025-12-15"],
      "domain": "diplomatic_relations"
    }
  ],
  "total": 25,
  "page": 1
}
```

#### 4. Get Cluster Details

**GET** `/clusters/{cluster_id}`

```json
{
  "cluster_id": "event_cluster_001",
  "cluster_label": "G7 Summit Diplomatic Meetings",
  "cluster_size": 45,
  "member_ids": ["doc_001_event_0", "doc_002_event_3", ...],
  "document_ids": ["doc_001", "doc_002", ...],
  "centroid_vector": [0.1, 0.2, ..., 0.3],
  "temporal_span": ["2025-12-10T00:00:00Z", "2025-12-15T00:00:00Z"],
  "primary_entities": ["G7", "Japan", "United States"],
  "domain": "diplomatic_relations",
  "event_type": "contact_meet",
  "intra_cluster_similarity": 0.87,
  "metadata": {
    "algorithm": "hdbscan",
    "timestamp": "2025-12-24T10:00:00Z"
  }
}
```

## Development

### Local Setup (without Docker)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Code quality
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_clustering_algorithms.py -v
```

## Integration with Infrastructure

**CRITICAL**: This service MUST use centralized infrastructure.

### Required Reading

1. `../infrastructure/.claude/rules/infrastructure-integration.md` (MANDATORY)
2. `STARTUP_GUIDE.md` (Complete handoff from Stage 3)
3. `../stage2-nlp-processing/` (Reference implementation)

### Key Integration Points

- **Redis DB**: 6 (Celery broker), 7 (cache) - Stage 4 allocation
- **PostgreSQL DB**: `stage4_clustering`
- **Traefik Route**: `/api/v1/clustering/*`
- **Network**: `storytelling` (external)
- **Volume**: Mount Stage 3's `data/vector_indices/` (read-only)

## Performance Benchmarks

**Target Performance** (RTX A4000):
- Event clustering (5K events): <10 minutes
- Document clustering (1K docs): <5 minutes
- Real-time cluster lookup: <100ms (cached)
- Concurrent jobs: 1 (GPU serialization)

## Monitoring

### Health Check

```bash
curl http://localhost/api/v1/clustering/health
```

### Statistics

```bash
curl http://localhost/api/v1/clustering/statistics
```

### Logs

```bash
# Orchestrator logs
docker logs clustering-orchestrator -f

# Celery worker logs
docker logs clustering-celery-worker -f
```

## Troubleshooting

### Cannot Load FAISS Indices

**Solution**:
- Verify Stage 3 is running and has created indices
- Check volume mount in `docker-compose.infrastructure.yml`
- Ensure path `/shared/stage3/data/vector_indices` is accessible

### Out of GPU Memory

**Solution**:
- Reduce clustering batch size
- Use CPU fallback for FAISS
- Clear GPU cache between operations

### Poor Clustering Quality

**Solution**:
- Tune `min_cluster_size` parameter
- Apply metadata filters (domain, event_type)
- Try different algorithms (HDBSCAN vs K-Means)

## Contributing

### Code Structure

```
src/
├── api/              # FastAPI orchestrator, routes
├── core/             # Clustering algorithms
├── schemas/          # Pydantic models
├── storage/          # FAISS loader, cluster storage
├── utils/            # Logging, monitoring
└── config/           # YAML settings loader
```

### Development Workflow

1. Create feature branch
2. Implement with tests (>80% coverage)
3. Format: `black src/ tests/`
4. Lint: `flake8 src/ tests/`
5. Type check: `mypy src/`
6. Submit pull request

## License

[Specify License]

## Support

For issues and questions:
- **Startup Guide**: See `STARTUP_GUIDE.md` for complete handoff
- **Infrastructure**: See `../infrastructure/README.md`
- **Stage 3 Integration**: See `../stage3_embedding_service/`

---

**Generated with**: Claude Code v2.0.69
**Last Updated**: December 24, 2025
