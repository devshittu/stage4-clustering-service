# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stage 4 Clustering Service** is a GPU-accelerated microservice in the Sequential Storytelling Pipeline that clusters semantically similar embeddings at multiple granularity levels (documents, events, entities, storylines) to identify patterns, themes, and narrative threads across document corpora.

**Position in Pipeline:**
- **Input**: 768D vector embeddings from Stage 3 (FAISS indices + metadata)
- **Output**: Clustered data with labels and relationships for Stage 5
- **Next Stage**: Stage 5 (Graph Construction)

**Hardware Environment:**
- AMD Threadripper (48 cores), 160GB RAM
- NVIDIA RTX A4000 GPU (16GB VRAM, CUDA 12.1+)
- Docker Compose deployment with GPU passthrough

## Essential Commands

### Docker Operations

```bash
# Start with infrastructure (REQUIRED for production)
./run-with-infrastructure.sh start

# Check status
./run-with-infrastructure.sh status

# View logs
./run-with-infrastructure.sh logs

# Stop services
./run-with-infrastructure.sh stop

# Manual Docker operations
docker compose -f docker-compose.infrastructure.yml up -d --build
docker compose -f docker-compose.infrastructure.yml down
docker compose -f docker-compose.infrastructure.yml ps
docker compose -f docker-compose.infrastructure.yml logs -f

# Service-specific logs
docker logs clustering-orchestrator -f
docker logs clustering-celery-worker -f
docker logs clustering-celery-beat -f
```

### Testing Commands

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_clustering_algorithms.py -v

# Integration tests only
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/ --line-length 100

# Lint
flake8 src/ tests/ --max-line-length=100

# Type checking
mypy src/ --strict

# Import sorting
isort src/ tests/ --profile black
```

## Architecture Overview

### Service Components

1. **Orchestrator** (`src/api/orchestrator.py`):
   - FastAPI server accessed via Traefik
   - Handles REST API requests for clustering jobs
   - Submits batch jobs to Celery
   - Provides health checks and statistics endpoints

2. **Celery Worker** (`src/api/celery_worker.py`):
   - GPU-accelerated batch clustering
   - Processes clustering jobs with HDBSCAN/K-Means
   - Concurrency=22 (multi-core parallelism where possible)
   - Auto-restart after 50 tasks to prevent memory leaks

3. **Celery Beat**:
   - Periodic task scheduler
   - Cleanup old jobs
   - Statistics aggregation

4. **Redis**:
   - Celery message broker (database 6)
   - Celery result backend + cache (database 7)
   - Job metadata and checkpoints

5. **PostgreSQL**:
   - Cluster metadata and results storage
   - Database: `stage4_clustering`

### Core Components

**FAISS Index Loader** (`src/storage/faiss_loader.py`):
- Loads vector indices from Stage 3
- Supports GPU-accelerated similarity search
- Metadata filtering (domain, event_type, etc.)
- Four index types: documents, events, entities, storylines

**Clustering Engine** (`src/core/clustering_engine.py`):
- Implements HDBSCAN, K-Means, Agglomerative algorithms
- Temporal clustering with decay weighting
- Metadata-aware clustering
- Multi-level clustering strategy

**Cluster Storage Manager** (`src/storage/cluster_storage_manager.py`):
- Multi-backend architecture (PostgreSQL, JSONL, Redis cache)
- Atomic cluster persistence
- Cluster retrieval and search

**Data Models** (`src/schemas/data_models.py`):
- All schemas use Pydantic v2 for validation
- Key models:
  - `ClusteringJob`: Batch job metadata
  - `DocumentCluster`, `EventCluster`, `EntityCluster`, `StorylineCluster`
  - `ClusterResult`: Unified output format
  - `ClusterSearchRequest`, `ClusterSearchResult`

### Enterprise Features

**Job Lifecycle Management** (`src/utils/job_manager.py`):
- Non-blocking job submission
- Pause/resume/cancel operations
- Progressive checkpointing
- Job status tracking

**Resource Management** (`src/utils/resource_manager.py`):
- GPU memory monitoring
- Idle detection and cleanup
- Model unloading on idle timeout
- Proactive OOM prevention

**Advanced Logging** (`src/utils/advanced_logging.py`):
- Structured JSON logging
- Correlation IDs for distributed tracing
- Performance metrics
- GPU/CPU/memory tracking

## Key Configuration

**Main Config**: `config/settings.yaml`

Critical settings to understand:

```yaml
clustering:
  default_algorithm: "hdbscan"
  algorithms:
    hdbscan:
      min_cluster_size: 5
      min_samples: 3
  temporal_clustering:
    enabled: true
    decay_factor: 7  # days

faiss:
  use_gpu: true
  indices_path: "/shared/stage3/data/vector_indices"

storage:
  enabled_backends: ["postgresql", "jsonl"]

batch:
  worker:
    concurrency: 22
    max_tasks_per_child: 50
```

**Environment Variables** (`.env`):
```bash
STAGE4_POSTGRES_PASSWORD=your_secure_password
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO
STAGE3_API_URL=http://embeddings-orchestrator:8000
```

## API Endpoints

Base URL: `http://localhost/api/v1/clustering` (via Traefik)

**Interactive Docs**: `http://localhost/api/v1/clustering/docs` (Swagger UI)

Key endpoints:

- `POST /batch` - Submit clustering job (async)
- `GET /jobs/{job_id}` - Get job status
- `PATCH /jobs/{job_id}` - Pause/resume/cancel job
- `GET /clusters` - List clusters (paginated)
- `GET /clusters/{cluster_id}` - Get cluster details
- `POST /clusters/search` - Search clusters by metadata
- `GET /health` - Service health check
- `GET /statistics` - Clustering statistics

## Critical Architecture Patterns

### Multi-Level Clustering Strategy

The service performs clustering at 4 granularity levels:

1. **Document-Level**: Group similar articles (topic modeling)
2. **Event-Level**: Identify related events across documents
3. **Entity-Level**: Entity disambiguation and coreference resolution
4. **Storyline-Level**: Meta-narratives across storylines

Each level uses appropriate algorithms and parameters.

### Temporal Clustering

**Critical**: Leverage temporal metadata for time-aware clustering.

```python
# Temporal decay weighting
temporal_weight = exp(-abs(date1 - date2) / decay_factor)
adjusted_similarity = semantic_similarity * temporal_weight
```

### Metadata-Aware Clustering

Use rich metadata from Stage 3 for filtered clustering:
- Domain filtering (12 domains)
- Event type filtering (20+ types)
- Entity type filtering (9 types)
- Temporal window filtering

### GPU Resource Management

**Critical**: FAISS-GPU operations require careful memory management.

- Monitor GPU memory usage
- Clear cache at threshold (14GB for 16GB GPU)
- Fall back to CPU if OOM
- Single GPU shared with Stage 3

## Data Flow

```
Stage 3 FAISS Indices → Index Loader → Clustering Engine → Cluster Storage → Stage 5
                                              ↓
                                         PostgreSQL / JSONL
```

**Handover Files**:
- Input: `../stage3_embedding_service/data/vector_indices/*.index` (from Stage 3)
- Output: `data/clusters/clusters_*.jsonl` (for Stage 5)

## Common Troubleshooting

### Cannot Load FAISS Indices
```bash
# Verify Stage 3 is running
cd ../stage3_embedding_service
./run-with-infrastructure.sh status

# Check volume mount
docker inspect clustering-orchestrator | jq '.[0].Mounts'

# Verify indices exist
ls -la ../stage3_embedding_service/data/vector_indices/
```

### Out of GPU Memory
- Reduce clustering batch size in `config/settings.yaml`
- Use CPU fallback for FAISS operations
- Clear GPU cache between batches
- Check GPU memory: `nvidia-smi`

### Poor Clustering Quality
- Tune `min_cluster_size` parameter
- Apply metadata filters (domain, event_type)
- Try different algorithms (HDBSCAN vs K-Means)
- Check similarity threshold

### Slow Performance
- Use FAISS-GPU for similarity search
- Parallelize with multiprocessing (22 workers)
- Pre-cluster with K-Means, refine with HDBSCAN
- Cache cluster results in Redis

## Performance Benchmarks

**Target Performance** (RTX A4000):
- Event clustering (5K events): <10 minutes
- Document clustering (1K docs): <5 minutes
- Entity clustering (10K entities): <15 minutes
- Real-time lookup: <100ms (cached)

**Scaling**:
- Multiple workers supported (CPU parallelism)
- Single GPU shared with Stage 3
- Horizontal scaling via Redis shared cache

## Integration Points

### Stage 3 → Stage 4
- **FAISS Indices**: Read from `../stage3_embedding_service/data/vector_indices/`
- **Metadata**: JSON files alongside indices
- **API**: Optional - call Stage 3 for on-demand similarity search
- Schema: `EmbeddedDocument` (see Stage 3 schemas)

### Stage 4 → Stage 5
- **Cluster Results**: JSONL files in `data/clusters/`
- **Metadata**: Stored in PostgreSQL
- **Format**: `ClusterResult` (see `src/schemas/data_models.py`)

## Code Organization Principles

- **Separation of concerns**: Core logic separate from API/storage
- **Interface-based design**: `ClusteringAlgorithm` base class for extensibility
- **Dependency injection**: Configuration passed to constructors
- **Error isolation**: Algorithm failures don't cascade
- **Pydantic validation**: All inputs/outputs validated

## Testing Strategy

- **Unit tests**: Test individual components (clustering algorithms, loaders)
- **Integration tests**: Test Stage 3 integration, full clustering pipeline
- **Performance tests**: Benchmark clustering speed and quality
- GPU tests require `CUDA_VISIBLE_DEVICES` environment variable

## Infrastructure Integration

⚠️ **CRITICAL**: This stage MUST use centralized infrastructure.

**Required Reading** (Before any infrastructure changes):

1. **Rules**: `../infrastructure/.claude/rules/infrastructure-integration.md` (MANDATORY)
2. **Guide**: `../infrastructure/.claude/references/infrastructure-integration-guide.md`
3. **Example**: `/stage2-nlp-processing/` and `/stage3_embedding_service/` (reference implementations)
4. **Startup Guide**: `STARTUP_GUIDE.md` (Complete Stage 3 handoff)

**This Stage's Configuration**:

- Stage Number: 4
- Redis Celery DB: 6 (calculated: (4-1)*2 = 6)
- Redis Cache DB: 7 (calculated: (4-1)*2+1 = 7)
- PostgreSQL DB: `stage4_clustering`
- Traefik Route: `/api/v1/clustering/*`

## Port Configuration

**Important**: No direct port exposure. Access via Traefik.

- API: Accessed via `http://localhost/api/v1/clustering/*` (Traefik routing)
- Internal: Container port 8000 (not exposed)
- Prometheus: Port 9090 (optional, for metrics)

## Development Tips

### Adding New Clustering Algorithm

1. Create class in `src/core/algorithms/`
2. Inherit from `ClusteringAlgorithm` base class
3. Implement `cluster()` method
4. Register in `ClusteringEngine`
5. Add configuration to `config/settings.yaml`
6. Add tests in `tests/unit/test_algorithms.py`

### Debugging Clustering Issues

1. Enable DEBUG logging in `.env`
2. Check cluster quality metrics (silhouette score, purity)
3. Visualize clusters (if small dataset)
4. Compare algorithms (HDBSCAN vs K-Means)
5. Inspect metadata filters

### Performance Optimization

1. Profile with `cProfile` or `line_profiler`
2. Use FAISS-GPU for similarity search
3. Batch operations to maximize GPU utilization
4. Cache frequently accessed clusters
5. Pre-cluster with fast algorithm, refine with quality algorithm

## Startup Checklist

Before starting development:

- [ ] Read `STARTUP_GUIDE.md` (complete handoff from Stage 3)
- [ ] Verify infrastructure is running (`cd ../infrastructure && ./scripts/start.sh`)
- [ ] Verify Stage 3 is running and producing indices
- [ ] Copy `.env.example` to `.env` and set passwords
- [ ] Build and start: `./run-with-infrastructure.sh start`
- [ ] Verify health: `curl http://localhost/api/v1/clustering/health`
- [ ] Run tests: `pytest tests/ -v`

## Support Resources

- **Startup Guide**: `STARTUP_GUIDE.md` (Stage 3 handoff document)
- **Infrastructure Docs**: `../infrastructure/README.md`
- **Stage 3 Reference**: `../stage3_embedding_service/.claude/CLAUDE.md`
- **FAISS Docs**: https://github.com/facebookresearch/faiss/wiki
- **HDBSCAN Docs**: https://hdbscan.readthedocs.io/

---

**Generated with**: Claude Code v2.0.69
**Last Updated**: December 24, 2025
