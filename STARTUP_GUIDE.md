# Stage 4: Clustering Service - Development Kickstart

## 1. PROJECT OVERVIEW

### The Sequential Storytelling Pipeline

You are developing **Stage 4 (Clustering Service)** of an 8-stage production-ready microservices pipeline that transforms raw news articles into coherent, temporal narratives by connecting events, entities, and relationships across stories.

**The Complete Pipeline:**

1. **Stage 1: Cleaning & Ingestion** - Preprocesses raw articles, normalizes data
2. **Stage 2: NLP Processing** - Extracts events, entities, SOA triplets, relationships
3. **Stage 3: Embedding Generation** - Generates 768D vector embeddings (UPSTREAM DEPENDENCY)
4. **Stage 4: Clustering Service** ← **YOUR FOCUS**
5. **Stage 5: Graph Construction** - Builds Neo4j knowledge graph
6. **Stage 6: Timeline Generation** - Creates temporal narratives with DeepSeek R1
7. **Stage 7: API Service** - FastAPI backend serving results
8. **Stage 8: Frontend Service** - React/TypeScript interactive UI

**Design Philosophy: Independent Black Box Services**
- Each stage is a separate repository with clear input/output contracts
- Loose coupling: Changes in one stage don't break others (contracts maintained)
- Independent deployment and testing
- Well-defined data transformation at each stage

### Hardware Environment

**Target System:**
- **CPU**: AMD Threadripper 48-core (24 physical, 48 threads)
- **RAM**: 160GB DDR4
- **GPU**: NVIDIA RTX A4000 (16GB VRAM, CUDA 12.1+)
- **OS**: Linux (Ubuntu 22.04+)
- **Docker**: Engine 29.1.2+, Compose 5.0.0+

**Resource Allocation for Stage 4:**
- **Recommended**: 10-15 cores, 20-40GB RAM
- **GPU Access**: Shared access to RTX A4000 (for FAISS-GPU operations)
- **Storage**: SSD for index caching and intermediate results

---

## 2. STAGE 3 CONTRACT (UPSTREAM DEPENDENCY)

### 2.1 Primary Output: FAISS Vector Indices

**Location**: `../stage3_embedding_service/data/vector_indices/`

**Four Index Files (Multi-Granularity Embeddings):**

1. **documents.index** - Document-level embeddings (1 per document)
2. **events.index** - Event-level embeddings (5-15 per document)
3. **entities.index** - Entity-level embeddings (20-30 per document)
4. **storylines.index** - Storyline-level embeddings (2-5 per document)

**Accompanying Metadata Files (JSON):**
- `documents_metadata.json`
- `events_metadata.json`
- `entities_metadata.json`
- `storylines_metadata.json`

### 2.2 Vector Specifications

**CRITICAL CONSTANTS:**
- **Embedding Dimension**: 768 (all-mpnet-base-v2 model)
- **Normalization**: L2-normalized vectors (cosine similarity ready)
- **Index Type**: FAISS flat index (exact search, GPU-compatible)
- **Format**: FAISS binary format (`.index` files)
- **Metadata Format**: JSON with `metadata_store`, `id_map`, `reverse_id_map`

### 2.3 Metadata Structure (Your Clustering Filters)

Each metadata JSON file contains:

```json
{
  "metadata_store": {
    "<vector_id>": {
      "source_id": "<document_id/event_id/entity_id>",
      "embedding_type": "document|event|entity|storyline",
      "metadata": {
        // Document-level metadata
        "title": "Article title",
        "publication_date": "2025-12-15T00:00:00Z",
        "word_count": 850,
        "entity_count": 26,
        "event_count": 15,
        "source_url": "https://...",
        "categories": ["politics", "technology"],
        "tags": ["AI", "regulation"],

        // Event-level metadata (events.index only)
        "event_type": "contact_meet|policy_change|conflict_attack|...",  // 20+ types
        "domain": "diplomatic|military|economic|...",  // 12 domains
        "sentiment": "positive|neutral|negative",
        "confidence": 0.95,
        "temporal_reference": "2025-12-14T00:00:00Z",
        "argument_count": 4,

        // Entity-level metadata (entities.index only)
        "entity_type": "PER|ORG|LOC|GPE|DATE|TIME|MONEY|MISC|EVENT",
        "entity_text": "Joe Biden",
        "context": "... surrounding text ...",

        // Storyline-level metadata (storylines.index only)
        "storyline_id": "storyline_batch123_0",
        "event_count": 12,
        "primary_entities": ["Joe Biden", "White House", "Congress"],
        "temporal_span": ["2025-12-01T00:00:00Z", "2025-12-15T00:00:00Z"],
        "temporal_span_days": 14
      }
    }
  },
  "id_map": {"<vector_id>": <faiss_index_position>},
  "reverse_id_map": {"<faiss_index_position>": "<vector_id>"}
}
```

### 2.4 Data Model Schemas (Pydantic)

**EmbeddedDocument** (Full output from Stage 3):

```python
class EmbeddedDocument(BaseModel):
    document_id: str
    job_id: str
    embedded_at: str  # ISO 8601

    # Source data preserved from Stage 2
    source_processed_document: ProcessedDocument  # See Stage 2 contract

    # Four levels of embeddings
    document_embedding: Optional[DocumentEmbedding]
    event_embeddings: List[EventEmbedding]
    entity_embeddings: List[EntityEmbedding]
    storyline_embeddings: List[StorylineEmbedding]

    # Processing metadata
    embedding_metadata: Dict[str, Any]  # {model_name, dimension, processing_time_ms, etc.}
```

**Individual Embedding Types:**

```python
class DocumentEmbedding(BaseModel):
    document_id: str
    vector_id: str  # Format: "doc_emb_{document_id}"
    vector: List[float]  # 768D, L2-normalized
    text_source: str
    normalized_date: str  # ISO 8601
    metadata: Dict[str, Any]

class EventEmbedding(BaseModel):
    event_id: str
    document_id: str
    vector_id: str  # Format: "event_emb_{event_id}"
    vector: List[float]  # 768D
    event_type: str  # ACE 2005 + custom types
    domain: str  # 12 domains
    storyline_id: Optional[str]  # Pre-clustered if from Stage 2
    event_description: str
    metadata: Dict[str, Any]

class EntityEmbedding(BaseModel):
    entity_id: str
    document_id: str
    vector_id: str  # Format: "entity_emb_{entity_id}"
    vector: List[float]  # 768D
    entity_text: str
    entity_type: str
    context: str
    metadata: Dict[str, Any]

class StorylineEmbedding(BaseModel):
    storyline_id: str
    vector_id: str  # Format: "storyline_emb_{storyline_id}"
    vector: List[float]  # 768D
    event_ids: List[str]
    primary_entities: List[str]
    domain: str
    temporal_span: Optional[List[str]]  # [start, end]
    storyline_summary: str
    metadata: Dict[str, Any]
```

### 2.5 Stage 3 API (Optional Integration Method)

**Base URL**: `http://embeddings-orchestrator:8000` (internal) or `http://localhost/api/v1/embeddings` (via Traefik)

**Key Endpoints for Stage 4:**

#### POST /api/v1/search
Search for similar vectors (if you need on-demand similarity search)

```json
// Request
{
  "embedding_type": "event",
  "query_vector": [0.1, 0.2, ..., 0.3],  // 768D
  "top_k": 100,
  "filter_metadata": {
    "domain": "diplomatic",
    "event_type": "contact_meet"
  }
}

// Response
{
  "query_id": "search_abc123",
  "results": [
    {
      "vector_id": "event_emb_doc_001_event_0",
      "source_id": "doc_001_event_0",
      "embedding_type": "event",
      "similarity_score": 0.95,
      "metadata": {...}
    }
  ],
  "search_time_ms": 12.3,
  "total_results": 100
}
```

#### GET /statistics
Get index statistics (useful for monitoring)

```json
{
  "enabled_backends": ["faiss"],
  "primary_backend": "faiss",
  "backends": {
    "faiss": {
      "indices": [
        {"name": "documents", "total_vectors": 1000, "dimension": 768},
        {"name": "events", "total_vectors": 5000, "dimension": 768},
        {"name": "entities", "total_vectors": 8000, "dimension": 768},
        {"name": "storylines", "total_vectors": 2000, "dimension": 768}
      ],
      "total_vectors": 16000
    }
  }
}
```

#### GET /health
Health check for upstream dependency monitoring

### 2.6 Performance Characteristics (Stage 3 Output)

**Throughput:**
- Documents embedded: ~500 docs/hour
- Events embedded: ~2000 events/hour
- Entities embedded: ~3000 entities/hour

**Typical Output per Document:**
- 1 document embedding
- 5-15 event embeddings
- 20-30 entity embeddings
- 2-5 storyline embeddings
- **Total**: ~30-50 vectors per document

**Index Growth Estimates:**
- 1,000 documents → 30,000-50,000 vectors
- 10,000 documents → 300,000-500,000 vectors
- FAISS index size: ~100-150MB per 100K vectors (768D)

---

## 3. STAGE 4 EXPECTATIONS (YOUR DELIVERABLES)

### 3.1 Core Clustering Tasks

**Primary Objective**: Group semantically similar embeddings at multiple granularity levels to identify patterns, themes, and narrative threads across the document corpus.

**Four-Level Clustering Strategy (Multi-Granularity):**

#### Level 1: Document-Level Clustering
**Input**: `documents.index` (1 embedding per document)
**Goal**: Group similar articles by overall theme/topic
**Use Cases**:
- Topic modeling (e.g., "Ukraine War Coverage", "AI Regulation News")
- Content deduplication
- Trend detection

**Expected Output**:
```python
class DocumentCluster(BaseModel):
    cluster_id: str
    cluster_label: str  # Auto-generated or LLM-summarized
    document_ids: List[str]
    centroid_vector: List[float]  # 768D
    cluster_size: int
    temporal_span: Optional[List[str]]  # [earliest, latest publication date]
    metadata: Dict[str, Any]  # {avg_word_count, dominant_categories, etc.}
```

#### Level 2: Event-Level Clustering
**Input**: `events.index` (5-15 embeddings per document)
**Goal**: Identify related events across documents
**Use Cases**:
- Cross-document event tracking (e.g., "G7 Summit Events")
- Event coreference resolution
- Multi-document event timelines

**Critical**: Leverage metadata filters:
- `domain`: Cluster within same domain (diplomatic, military, etc.)
- `event_type`: Separate attack events from meetings
- `temporal_reference`: Temporal proximity weighting

**Expected Output**:
```python
class EventCluster(BaseModel):
    cluster_id: str
    cluster_label: str
    event_ids: List[str]  # Cross-document event IDs
    document_ids: List[str]  # Source documents
    centroid_vector: List[float]
    cluster_size: int
    event_type: str  # Dominant event type
    domain: str
    temporal_span: Optional[List[str]]
    primary_entities: List[str]  # Most frequent entities
    metadata: Dict[str, Any]
```

#### Level 3: Entity-Level Clustering
**Input**: `entities.index` (20-30 embeddings per document)
**Goal**: Entity disambiguation and coreference resolution
**Use Cases**:
- Resolve entity mentions (e.g., "Biden", "President Biden", "Joe Biden" → same cluster)
- Cross-document entity tracking
- Entity alias detection

**Expected Output**:
```python
class EntityCluster(BaseModel):
    cluster_id: str
    canonical_entity: str  # Representative entity text
    entity_type: str
    entity_mentions: List[str]  # All variant mentions
    document_ids: List[str]
    centroid_vector: List[float]
    cluster_size: int
    metadata: Dict[str, Any]
```

#### Level 4: Storyline-Level Clustering
**Input**: `storylines.index` (2-5 embeddings per document)
**Goal**: Identify meta-narratives across multiple storylines
**Use Cases**:
- Higher-order thematic grouping
- Multi-storyline narrative arcs
- Cross-domain story connections

**Expected Output**:
```python
class StorylineCluster(BaseModel):
    cluster_id: str
    cluster_label: str
    storyline_ids: List[str]
    event_ids: List[str]  # All events across storylines
    primary_entities: List[str]
    domain: str
    temporal_span: Optional[List[str]]
    centroid_vector: List[float]
    metadata: Dict[str, Any]
```

### 3.2 Clustering Algorithms & Techniques

**Recommended Approaches:**

1. **HDBSCAN** (Hierarchical Density-Based Clustering)
   - **Why**: No need to pre-specify cluster count, handles noise, variable density
   - **Library**: `hdbscan` (GPU-accelerated if available)
   - **Tuning**: `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`

2. **FAISS-GPU K-Means** (for large-scale clustering)
   - **Why**: Extremely fast on GPU, scalable to millions of vectors
   - **Library**: `faiss-gpu`
   - **Tuning**: Number of clusters (k), iterations, initialization

3. **Agglomerative Clustering** (for hierarchical relationships)
   - **Why**: Produces dendrogram, interpretable hierarchy
   - **Library**: `scipy.cluster.hierarchy`, `scikit-learn`
   - **Tuning**: Linkage method (ward, average, complete), distance threshold

4. **Two-Stage Approach** (Recommended for Scale)
   - Stage 4a: Pre-cluster with FAISS K-Means (fast, approximate)
   - Stage 4b: Refine with HDBSCAN on pre-clusters (quality)

**Similarity Metrics:**
- **Primary**: Cosine similarity (vectors are L2-normalized)
- **Alternative**: Euclidean distance (equivalent for normalized vectors)

### 3.3 Temporal Clustering

**Critical Requirement**: Leverage temporal metadata for time-aware clustering.

**Strategies**:
1. **Temporal Decay Weighting**: Recent events weighted higher
2. **Temporal Windows**: Cluster only within time windows (e.g., ±7 days)
3. **Temporal Sorting**: Post-cluster sorting by `temporal_reference`
4. **Temporal Coherence**: Penalize clusters with large temporal gaps

**Implementation**:
```python
# Adjust similarity score with temporal proximity
temporal_weight = exp(-abs(date1 - date2) / decay_factor)
adjusted_similarity = semantic_similarity * temporal_weight
```

### 3.4 Metadata-Aware Clustering

**Leverage Rich Metadata from Stage 3:**

**Domain Filtering** (12 domains):
- `diplomatic_relations`
- `military_operations`
- `economic_activity`
- `political_events`
- `legal_judicial`
- `health_medical`
- `environmental`
- `technology_science`
- `cultural_social`
- `sports_entertainment`
- `infrastructure_development`
- `general_news`

**Event Type Filtering** (20+ types):
- ACE 2005: `contact_meet`, `conflict_attack`, `movement_transport`, `transaction_transfer`, etc.
- Custom: `policy_change`, `economic_indicator`, `natural_disaster`, etc.

**Strategy**: Perform separate clusterings per domain/event_type, then merge.

### 3.5 Output Format for Stage 5 (Graph Construction)

**Primary Output**: Clustered data with relationships

**File Format**: JSONL (one ClusterResult per line)

```json
{
  "cluster_id": "event_cluster_001",
  "cluster_type": "event",
  "cluster_label": "G7 Summit Diplomatic Meetings",
  "cluster_size": 45,
  "member_ids": ["doc_001_event_0", "doc_002_event_3", ...],
  "document_ids": ["doc_001", "doc_002", ...],
  "centroid_vector": [0.1, 0.2, ..., 0.3],  // 768D
  "temporal_span": ["2025-12-10T00:00:00Z", "2025-12-15T00:00:00Z"],
  "primary_entities": ["G7", "Japan", "United States"],
  "domain": "diplomatic_relations",
  "event_type": "contact_meet",
  "intra_cluster_similarity": 0.87,  // Avg pairwise similarity
  "metadata": {
    "algorithm": "hdbscan",
    "min_cluster_size": 5,
    "timestamp": "2025-12-24T10:00:00Z"
  }
}
```

**Storage Options**:
- **JSONL**: `data/clustered_events_YYYY-MM-DD.jsonl`
- **PostgreSQL**: `stage4_clustering` database
- **Redis Cache**: Cluster metadata for fast lookups

### 3.6 API Endpoints (Your Service)

**Design a RESTful API for downstream stages:**

```
POST /api/v1/cluster/batch        - Submit batch clustering job
GET  /api/v1/cluster/status/{id}  - Check job status
GET  /api/v1/clusters              - List all clusters (paginated)
GET  /api/v1/clusters/{id}         - Get cluster details
POST /api/v1/clusters/search       - Search clusters by metadata
GET  /health                       - Health check
GET  /statistics                   - Clustering statistics
```

### 3.7 Performance Requirements

**Throughput Targets:**
- Process 1,000 documents worth of embeddings in <30 minutes
- Event clustering: 5,000 events in <10 minutes
- Real-time cluster lookup: <100ms (cached)

**Scalability:**
- Support up to 1M vectors per index
- Horizontal scaling with multiple workers
- Incremental clustering (update clusters as new data arrives)

---

## 4. ULTIMATE GOAL ALIGNMENT

### 4.1 End-to-End Pipeline Goal

**Transform raw news into interactive temporal narratives** by:
1. Cleaning articles (Stage 1)
2. Extracting structured information (Stage 2)
3. Generating semantic embeddings (Stage 3)
4. **Clustering related content** (Stage 4 ← YOU)
5. Building knowledge graph (Stage 5)
6. Creating timelines (Stage 6)
7. Serving via API (Stage 7)
8. Displaying in UI (Stage 8)

### 4.2 Stage 4's Critical Role

**You enable downstream stages to:**
- **Stage 5 (Graph)**: Connect clustered entities/events as graph nodes
- **Stage 6 (Timeline)**: Summarize cluster-level narratives
- **Stage 7 (API)**: Query by cluster themes
- **Stage 8 (Frontend)**: Display grouped stories

**Key Value Propositions:**
1. **Reduce Information Overload**: 1,000 articles → 20-50 thematic clusters
2. **Enable Multi-Document Understanding**: Track events across sources
3. **Power Entity Disambiguation**: Resolve entity mentions globally
4. **Create Narrative Threads**: Identify evolving storylines

### 4.3 Quality Metrics

**Your clustering quality determines:**
- **Precision**: Are cluster members truly related? (Target: >85%)
- **Recall**: Are related items grouped together? (Target: >80%)
- **Purity**: Single-topic clusters (Target: >90%)
- **Coverage**: % of items successfully clustered (Target: >95%)

**Downstream Impact:**
- Poor clustering → Incorrect graph connections (Stage 5)
- Good clustering → Coherent timelines (Stage 6)

---

## 5. INDUSTRIAL STANDARDS ENFORCEMENT (December 2025)

### 5.1 Security (OWASP Top 10 Compliance)

**Mandatory Requirements:**

1. **Input Validation**:
   - Validate all API inputs with Pydantic models
   - Sanitize file paths (prevent directory traversal)
   - Limit request sizes (prevent DoS)

2. **Authentication & Authorization**:
   - JWT-based API authentication (optional for MVP, required for production)
   - Role-based access control (RBAC)
   - API rate limiting (per IP/user)

3. **Data Protection**:
   - No sensitive data in logs
   - Encrypted connections (TLS/SSL for external APIs)
   - Secure storage of credentials (HashiCorp Vault or Docker Secrets)

4. **Dependency Security**:
   - Pin exact versions in `requirements.txt`
   - Regular security scans: `pip-audit`, `safety check`
   - Update dependencies monthly

5. **Error Handling**:
   - Never expose stack traces to users
   - Generic error messages externally, detailed logs internally
   - Structured error responses

**Prohibited Practices**:
- ❌ Hardcoded credentials
- ❌ SQL injection vulnerabilities
- ❌ Unvalidated redirects
- ❌ Insecure deserialization

### 5.2 Scalability & Containerization

**Docker Compose v2 Syntax** (Mandatory):
```yaml
# ✅ Correct (v2 syntax)
docker compose up -d

# ❌ Wrong (v1 syntax - deprecated)
docker-compose up -d
```

**Container Best Practices**:
1. **Multi-Stage Builds**: Reduce image size
2. **Non-Root User**: Run as `USER 1000:1000`
3. **Health Checks**: Implement `/health` endpoint
4. **Resource Limits**: Define CPU/RAM limits
5. **Logging**: Structured JSON logs to stdout/stderr

**Docker Compose Structure**:
```yaml
services:
  orchestrator:
    build: .
    container_name: clustering-orchestrator
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 5.3 Code Quality (Python 3.11+)

**Mandatory Tools**:

1. **Formatting**: `black` (line length: 100)
   ```bash
   black src/ tests/ --line-length 100
   ```

2. **Linting**: `flake8` + `pylint`
   ```bash
   flake8 src/ tests/ --max-line-length=100
   pylint src/
   ```

3. **Type Checking**: `mypy` (strict mode)
   ```bash
   mypy src/ --strict
   ```

4. **Import Sorting**: `isort`
   ```bash
   isort src/ tests/ --profile black
   ```

**Code Architecture**:
- **SOLID Principles**: Single responsibility, dependency injection
- **DRY**: No code duplication
- **Clean Code**: Self-documenting, minimal comments
- **Modular**: Clear separation of concerns

**Directory Structure** (Recommended):
```
src/
├── api/                  # FastAPI orchestrator, routes
├── core/                 # Clustering algorithms
├── schemas/              # Pydantic models
├── storage/              # FAISS/index readers
├── utils/                # Logging, monitoring
└── config/               # YAML settings loader
```

### 5.4 Testing (Comprehensive Coverage)

**Test Pyramid**:
1. **Unit Tests** (70%): Test individual functions
2. **Integration Tests** (25%): Test service interactions
3. **End-to-End Tests** (5%): Full pipeline tests

**Required Tools**:
- `pytest` (test framework)
- `pytest-cov` (coverage reporting, target: >80%)
- `pytest-asyncio` (async tests)
- `pytest-docker` (test containers)

**Test Structure**:
```
tests/
├── unit/
│   ├── test_clustering_algorithms.py
│   ├── test_faiss_loader.py
│   └── test_metadata_parser.py
├── integration/
│   ├── test_api_endpoints.py
│   └── test_clustering_pipeline.py
└── e2e/
    └── test_full_workflow.py
```

**CI/CD Integration**:
- Run tests on every commit (GitHub Actions, GitLab CI)
- Fail build if coverage <80%
- Automated deployment on passing tests

### 5.5 Documentation

**Mandatory Files**:
1. **README.md**: Quick start, architecture overview
2. **CLAUDE.md**: Stage-specific context for AI assistants
3. **API.md**: Endpoint documentation (or OpenAPI/Swagger)
4. **ARCHITECTURE.md**: Design decisions, data flow
5. **DEPLOYMENT.md**: Production deployment guide

**Inline Documentation**:
- Docstrings for all public functions (Google or NumPy style)
- Type hints for all function signatures
- Complex algorithms: Explain the "why", not the "what"

**Example**:
```python
def cluster_events(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    min_cluster_size: int = 5
) -> List[EventCluster]:
    """
    Cluster event embeddings using HDBSCAN algorithm.

    Args:
        embeddings: Normalized 768D vectors (N x 768)
        metadata: Event metadata for filtering
        min_cluster_size: Minimum points per cluster

    Returns:
        List of EventCluster objects with member assignments

    Raises:
        ValueError: If embeddings are not normalized
    """
    ...
```

### 5.6 Observability & Monitoring

**Structured Logging** (JSON format):
```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "clustering_completed",
    cluster_count=25,
    total_vectors=5000,
    duration_ms=3200,
    algorithm="hdbscan",
    stage=4
)
```

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram

clustering_jobs_total = Counter(
    'clustering_jobs_total',
    'Total clustering jobs processed',
    ['stage', 'algorithm', 'status']
)

clustering_duration = Histogram(
    'clustering_duration_seconds',
    'Time to cluster',
    ['stage', 'embedding_type']
)
```

**Health Checks**:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "stage4-clustering",
        "version": "1.0.0",
        "dependencies": {
            "stage3_embedding": await check_stage3_health(),
            "redis": await redis_client.ping(),
            "postgres": await db.execute("SELECT 1")
        }
    }
```

### 5.7 Data Privacy & Compliance

**GDPR/CCPA Considerations** (if applicable):
- Implement data retention policies (e.g., 90-day TTL)
- Support data deletion requests (delete clusters by document_id)
- Anonymize PII in logs
- Document data processing purposes

### 5.8 Performance Best Practices

1. **GPU Optimization**:
   - Use FAISS-GPU for similarity searches
   - Batch operations to maximize GPU utilization
   - Monitor GPU memory (prevent OOM)

2. **CPU Optimization**:
   - Parallel processing with `multiprocessing` (up to 22 workers)
   - NumPy vectorization (avoid loops)
   - Cache expensive computations (Redis)

3. **Memory Management**:
   - Stream large files (avoid loading all in memory)
   - Use memory-mapped files for FAISS indices
   - Garbage collection after batch jobs

4. **I/O Optimization**:
   - Asynchronous file I/O (`aiofiles`)
   - Batch database writes (bulk inserts)
   - Connection pooling (Redis, PostgreSQL)

---

## 6. INSTRUCTIONS FOR STAGE 4 TEAM

### 6.1 Quick Start (Clean Setup)

**Step 1: Clone and Setup**
```bash
# Create stage 4 repository
mkdir stage4-clustering-service
cd stage4-clustering-service

# Initialize git
git init

# Create directory structure
mkdir -p src/{api,core,schemas,storage,utils,config}
mkdir -p tests/{unit,integration,e2e}
mkdir -p data/{indices,output}
mkdir -p logs
mkdir -p .claude

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

**Step 2: Install Dependencies**
```bash
# Create requirements.txt
cat > requirements.txt <<EOF
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
celery[redis]==5.3.4

# Clustering
faiss-gpu==1.7.4  # Or faiss-cpu if no GPU
hdbscan==0.8.33
scikit-learn==1.3.2
scipy==1.11.4
numpy==1.26.2

# Data handling
pandas==2.1.4
orjson==3.9.10

# Storage
redis==5.0.1
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1

# Code quality
black==23.12.0
flake8==6.1.0
mypy==1.7.1
isort==5.13.2
EOF

pip install -r requirements.txt
```

**Step 3: Integrate with Infrastructure**

**CRITICAL**: Read these files BEFORE starting:
1. `../infrastructure/README.md`
2. `../infrastructure/.claude/rules/infrastructure-integration.md`
3. `../stage2-nlp-processing/docker-compose.infrastructure.yml` (reference implementation)

**Create `docker-compose.infrastructure.yml`**:
```yaml
version: '3.9'

networks:
  storytelling:
    external: true
    name: storytelling

services:
  orchestrator-service:
    container_name: clustering-orchestrator
    build:
      context: .
      dockerfile: Dockerfile
    networks:
      - storytelling
    environment:
      # Infrastructure services
      - REDIS_HOST=redis-broker
      - REDIS_PORT=6379
      - REDIS_DB=6  # Stage 4: (4-1)*2 = 6
      - REDIS_CACHE_HOST=redis-cache
      - REDIS_CACHE_PORT=6379
      - REDIS_CACHE_DB=7  # Stage 4: (4-1)*2+1 = 7

      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=stage4_clustering
      - POSTGRES_USER=stage4_user
      - POSTGRES_PASSWORD=${STAGE4_POSTGRES_PASSWORD}

      # Stage 3 dependency (upstream)
      - STAGE3_API_URL=http://embeddings-orchestrator:8000
      - STAGE3_INDICES_PATH=/shared/stage3/data/vector_indices

      # Observability
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
      - OTEL_SERVICE_NAME=stage4-clustering-orchestrator
      - LOG_LEVEL=INFO

    volumes:
      - ../stage3_embedding_service/data/vector_indices:/shared/stage3/data/vector_indices:ro
      - ./data:/app/data
      - ./logs:/app/logs

    deploy:
      resources:
        limits:
          cpus: '15'
          memory: 40G
        reservations:
          cpus: '8'
          memory: 20G

    labels:
      - "stage=4"
      - "service=orchestrator"
      - "com.docker.compose.project=stage4-clustering"

    # NO ports: directive - Traefik handles routing
    # Access via: http://localhost/api/v1/clustering/*

  celery-worker:
    container_name: clustering-celery-worker
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A src.api.celery_worker:celery_app worker --loglevel=info --concurrency=22
    networks:
      - storytelling
    environment:
      - CELERY_BROKER_URL=redis://redis-broker:6379/6
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/7
      # (same environment variables as orchestrator)
    volumes:
      - ../stage3_embedding_service/data/vector_indices:/shared/stage3/data/vector_indices:ro
      - ./data:/app/data
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '22'
          memory: 100G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Register Traefik Route** (in `../infrastructure/traefik/dynamic.yml`):
```yaml
http:
  routers:
    stage4-clustering:
      rule: "PathPrefix(`/api/v1/clustering`)"
      service: stage4-clustering
      middlewares:
        - rate-limit
        - request-id
        - strip-clustering
      priority: 100

  services:
    stage4-clustering:
      loadBalancer:
        servers:
          - url: "http://clustering-orchestrator:8000"
        healthCheck:
          path: /health
          interval: 10s
          timeout: 5s

  middlewares:
    strip-clustering:
      stripPrefix:
        prefixes:
          - "/api/v1/clustering"
```

**Step 4: Load FAISS Indices from Stage 3**

**Example Code** (`src/storage/faiss_loader.py`):
```python
import faiss
import json
from pathlib import Path
from typing import Dict, Any

class FAISSIndexLoader:
    """Load FAISS indices and metadata from Stage 3."""

    def __init__(self, indices_path: str = "/shared/stage3/data/vector_indices"):
        self.indices_path = Path(indices_path)

    def load_index(self, index_type: str) -> tuple[faiss.Index, Dict[str, Any]]:
        """
        Load FAISS index and metadata.

        Args:
            index_type: "documents", "events", "entities", or "storylines"

        Returns:
            (faiss_index, metadata_dict)
        """
        index_file = self.indices_path / f"{index_type}.index"
        metadata_file = self.indices_path / f"{index_type}_metadata.json"

        # Load FAISS index
        index = faiss.read_index(str(index_file))

        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        return index, metadata

    def search_similar(
        self,
        index: faiss.Index,
        metadata: Dict[str, Any],
        query_vector: np.ndarray,
        k: int = 100,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for top-k similar vectors with metadata filtering."""
        # Search FAISS index
        D, I = index.search(query_vector.reshape(1, -1), k * 10)  # Over-fetch for filtering

        results = []
        metadata_store = metadata["metadata_store"]
        reverse_id_map = metadata["reverse_id_map"]

        for dist, idx in zip(D[0], I[0]):
            vector_id = reverse_id_map[str(idx)]
            meta = metadata_store[vector_id]

            # Apply metadata filters
            if filter_metadata:
                if not all(meta["metadata"].get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append({
                "vector_id": vector_id,
                "source_id": meta["source_id"],
                "similarity_score": float(1 / (1 + dist)),  # Convert distance to similarity
                "metadata": meta["metadata"]
            })

            if len(results) >= k:
                break

        return results
```

### 6.2 Development Workflow

**Daily Development**:
1. Pull latest from Stage 3 (indices may update)
2. Run tests: `pytest tests/ -v`
3. Format code: `black src/ tests/`
4. Lint: `flake8 src/ tests/`
5. Type check: `mypy src/`
6. Commit with conventional commits: `feat: add HDBSCAN clustering`

**Testing Against Real Data**:
```bash
# Start infrastructure
cd ../infrastructure
./scripts/start.sh

# Start Stage 3 (upstream dependency)
cd ../stage3_embedding_service
./run-with-infrastructure.sh start

# Start Stage 4 (your service)
cd ../stage4-clustering-service
docker compose -f docker-compose.infrastructure.yml up -d

# Test clustering
curl -X POST http://localhost/api/v1/clustering/batch \
  -H "Content-Type: application/json" \
  -d '{"embedding_type": "events", "algorithm": "hdbscan"}'
```

### 6.3 Incremental Learning & Error Avoidance

**Learn from Stage 2 & 3 Mistakes**:

1. **Memory Management**: Stage 2 had GPU OOM issues
   - ✅ Implement proactive memory monitoring
   - ✅ Batch operations to stay under limits
   - ✅ Clear cache after each batch

2. **Checkpoint Mechanism**: Stage 3 added progressive persistence
   - ✅ Save clustering results incrementally
   - ✅ Enable pause/resume for long jobs
   - ✅ Track processed items in Redis

3. **Metadata Preservation**: Critical for downstream stages
   - ✅ Pass through all Stage 3 metadata
   - ✅ Add clustering metadata (cluster_id, similarity scores)
   - ✅ Validate schemas with Pydantic

4. **Resource Cleanup**: Prevent leaks
   - ✅ Explicit model unloading after jobs
   - ✅ Connection pool cleanup
   - ✅ Periodic garbage collection

### 6.4 Senior Engineering Mindset

**Treat This as Production Code:**
- Every PR needs tests
- No direct commits to main branch
- Code review required (even if solo, review your own code critically)
- Document all design decisions
- Optimize for maintainability over cleverness

**Ask Questions Early:**
- If Stage 3 contract is unclear, ask (via GitHub issues)
- If hardware limits are hit, document and propose solutions
- If clustering quality is poor, iterate on algorithms

**Deliver Incrementally:**
1. **Week 1**: Basic FAISS loading, simple K-Means clustering
2. **Week 2**: HDBSCAN implementation, metadata filtering
3. **Week 3**: Temporal clustering, API endpoints
4. **Week 4**: Batch processing, job lifecycle
5. **Week 5**: Testing, optimization, documentation

### 6.5 Success Criteria

**Minimum Viable Product (MVP)**:
- [ ] Load all 4 FAISS indices from Stage 3
- [ ] Implement event-level clustering (HDBSCAN or K-Means)
- [ ] Output JSONL with cluster assignments
- [ ] Basic API: submit job, check status, get results
- [ ] Health check endpoint
- [ ] Integration with infrastructure (Traefik, Redis, Postgres)
- [ ] Tests with >60% coverage

**Production-Ready**:
- [ ] All 4 clustering levels (document, event, entity, storyline)
- [ ] Temporal clustering with decay weighting
- [ ] Metadata-aware clustering (domain, event_type filters)
- [ ] Batch processing with Celery
- [ ] Pause/resume/cancel jobs
- [ ] Prometheus metrics
- [ ] Structured logging
- [ ] Tests with >80% coverage
- [ ] Full documentation

**Excellence**:
- [ ] Incremental clustering (update clusters with new data)
- [ ] Cluster quality metrics (silhouette score, purity)
- [ ] Automatic hyperparameter tuning
- [ ] WebSocket real-time progress
- [ ] Cluster visualization API
- [ ] Benchmark results documented

---

## 7. ADDITIONAL CONTEXT

### 7.1 Stage 2 → Stage 3 → Stage 4 Data Flow

```
Raw Articles
    ↓
[Stage 1: Cleaning]
    ↓
Cleaned Documents (text, title, date, etc.)
    ↓
[Stage 2: NLP Processing]
    ↓
ProcessedDocuments (events, entities, relationships)
    ↓
[Stage 3: Embedding Generation]
    ↓
768D Vector Embeddings (4 indices: docs, events, entities, storylines)
    ↓
[Stage 4: Clustering] ← YOU ARE HERE
    ↓
Clustered Entities/Events (grouped by semantic similarity)
    ↓
[Stage 5: Graph Construction]
    ↓
Neo4j Knowledge Graph
    ↓
[Stage 6: Timeline Generation]
    ↓
Temporal Narratives
    ↓
[Stage 7: API Service]
    ↓
[Stage 8: Frontend]
    ↓
User-Facing Interactive Timelines
```

### 7.2 Key Assumptions

1. **Stage 3 is Running**: Indices are up-to-date and accessible
2. **Shared Volume**: You can read Stage 3's `data/vector_indices/` directory
3. **Infrastructure Active**: Redis, Postgres, Traefik are running
4. **GPU Access**: RTX A4000 available for FAISS-GPU operations
5. **Network Connectivity**: Docker DNS resolves `embeddings-orchestrator`, `redis-broker`, etc.

### 7.3 Troubleshooting Common Issues

**Issue**: Cannot load FAISS indices
**Solution**: Check volume mount in `docker-compose.infrastructure.yml`, verify Stage 3 has created indices

**Issue**: Out of GPU memory
**Solution**: Reduce batch size, use CPU fallback for FAISS, clear GPU cache between operations

**Issue**: Clustering quality poor
**Solution**: Tune hyperparameters (min_cluster_size), apply metadata filters, try different algorithms

**Issue**: Slow clustering performance
**Solution**: Use FAISS-GPU, parallelize with multiprocessing, pre-cluster with K-Means

### 7.4 References

**Essential Documentation**:
- FAISS: https://github.com/facebookresearch/faiss/wiki
- HDBSCAN: https://hdbscan.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- Celery: https://docs.celeryproject.org/

**Example Repositories**:
- Stage 2 (NLP): `../stage2-nlp-processing/` (reference for infrastructure integration)
- Stage 3 (Embeddings): `../stage3_embedding_service/` (your upstream dependency)

**Hardware Docs**:
- RTX A4000: https://www.nvidia.com/en-us/design-visualization/rtx-a4000/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

---

## 8. FINAL NOTES

**Remember**:
- You are a **black box** service: Well-defined inputs from Stage 3, well-defined outputs to Stage 5
- **Contracts are sacred**: Don't break Stage 3's output expectations or Stage 5's input expectations
- **Test obsessively**: Your clustering quality affects 4 downstream stages
- **Document everything**: Future you (and your team) will thank you
- **Ask for help**: Stage 2 and Stage 3 teams have solved similar problems

**This is production code for a real pipeline.** Quality, security, and performance matter. Build it right the first time.

**Good luck! 🚀**

---

**Generated by**: Claude Code v2.0.69
**Date**: December 24, 2025
**Source**: Stage 3 Embedding Service Codebase Analysis
**Target**: Stage 4 Clustering Service Development Team
