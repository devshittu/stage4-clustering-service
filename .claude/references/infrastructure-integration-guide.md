# Infrastructure Integration Reference Guide

This comprehensive guide shows how to integrate any stage (1-8) with the centralized Storytelling Platform infrastructure.

## Table of Contents

1. [Quick Integration Checklist](#quick-integration-checklist)
2. [Step-by-Step Integration](#step-by-step-integration)
3. [Service-Specific Examples](#service-specific-examples)
4. [Environment Variables Reference](#environment-variables-reference)
5. [Troubleshooting Integration Issues](#troubleshooting-integration-issues)
6. [Testing Your Integration](#testing-your-integration)
7. [Advanced Patterns](#advanced-patterns)

---

## Quick Integration Checklist

Before you start developing a new stage, verify:

- [ ] Infrastructure is running: `cd /path/to/infrastructure && docker compose ps`
- [ ] You know your stage number (1-8)
- [ ] You know your stage name (e.g., "cleaning", "nlp", "embeddings")
- [ ] You've read `/infrastructure/README.md`
- [ ] You have Stage 2 as reference: `/stage2-nlp-processing/`

---

## Step-by-Step Integration

### Step 1: Create Your Stage Repository

```bash
# Navigate to project root
cd /home/mshittu/projects/nlp

# Create stage directory
mkdir stage{N}-{service-name}
cd stage{N}-{service-name}

# Initialize git (if separate repo)
git init
```

**Directory Structure**:
```
stage{N}-{service-name}/
├── .claude/
│   └── CLAUDE.md                       # Stage-specific documentation
├── docker-compose.yml                  # Standalone mode
├── docker-compose.infrastructure.yml   # Infrastructure integration ← KEY FILE
├── .env.example                        # Environment template
├── .gitignore
├── run-with-infrastructure.sh          # Helper script
├── Dockerfile
├── requirements.txt
└── src/
    └── ...
```

---

### Step 2: Create docker-compose.infrastructure.yml

This is the **most important file** for infrastructure integration.

```yaml
version: '3.9'

# =============================================================================
# STAGE {N}: {SERVICE_NAME} - INFRASTRUCTURE INTEGRATION
# =============================================================================
# Connects this stage to centralized infrastructure
# =============================================================================

networks:
  storytelling:
    external: true
    name: storytelling

services:
  # ===========================================================================
  # ORCHESTRATOR SERVICE
  # ===========================================================================
  orchestrator-service:
    container_name: {stage-name}-orchestrator
    networks:
      - storytelling
    environment:
      # -------------------------------------------------------------------------
      # INFRASTRUCTURE SERVICES
      # -------------------------------------------------------------------------
      # Redis Broker (Celery tasks)
      - REDIS_HOST=redis-broker
      - REDIS_PORT=6379
      - REDIS_DB={N*2-2}  # e.g., Stage 1=0, Stage 2=2, Stage 3=4

      # Redis Cache (Application data)
      - REDIS_CACHE_HOST=redis-cache
      - REDIS_CACHE_PORT=6379
      - REDIS_CACHE_DB={N*2-1}  # e.g., Stage 1=1, Stage 2=3, Stage 3=5

      # PostgreSQL
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=stage{N}_{service_name}
      - POSTGRES_USER=stage{N}_user
      - POSTGRES_PASSWORD=${STAGE{N}_POSTGRES_PASSWORD:-stage{N}_secure_password}

      # Elasticsearch (optional)
      - ELASTICSEARCH_URL=http://elasticsearch:9200

      # Neo4j (optional)
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-changeme_neo4j_password}

      # -------------------------------------------------------------------------
      # CELERY CONFIGURATION
      # -------------------------------------------------------------------------
      - CELERY_BROKER_URL=redis://redis-broker:6379/{N*2-2}
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/{N*2-1}

      # -------------------------------------------------------------------------
      # OBSERVABILITY (OpenTelemetry)
      # -------------------------------------------------------------------------
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
      - OTEL_SERVICE_NAME=stage{N}-{service-name}-orchestrator

      # -------------------------------------------------------------------------
      # STAGE-SPECIFIC SERVICES (if any)
      # -------------------------------------------------------------------------
      # Example: If your stage has internal microservices
      # - SERVICE_A_URL=http://{stage-name}-service-a:8001

    labels:
      # Required for Prometheus service discovery
      - "stage={N}"
      - "service=orchestrator"
      - "com.docker.compose.project=stage{N}-{service-name}"

    # CRITICAL: Remove port mappings - Traefik handles routing
    ports: []

    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s

  # ===========================================================================
  # CELERY WORKER (if applicable)
  # ===========================================================================
  celery-worker:
    networks:
      - storytelling
    environment:
      # Celery configuration
      - CELERY_BROKER_URL=redis://redis-broker:6379/{N*2-2}
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/{N*2-1}

      # Infrastructure services (same as orchestrator)
      - REDIS_HOST=redis-broker
      - REDIS_CACHE_HOST=redis-cache
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=stage{N}_{service_name}
      - POSTGRES_USER=stage{N}_user
      - POSTGRES_PASSWORD=${STAGE{N}_POSTGRES_PASSWORD}
      - ELASTICSEARCH_URL=http://elasticsearch:9200

      # Observability
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
      - OTEL_SERVICE_NAME=stage{N}-{service-name}-celery-worker

    labels:
      - "stage={N}"
      - "service=celery-worker"

    # Resource limits (REQUIRED)
    deploy:
      resources:
        limits:
          cpus: '{cpu_count}'  # Adjust based on workload
          memory: {memory}G
        reservations:
          cpus: '{min_cpu_count}'
          memory: {min_memory}G

  # ===========================================================================
  # DISABLE LOCAL INFRASTRUCTURE SERVICES
  # ===========================================================================
  # If your docker-compose.yml has local Redis/Postgres, disable them
  redis:
    profiles:
      - disabled  # Only runs when explicitly specified

  postgres:
    profiles:
      - disabled
```

**Configuration Values**:

| Placeholder | Example (Stage 3) | Description |
|-------------|-------------------|-------------|
| `{N}` | 3 | Stage number |
| `{service-name}` | embeddings | Stage name |
| `{stage-name}` | embeddings | Container prefix |
| `{N*2-2}` | 4 | Redis Celery DB (formula) |
| `{N*2-1}` | 5 | Redis Cache DB (formula) |
| `{cpu_count}` | 4 | CPU cores to allocate |
| `{memory}` | 8 | RAM in GB |

---

### Step 3: Create Helper Script

```bash
# stage{N}-{service-name}/run-with-infrastructure.sh
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")/infrastructure"

ACTION="${1:-help}"

case "$ACTION" in
    start)
        echo "🚀 Starting Stage {N} ({service-name}) with Infrastructure..."

        # Check infrastructure
        if ! docker network ls | grep -q storytelling; then
            echo "⚠️  Infrastructure not running. Starting..."
            cd "$INFRA_DIR"
            ./scripts/start.sh
            sleep 10
        fi

        # Start stage
        cd "$SCRIPT_DIR"
        docker compose -f docker-compose.yml -f docker-compose.infrastructure.yml up -d

        echo "✅ Stage {N} started!"
        echo "📊 API: http://localhost/api/v1/{service-name}/health"
        ;;

    stop)
        docker compose -f docker-compose.yml -f docker-compose.infrastructure.yml down
        ;;

    logs)
        docker compose -f docker-compose.yml -f docker-compose.infrastructure.yml logs -f "${2:-orchestrator-service}"
        ;;

    health)
        curl -s http://localhost/api/v1/{service-name}/health | jq '.'
        ;;

    *)
        echo "Usage: $0 {start|stop|logs|health}"
        ;;
esac
```

```bash
chmod +x run-with-infrastructure.sh
```

---

### Step 4: Register Traefik Route

Add your stage to `/infrastructure/traefik/dynamic.yml`:

```yaml
http:
  routers:
    stage{N}-{service-name}:
      rule: "PathPrefix(`/api/v1/{service-name}`)"
      service: stage{N}-{service-name}
      middlewares:
        - rate-limit
        - request-id
        - strip-stage{N}
      priority: 100

  services:
    stage{N}-{service-name}:
      loadBalancer:
        servers:
          - url: "http://{stage-name}-orchestrator:8000"
        healthCheck:
          path: /health
          interval: 10s
          timeout: 5s

  middlewares:
    strip-stage{N}:
      stripPrefix:
        prefixes:
          - "/api/v1/{service-name}"
```

**After editing**, restart Traefik:
```bash
cd /infrastructure
docker compose restart traefik
```

---

### Step 5: Update Prometheus Configuration

Add scrape target to `/infrastructure/observability/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'stage{N}-{service-name}'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        filters:
          - name: label
            values: ['stage={N}']
    relabel_configs:
      - source_labels: [__meta_docker_container_name]
        target_label: container_name
      - source_labels: [__meta_docker_container_label_service]
        target_label: service
```

**After editing**, reload Prometheus:
```bash
cd /infrastructure
docker compose exec prometheus kill -HUP 1
```

---

### Step 6: Create .env.example

```bash
# stage{N}-{service-name}/.env.example
# =============================================================================
# STAGE {N}: {SERVICE_NAME} - ENVIRONMENT VARIABLES
# =============================================================================
# Copy to .env and fill in actual values
# =============================================================================

# PostgreSQL (infrastructure)
STAGE{N}_POSTGRES_PASSWORD=stage{N}_secure_password

# Optional: Service-specific secrets
STAGE{N}_API_KEY=your_api_key_here
STAGE{N}_SECRET_KEY=your_secret_key_here

# HuggingFace (if using models)
HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE

# Optional: External API credentials
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

### Step 7: Implement Health Check Endpoint

```python
# src/api/orchestrator_service.py
from fastapi import FastAPI, HTTPException
from datetime import datetime
import redis
import asyncpg

app = FastAPI()

# Global connections (initialized on startup)
redis_client = None
db_pool = None

@app.on_event("startup")
async def startup():
    global redis_client, db_pool

    # Initialize Redis
    redis_client = redis.Redis(
        host=os.getenv("REDIS_CACHE_HOST", "redis-cache"),
        port=int(os.getenv("REDIS_CACHE_PORT", "6379")),
        db=int(os.getenv("REDIS_CACHE_DB", "0")),
        decode_responses=True
    )

    # Initialize PostgreSQL
    db_pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        min_size=5,
        max_size=20
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Traefik.

    Verifies:
    - Service is running
    - Redis connectivity
    - PostgreSQL connectivity
    - Critical dependencies

    Returns:
        200 OK: Service healthy
        503 Service Unavailable: Service unhealthy
    """
    health_status = {
        "status": "healthy",
        "service": "stage{N}-{service-name}",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    try:
        # Check Redis
        redis_client.ping()
        health_status["checks"]["redis"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["redis"] = f"failed: {str(e)}"

    try:
        # Check PostgreSQL
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health_status["checks"]["postgres"] = "ok"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["postgres"] = f"failed: {str(e)}"

    # Return 503 if unhealthy
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status
```

---

### Step 8: Configure Logging

```python
# src/utils/logger.py
import structlog
import logging
import os

def setup_logging():
    """
    Configure structured logging for Loki ingestion.
    """
    logging.basicConfig(
        format="%(message)s",
        level=os.getenv("LOG_LEVEL", "INFO"),
        handlers=[logging.StreamHandler()]
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # JSON for Loki
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()

# Usage
logger = setup_logging()

logger.info(
    "document_processed",
    stage="{N}",
    service="orchestrator",
    document_id="doc123",
    processing_time_ms=1234,
    status="success"
)
```

---

## Service-Specific Examples

### Example 1: Stage 1 (Cleaning & Ingestion)

**Stage Info**:
- Number: 1
- Name: cleaning
- Redis: DB 0 (Celery), DB 1 (Cache)
- Database: `stage1_cleaning`

**docker-compose.infrastructure.yml**:
```yaml
version: '3.9'

networks:
  storytelling:
    external: true

services:
  orchestrator-service:
    container_name: cleaning-orchestrator
    networks:
      - storytelling
    environment:
      - REDIS_DB=0
      - REDIS_CACHE_DB=1
      - POSTGRES_DB=stage1_cleaning
      - POSTGRES_USER=stage1_user
      - CELERY_BROKER_URL=redis://redis-broker:6379/0
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/1
      - OTEL_SERVICE_NAME=stage1-cleaning-orchestrator
    labels:
      - "stage=1"
      - "service=orchestrator"
    ports: []
```

**Traefik Route**:
```yaml
stage1-cleaning:
  rule: "PathPrefix(`/api/v1/cleaning`)"
  service: stage1-cleaning
```

---

### Example 2: Stage 3 (Embedding Generation)

**Stage Info**:
- Number: 3
- Name: embeddings
- Redis: DB 4 (Celery), DB 5 (Cache)
- Database: `stage3_embeddings`

**docker-compose.infrastructure.yml**:
```yaml
version: '3.9'

networks:
  storytelling:
    external: true

services:
  orchestrator-service:
    container_name: embeddings-orchestrator
    networks:
      - storytelling
    environment:
      - REDIS_DB=4
      - REDIS_CACHE_DB=5
      - POSTGRES_DB=stage3_embeddings
      - POSTGRES_USER=stage3_user
      - CELERY_BROKER_URL=redis://redis-broker:6379/4
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/5
      - OTEL_SERVICE_NAME=stage3-embeddings-orchestrator
    labels:
      - "stage=3"
      - "service=orchestrator"
    ports: []

  celery-worker:
    networks:
      - storytelling
    environment:
      - CELERY_BROKER_URL=redis://redis-broker:6379/4
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/5
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    labels:
      - "stage=3"
      - "service=celery-worker"
```

---

### Example 3: Stage 7 (Public API)

**Stage Info**:
- Number: 7
- Name: api
- Redis: DB 12 (Celery), DB 13 (Cache)
- Database: `stage7_api`
- Special: Public-facing, needs stricter rate limiting

**docker-compose.infrastructure.yml**:
```yaml
version: '3.9'

networks:
  storytelling:
    external: true

services:
  api-service:
    container_name: api-orchestrator
    networks:
      - storytelling
    environment:
      - REDIS_DB=12
      - REDIS_CACHE_DB=13
      - POSTGRES_DB=stage7_api
      - POSTGRES_USER=stage7_user
      - CELERY_BROKER_URL=redis://redis-broker:6379/12
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/13
      - OTEL_SERVICE_NAME=stage7-api
    labels:
      - "stage=7"
      - "service=api"
    ports: []
```

**Traefik Route** (with public rate limit):
```yaml
stage7-api:
  rule: "PathPrefix(`/api/v1/public`)"
  service: stage7-api
  middlewares:
    - rate-limit-public  # Stricter limit
    - request-id
    - strip-stage7
```

---

## Environment Variables Reference

### Required Infrastructure Variables

```bash
# Redis Broker (Celery)
REDIS_HOST=redis-broker
REDIS_PORT=6379
REDIS_DB={N*2-2}  # Formula: (Stage_Number - 1) * 2

# Redis Cache
REDIS_CACHE_HOST=redis-cache
REDIS_CACHE_PORT=6379
REDIS_CACHE_DB={N*2-1}  # Formula: (Stage_Number - 1) * 2 + 1

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=stage{N}_{service_name}
POSTGRES_USER=stage{N}_user
POSTGRES_PASSWORD=${STAGE{N}_POSTGRES_PASSWORD}

# Celery
CELERY_BROKER_URL=redis://redis-broker:6379/{N*2-2}
CELERY_RESULT_BACKEND=redis://redis-cache:6379/{N*2-1}

# Elasticsearch (optional)
ELASTICSEARCH_URL=http://elasticsearch:9200

# Neo4j (optional)
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=${NEO4J_PASSWORD}

# Observability (OpenTelemetry)
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
OTEL_SERVICE_NAME=stage{N}-{service-name}-{component}
```

### Redis DB Allocation Table

| Stage | Service | Celery DB | Cache DB | Formula |
|-------|---------|-----------|----------|---------|
| 1 | Cleaning | 0 | 1 | (1-1)*2=0, (1-1)*2+1=1 |
| 2 | NLP | 2 | 3 | (2-1)*2=2, (2-1)*2+1=3 |
| 3 | Embeddings | 4 | 5 | (3-1)*2=4, (3-1)*2+1=5 |
| 4 | Clustering | 6 | 7 | (4-1)*2=6, (4-1)*2+1=7 |
| 5 | Graph | 8 | 9 | (5-1)*2=8, (5-1)*2+1=9 |
| 6 | Timeline | 10 | 11 | (6-1)*2=10, (6-1)*2+1=11 |
| 7 | API | 12 | 13 | (7-1)*2=12, (7-1)*2+1=13 |
| 8 | Frontend | 14 | 15 | (8-1)*2=14, (8-1)*2+1=15 |

---

## Troubleshooting Integration Issues

### Issue 1: Service Can't Connect to Infrastructure

**Symptoms**:
```
ConnectionError: Cannot connect to redis://localhost:6379
```

**Causes**:
1. Using `localhost` instead of service names
2. Not on `storytelling` network
3. Infrastructure not running

**Solution**:
```yaml
# ❌ Wrong
environment:
  - REDIS_HOST=localhost

# ✅ Correct
environment:
  - REDIS_HOST=redis-broker

networks:
  storytelling:
    external: true  # Must be external!
```

---

### Issue 2: Traefik Returns 404

**Symptoms**:
```bash
curl http://localhost/api/v1/myservice/health
# 404 Not Found
```

**Causes**:
1. Route not registered in `traefik/dynamic.yml`
2. Container name doesn't match Traefik config
3. Service unhealthy

**Solution**:
```bash
# Check Traefik logs
docker logs storytelling-traefik

# Verify route exists
docker exec storytelling-traefik cat /etc/traefik/dynamic/dynamic.yml | grep myservice

# Check container name
docker ps | grep myservice-orchestrator
```

---

### Issue 3: Health Check Fails

**Symptoms**:
Traefik marks service as unhealthy, no traffic routed.

**Causes**:
1. `/health` endpoint not implemented
2. Dependencies (Redis, Postgres) unreachable
3. Service crashed during startup

**Solution**:
```bash
# Check service logs
docker logs {stage-name}-orchestrator

# Test health endpoint directly
docker exec {stage-name}-orchestrator curl -f http://localhost:8000/health

# Verify dependencies
docker exec {stage-name}-orchestrator nc -zv redis-broker 6379
docker exec {stage-name}-orchestrator nc -zv postgres 5432
```

---

### Issue 4: Resource Exhaustion

**Symptoms**:
```
OOMKilled: Container exceeded memory limit
```

**Causes**:
1. No resource limits set
2. Limits too high for available resources
3. Memory leak

**Solution**:
```yaml
# Add resource limits
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

---

## Testing Your Integration

### Test 1: Network Connectivity

```bash
# From your stage container
docker exec {stage-name}-orchestrator sh -c "
  nc -zv redis-broker 6379 &&
  nc -zv redis-cache 6379 &&
  nc -zv postgres 5432 &&
  nc -zv elasticsearch 9200 &&
  nc -zv neo4j 7687 &&
  echo 'All infrastructure services reachable!'
"
```

---

### Test 2: Redis Connection

```python
# test_redis.py
import redis
import os

def test_redis():
    # Broker
    broker = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        db=int(os.getenv("REDIS_DB"))
    )
    assert broker.ping() == True

    # Cache
    cache = redis.Redis(
        host=os.getenv("REDIS_CACHE_HOST"),
        port=int(os.getenv("REDIS_CACHE_PORT")),
        db=int(os.getenv("REDIS_CACHE_DB"))
    )
    assert cache.ping() == True

    # Test set/get
    cache.set("test_key", "test_value")
    assert cache.get("test_key") == b"test_value"

    print("✅ Redis connection successful!")

if __name__ == "__main__":
    test_redis()
```

---

### Test 3: PostgreSQL Connection

```python
# test_postgres.py
import asyncpg
import os
import asyncio

async def test_postgres():
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

    # Test query
    result = await conn.fetchval("SELECT 1")
    assert result == 1

    # List tables
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname='public'"
    )
    print(f"Tables: {[t['tablename'] for t in tables]}")

    await conn.close()
    print("✅ PostgreSQL connection successful!")

if __name__ == "__main__":
    asyncio.run(test_postgres())
```

---

### Test 4: Traefik Routing

```bash
# Test health endpoint via Traefik
curl -v http://localhost/api/v1/{service-name}/health

# Expected response:
# HTTP/1.1 200 OK
# X-Request-Id: <auto-generated>
# {
#   "status": "healthy",
#   "service": "stage{N}-{service-name}",
#   ...
# }

# Test rate limiting
for i in {1..200}; do
  curl -s http://localhost/api/v1/{service-name}/health > /dev/null
done
# Should see 429 Too Many Requests after ~100 requests
```

---

### Test 5: Observability

**Logs (Loki)**:
```bash
# View logs in Grafana
# http://localhost:3000
# Explore > Loki
# Query: {stage="{N}", service="orchestrator"}
```

**Metrics (Prometheus)**:
```bash
# http://localhost:9090
# Query: up{stage="{N}"}
# Should return: 1 (service is up)
```

**Traces (Tempo)**:
```bash
# Instrument your code with OpenTelemetry
# Traces appear in Grafana > Explore > Tempo
```

---

## Advanced Patterns

### Pattern 1: Multi-Service Stage

If your stage has multiple internal services (like Stage 2 NER, DP, Event LLM):

```yaml
services:
  orchestrator-service:
    container_name: {stage}-orchestrator
    networks:
      - storytelling
    environment:
      - SERVICE_A_URL=http://{stage}-service-a:8001
      - SERVICE_B_URL=http://{stage}-service-b:8002

  service-a:
    container_name: {stage}-service-a
    networks:
      - storytelling
    labels:
      - "stage={N}"
      - "service=service-a"
    ports: []  # Internal only

  service-b:
    container_name: {stage}-service-b
    networks:
      - storytelling
    labels:
      - "stage={N}"
      - "service=service-b"
    ports: []
```

---

### Pattern 2: GPU Allocation

If your stage needs GPU (like Stage 2 Event LLM):

```yaml
services:
  gpu-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

---

### Pattern 3: Inter-Stage Communication

Stages can communicate via:

**Option 1: Direct HTTP** (if both running):
```python
import httpx

# Call Stage 2 from Stage 3
async with httpx.AsyncClient() as client:
    response = await client.get("http://nlp-orchestrator:8000/health")
```

**Option 2: Via Traefik** (recommended for external access):
```python
# From outside Docker network
response = await client.get("http://localhost/api/v1/nlp/health")
```

**Option 3: Message Queue** (async, decoupled):
```python
# Stage 2 publishes to queue
await queue.publish("document_processed", {"doc_id": "123", "result": ...})

# Stage 3 consumes from queue
async for message in queue.consume("document_processed"):
    process_document(message)
```

---

### Pattern 4: Shared Data via PostgreSQL

Stages can share data through database:

```python
# Stage 2 writes
await db.execute(
    "INSERT INTO shared.documents (id, text, stage) VALUES ($1, $2, $3)",
    doc_id, text, 2
)

# Stage 3 reads
doc = await db.fetchrow(
    "SELECT * FROM shared.documents WHERE id = $1 AND stage = 2",
    doc_id
)
```

**Schema**:
```sql
-- Create shared schema (in postgres-init)
CREATE SCHEMA IF NOT EXISTS shared;
GRANT ALL ON SCHEMA shared TO stage1_user, stage2_user, stage3_user, ...;
```

---

## Checklist: Integration Complete

Before considering your integration complete, verify:

- [ ] `docker-compose.infrastructure.yml` created
- [ ] Connected to `storytelling` network
- [ ] Using infrastructure Redis (broker + cache)
- [ ] Using infrastructure PostgreSQL
- [ ] No port mappings (Traefik routes traffic)
- [ ] Container names follow convention
- [ ] `/health` endpoint implemented
- [ ] Traefik route registered
- [ ] Prometheus scrape target added
- [ ] Resource limits defined
- [ ] Environment variables follow naming convention
- [ ] Helper script created (`run-with-infrastructure.sh`)
- [ ] `.env.example` created
- [ ] Structured logging implemented
- [ ] OpenTelemetry configured (optional)
- [ ] Tests pass
- [ ] Documentation updated

---

## Getting Help

1. **Reference Implementation**: `/stage2-nlp-processing/`
2. **Infrastructure Docs**: `/infrastructure/README.md`
3. **Quick Start**: `/QUICK_START.md`
4. **Troubleshooting**: This guide's troubleshooting section

**Common Issues**: 90% of integration problems are due to:
1. Not using external network
2. Using `localhost` instead of service names
3. Missing Traefik route
4. No health check endpoint

---

**Last Updated**: December 2025
**Maintained By**: Platform Team
**Questions**: See `/infrastructure/README.md#getting-help`
