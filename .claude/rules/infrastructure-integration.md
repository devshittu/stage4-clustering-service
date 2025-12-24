# Infrastructure Integration Rules

## CRITICAL: Mandatory Infrastructure Usage

All stages (1-8) in the Storytelling Platform **MUST** use the centralized infrastructure. Violating these rules creates technical debt, resource conflicts, and breaks the platform architecture.

## Rule 1: NEVER Declare Infrastructure Services Locally

### ❌ FORBIDDEN

```yaml
# stage{N}-service/docker-compose.yml
services:
  redis:  # ❌ NEVER declare Redis locally
    image: redis:7-alpine

  postgres:  # ❌ NEVER declare Postgres locally
    image: postgres:16

  neo4j:  # ❌ NEVER declare Neo4j locally
    image: neo4j:5
```

### ✅ REQUIRED

```yaml
# stage{N}-service/docker-compose.infrastructure.yml
networks:
  storytelling:
    external: true  # Use centralized network

services:
  your-service:
    networks:
      - storytelling
    environment:
      # Connect to infrastructure services
      - REDIS_HOST=redis-broker
      - POSTGRES_HOST=postgres
```

**Rationale**: Declaring services locally causes port conflicts, resource waste, and breaks service discovery.

---

## Rule 2: Use Dedicated Database and Keyspace

### Database Naming Convention

```
stage{N}_{service_name}
```

**Examples**:
- Stage 1: `stage1_cleaning`
- Stage 2: `stage2_nlp`
- Stage 3: `stage3_embeddings`
- Stage 4: `stage4_clustering`
- Stage 5: `stage5_graph`
- Stage 6: `stage6_timeline`
- Stage 7: `stage7_api`
- Stage 8: `stage8_frontend`

### Redis Database Allocation

**Pattern**: Stage N uses DB `(N-1)*2` for Celery, `(N-1)*2+1` for cache

| Stage | Celery Broker DB | Cache DB |
|-------|-----------------|----------|
| 1 | 0 | 1 |
| 2 | 2 | 3 |
| 3 | 4 | 5 |
| 4 | 6 | 7 |
| 5 | 8 | 9 |
| 6 | 10 | 11 |
| 7 | 12 | 13 |
| 8 | 14 | 15 |

### ✅ REQUIRED Configuration

```yaml
# stage{N}-service/docker-compose.infrastructure.yml
services:
  orchestrator:
    environment:
      # PostgreSQL
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=stage{N}_{service_name}
      - POSTGRES_USER=stage{N}_user
      - POSTGRES_PASSWORD=${STAGE{N}_POSTGRES_PASSWORD}

      # Redis Broker (Celery)
      - CELERY_BROKER_URL=redis://redis-broker:6379/{N*2-2}

      # Redis Cache
      - REDIS_CACHE_URL=redis://redis-cache:6379/{N*2-1}
```

---

## Rule 3: NEVER Expose Ports Directly

### ❌ FORBIDDEN

```yaml
services:
  orchestrator-service:
    ports:
      - "8000:8000"  # ❌ Causes port conflicts across stages
```

### ✅ REQUIRED

```yaml
services:
  orchestrator-service:
    container_name: {stage}-orchestrator  # Must match Traefik config
    networks:
      - storytelling
    labels:
      - "stage={N}"
      - "service=orchestrator"
    # NO ports: directive - Traefik handles routing
```

**Access**: Via Traefik at `http://localhost/api/v1/{stage-name}/*`

**Rationale**: Direct port exposure causes conflicts when multiple stages run simultaneously. Traefik provides centralized routing, load balancing, and health checks.

---

## Rule 4: Implement Required Health Check Endpoint

### ✅ REQUIRED

Every orchestrator service MUST expose a `/health` endpoint:

```python
# src/api/orchestrator_service.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Traefik load balancer.

    Returns:
        200 OK if service is healthy
        503 Service Unavailable if unhealthy
    """
    # Check critical dependencies
    try:
        # Verify Redis connection
        await redis_client.ping()

        # Verify database connection
        await db.execute("SELECT 1")

        return {
            "status": "healthy",
            "service": "stage{N}-orchestrator",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )
```

**Rationale**: Traefik uses `/health` for health checks. Unhealthy services are automatically removed from routing.

---

## Rule 5: Use Docker Labels for Service Discovery

### ✅ REQUIRED Labels

```yaml
services:
  orchestrator-service:
    labels:
      # Stage identification (REQUIRED)
      - "stage={N}"

      # Service type (REQUIRED)
      - "service=orchestrator"

      # Project grouping (REQUIRED)
      - "com.docker.compose.project=stage{N}-{service-name}"

      # Optional: Custom metadata
      - "version=1.0.0"
      - "maintainer=team@example.com"
```

**Rationale**: Prometheus uses these labels for service discovery. Missing labels break monitoring.

---

## Rule 6: Register Routes in Traefik Dynamic Configuration

### ✅ REQUIRED

After creating your stage, add routing to `infrastructure/traefik/dynamic.yml`:

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
          - url: "http://{stage}-orchestrator:8000"
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

**Rationale**: Without Traefik configuration, your service is unreachable via the API gateway.

---

## Rule 7: Use Infrastructure Network Exclusively

### ✅ REQUIRED

```yaml
# stage{N}-service/docker-compose.infrastructure.yml
networks:
  storytelling:
    external: true
    name: storytelling

services:
  all-services:
    networks:
      - storytelling
```

### ❌ FORBIDDEN

```yaml
networks:
  custom-network:  # ❌ Creates isolation, breaks service discovery
    driver: bridge
```

**Rationale**: All services must be on the `storytelling` network for Docker DNS resolution.

---

## Rule 8: Never Hardcode Service URLs

### ❌ FORBIDDEN

```python
# ❌ Hardcoded URLs break when infrastructure changes
REDIS_URL = "redis://localhost:6379"
POSTGRES_URL = "postgresql://localhost:5432/mydb"
```

### ✅ REQUIRED

```python
# ✅ Environment variables from docker-compose.infrastructure.yml
import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis-broker")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
    f"/{os.getenv('POSTGRES_DB')}"
)
```

**Rationale**: Hardcoded URLs break in different environments and prevent infrastructure evolution.

---

## Rule 9: Implement Observability Integration

### ✅ REQUIRED: Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "document_processed",
    document_id=doc_id,
    stage="{N}",
    service="orchestrator",
    processing_time_ms=elapsed_ms,
    status="success"
)
```

**Rationale**: Loki ingests structured logs. Unstructured logs are difficult to query.

### ✅ REQUIRED: Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
documents_processed = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['stage', 'service', 'status']
)

processing_duration = Histogram(
    'processing_duration_seconds',
    'Document processing duration',
    ['stage', 'service']
)

# Instrument code
with processing_duration.labels(stage="{N}", service="orchestrator").time():
    result = process_document(doc)

documents_processed.labels(
    stage="{N}",
    service="orchestrator",
    status="success"
).inc()
```

### ✅ OPTIONAL: OpenTelemetry Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span(
    "process_document",
    attributes={
        "stage": "{N}",
        "service": "orchestrator",
        "document_id": doc_id
    }
):
    # Your processing logic
    result = process_document(doc)
```

**Environment Variables**:
```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  - OTEL_SERVICE_NAME=stage{N}-{service-name}
```

**Rationale**: Without observability, debugging production issues is impossible.

---

## Rule 10: Resource Limits MUST Be Defined

### ✅ REQUIRED

```yaml
services:
  celery-worker:
    deploy:
      resources:
        limits:
          cpus: '{cpu_count}'
          memory: {memory}G
        reservations:
          cpus: '{min_cpu_count}'
          memory: {min_memory}G
```

**Allocation Guidelines** (48 cores, 160GB RAM total):

| Component | Allocated | Reasoning |
|-----------|-----------|-----------|
| Infrastructure | 35 cores, 85GB | Fixed overhead |
| Stage 2 Celery | 22 cores, 100GB | Largest workload (NLP) |
| **Remaining** | **10 cores, 15GB** | **For Stages 1, 3-8** |

**Per-Stage Recommendations**:
- Light stages (1, 7, 8): 1-2 cores, 2-4GB each
- Medium stages (3, 4, 6): 2-4 cores, 4-8GB each
- Heavy stages (2, 5): Shared pool

### ❌ FORBIDDEN

```yaml
services:
  celery-worker:
    # ❌ No limits - can consume all resources
```

**Rationale**: Without limits, one stage can starve others. With 8 stages, resource contention is guaranteed.

---

## Rule 11: Celery Workers MUST Use Infrastructure Broker

### ✅ REQUIRED

```yaml
# docker-compose.infrastructure.yml
services:
  celery-worker:
    environment:
      # Use infrastructure Redis broker
      - CELERY_BROKER_URL=redis://redis-broker:6379/{N*2-2}
      - CELERY_RESULT_BACKEND=redis://redis-cache:6379/{N*2-1}
    networks:
      - storytelling
```

```python
# src/core/celery_app.py
from celery import Celery
import os

app = Celery(
    f'stage{N}-worker',
    broker=os.getenv('CELERY_BROKER_URL'),
    backend=os.getenv('CELERY_RESULT_BACKEND')
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True
)
```

### ❌ FORBIDDEN

```python
# ❌ Local broker - breaks distributed processing
app = Celery('myapp', broker='redis://localhost:6379/0')
```

**Rationale**: Each stage needs isolated task queues to prevent cross-contamination.

---

## Rule 12: Container Naming Convention

### ✅ REQUIRED Pattern

```
{stage-name}-{service-type}
```

**Examples**:
```yaml
services:
  orchestrator-service:
    container_name: nlp-orchestrator  # Stage 2

  ner-service:
    container_name: nlp-ner  # Stage 2

  celery-worker:
    container_name: nlp-celery-worker  # Stage 2
```

**Full Naming Matrix**:

| Stage | Orchestrator | Worker | Other |
|-------|-------------|---------|-------|
| 1 | cleaning-orchestrator | cleaning-celery-worker | cleaning-{service} |
| 2 | nlp-orchestrator | nlp-celery-worker | nlp-ner, nlp-dp |
| 3 | embeddings-orchestrator | embeddings-celery-worker | embeddings-{service} |
| 4 | clustering-orchestrator | clustering-celery-worker | clustering-{service} |
| 5 | graph-orchestrator | graph-celery-worker | graph-{service} |
| 6 | timeline-orchestrator | timeline-celery-worker | timeline-{service} |
| 7 | api-orchestrator | api-celery-worker | api-{service} |
| 8 | frontend | frontend-server | frontend-{service} |

**Rationale**: Consistent naming enables:
- Easy Docker commands: `docker logs nlp-orchestrator`
- Traefik service discovery
- Debugging across stages

---

## Rule 13: Environment Variable Prefix Convention

### ✅ REQUIRED

Use service-specific prefixes to avoid collisions:

```bash
# Infrastructure services (no prefix)
REDIS_HOST=redis-broker
POSTGRES_HOST=postgres
NEO4J_HOST=neo4j

# Stage-specific (use prefix)
STAGE2_NER_SERVICE_URL=http://nlp-ner:8001
STAGE2_DP_SERVICE_URL=http://nlp-dp:8002
STAGE2_EVENT_LLM_SERVICE_URL=http://nlp-event-llm:8003

# Observability (standard OTEL_ prefix)
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
OTEL_SERVICE_NAME=stage2-nlp-orchestrator
```

### ❌ FORBIDDEN

```bash
# ❌ Generic names cause conflicts
NER_URL=http://localhost:8001
DB_HOST=localhost
```

**Rationale**: Multiple stages may have services with similar names. Prefixes prevent conflicts.

---

## Rule 14: Secrets Management

### ✅ REQUIRED (Development)

```yaml
# docker-compose.infrastructure.yml
services:
  orchestrator:
    environment:
      - POSTGRES_PASSWORD=${STAGE{N}_POSTGRES_PASSWORD}
    env_file:
      - .env  # Load from file
```

```bash
# .env (gitignored)
STAGE2_POSTGRES_PASSWORD=stage2_secure_password
STAGE2_NEO4J_PASSWORD=stage2_neo4j_password
```

### ✅ REQUIRED (Production)

Use Docker Secrets:

```yaml
services:
  orchestrator:
    secrets:
      - postgres_password
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

secrets:
  postgres_password:
    external: true
```

```bash
# Create secret
echo "secure_password" | docker secret create stage{N}_postgres_password -
```

### ❌ FORBIDDEN

```yaml
environment:
  - POSTGRES_PASSWORD=hardcoded_password  # ❌ Security violation
```

**Rationale**: Hardcoded secrets in docker-compose.yml are committed to git, creating security vulnerabilities.

---

## Rule 15: File Organization

### ✅ REQUIRED Structure

```
stage{N}-{service-name}/
├── docker-compose.yml                  # Standalone mode (dev/testing)
├── docker-compose.infrastructure.yml   # Infrastructure integration (REQUIRED)
├── .env.example                        # Template for secrets
├── .env                                # Actual secrets (gitignored)
├── run-with-infrastructure.sh          # Helper script (REQUIRED)
├── Dockerfile                          # Main service
├── Dockerfile_{service}                # Additional services
├── requirements.txt                    # Python dependencies
├── .claude/
│   └── CLAUDE.md                       # Integration instructions
└── src/
    └── ...
```

### ✅ REQUIRED: Infrastructure Integration File

Every stage MUST have `docker-compose.infrastructure.yml`:

```yaml
version: '3.9'

networks:
  storytelling:
    external: true
    name: storytelling

services:
  # Override base services with infrastructure config
  orchestrator-service:
    container_name: {stage}-orchestrator
    networks:
      - storytelling
    environment:
      - REDIS_HOST=redis-broker
      - POSTGRES_HOST=postgres
      # ... infrastructure connection strings
    labels:
      - "stage={N}"
      - "service=orchestrator"
    ports: []  # Remove port mappings
```

**Rationale**: Separation of standalone (dev) and infrastructure (production) configs enables flexible deployment.

---

## Rule 16: Startup Dependencies

### ✅ REQUIRED

```yaml
services:
  orchestrator-service:
    depends_on:
      infrastructure-check:
        condition: service_completed_successfully

  infrastructure-check:
    image: busybox:latest
    networks:
      - storytelling
    command: >
      sh -c "
        echo 'Checking infrastructure availability...' &&
        nc -zv redis-broker 6379 &&
        nc -zv postgres 5432 &&
        echo 'Infrastructure ready!'
      "
```

**Rationale**: Services crash if infrastructure isn't ready. Health checks prevent race conditions.

---

## Rule 17: Logging Configuration

### ✅ REQUIRED

```yaml
services:
  orchestrator-service:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        labels: "stage,service"
```

**Rationale**: Without log rotation, disk fills up. Loki ingests logs via Docker socket.

---

## Enforcement

These rules are enforced via:

1. **Code Review**: PRs must pass infrastructure compliance check
2. **CI/CD**: Automated validation of docker-compose.infrastructure.yml
3. **Documentation**: .claude/CLAUDE.md must reference this file
4. **Health Checks**: Non-compliant services fail health checks

## Violations

Violation of these rules results in:

- ❌ Service fails to start
- ❌ Service unreachable via Traefik
- ❌ No monitoring/observability
- ❌ Resource conflicts with other stages
- ❌ Data corruption from shared resources
- ❌ PR rejection

## Getting Help

- **Documentation**: `/infrastructure/README.md`
- **Examples**: Stage 2 NLP Processing (reference implementation)
- **Quick Start**: `/QUICK_START.md`
- **Troubleshooting**: `/infrastructure/README.md#troubleshooting`

---

**Last Updated**: December 2025
**Applies To**: All 8 stages of Storytelling Platform
**Compliance**: Mandatory for all new services
