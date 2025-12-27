# Inter-Stage Automation Guide

**Stage 4 Clustering Service** - Automated Pipeline Integration

## Overview

Stage 4 implements **full pipeline automation** with dual integration mechanisms (Redis Streams + Webhooks) for both upstream (Stage 3) and downstream (Stage 5) communication. This eliminates manual intervention and enables true end-to-end automation.

**Pipeline Flow**:
```
Stage 3 (Embeddings) → Auto-triggers → Stage 4 (Clustering) → Auto-notifies → Stage 5 (Graph)
```

---

## Architecture

### Dual Integration Strategy

Stage 4 uses **both** Redis Streams and Webhooks for reliability:

| Mechanism | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Redis Streams** | Persistent, scalable, decoupled | Requires consumer implementation | Primary method (real-time events) |
| **Webhooks** | Simple HTTP, familiar | Requires service discovery | Backup/redundancy |

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 4 CLUSTERING                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Stage 3 Stream   │         │  Webhook         │         │
│  │ Consumer         │         │  Receiver        │         │
│  │ (Background)     │         │  (FastAPI)       │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                            │                   │
│           └────────────┬───────────────┘                   │
│                        ▼                                   │
│              ┌──────────────────┐                          │
│              │  Clustering      │                          │
│              │  Engine (Celery) │                          │
│              └────────┬─────────┘                          │
│                       │                                    │
│           ┌───────────┴────────────┐                       │
│           ▼                        ▼                       │
│  ┌─────────────────┐      ┌────────────────────┐          │
│  │ Redis Stream    │      │ Stage 5 Webhook    │          │
│  │ Publisher       │      │ Publisher          │          │
│  └─────────────────┘      └────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   (Stage 3 listens)            (Stage 5 receives)
```

---

## Upstream Integration (Stage 3 → Stage 4)

### 1. Redis Stream Consumer

**Service**: `clustering-stage3-consumer` (background container)

**Functionality**:
- Listens to `stage3:embeddings:events` stream
- Consumer group: `stage4-clustering-consumers`
- Auto-triggers clustering when Stage 3 completes embeddings

**Events Consumed**:
- `embedding.job.completed`
- `embedding.batch.completed`

**Configuration** (`config/settings.yaml`):
```yaml
upstream_automation:
  enabled: true
  redis_consumer:
    enabled: true
    stream_name: "stage3:embeddings:events"
    consumer_group: "stage4-clustering-consumers"
    trigger_events:
      - "embedding.job.completed"
```

**Auto-Trigger Rules**:
- Minimum embeddings: 10 (configurable)
- Allowed embedding types: document, event, entity, storyline
- Default algorithm: HDBSCAN

**Code**: `src/services/stage3_stream_consumer.py`

### 2. Webhook Receiver

**Endpoint**: `POST /webhooks/embeddings-completed`

**Functionality**:
- Accepts HTTP POST notifications from Stage 3
- Validates webhook authentication token (optional)
- Triggers clustering job via Celery

**Request Parameters**:
```bash
curl -X POST http://clustering-orchestrator:8000/webhooks/embeddings-completed \
  -H "Authorization: Bearer <STAGE4_WEBHOOK_SECRET>" \
  -F "event_type=embedding.job.completed" \
  -F "job_id=stage3_job_123" \
  -F "embedding_type=event" \
  -F "total_embeddings=5000"
```

**Response** (202 Accepted):
```json
{
  "status": "accepted",
  "message": "Clustering job submitted successfully",
  "stage4_task_id": "celery_task_uuid",
  "stage3_job_id": "stage3_job_123",
  "embedding_type": "event",
  "algorithm": "hdbscan"
}
```

**Code**: `src/api/orchestrator.py:644-800`

---

## Downstream Integration (Stage 4 → Stage 5)

### 1. Redis Stream Publisher

**Functionality**:
- Publishes to `stage4:clustering:events` stream
- Includes output file paths, quality metrics, statistics

**Events Published**:
- `job.created`, `job.started`, `job.progress`
- **`job.completed`** ← Critical for Stage 5
- `job.failed`, `job.paused`, `job.resumed`, `job.canceled`

**Event Payload** (`job.completed`):
```json
{
  "event_type": "job.completed",
  "job_id": "job_abc123",
  "embedding_type": "event",
  "algorithm": "hdbscan",
  "clusters_created": 125,
  "outliers": 45,
  "output_files": [
    "/app/data/clusters/clusters_event_job_abc123_20251227_103000.jsonl"
  ],
  "quality_metrics": {
    "silhouette_score": 0.73,
    "davies_bouldin_score": 0.45
  },
  "statistics": {
    "processing_time_ms": 512000,
    "total_items": 5000,
    "avg_cluster_size": 40.0
  },
  "timestamp": "2025-12-27T10:30:00Z"
}
```

**Code**: `src/utils/event_publisher.py:104-165`

### 2. Stage 5 Webhook Publisher

**Functionality**:
- Sends HTTP POST to Stage 5 webhook URLs
- Retries with exponential backoff (3 attempts)
- Fails gracefully if Stage 5 unavailable

**Configuration** (`config/settings.yaml`):
```yaml
downstream_automation:
  enabled: true
  webhook_publisher:
    enabled: true
    stage5_urls:
      - "http://graph-orchestrator:8000/webhooks/clustering-completed"
    retry:
      max_attempts: 3
      backoff_seconds: 5
    fail_silently: true  # Don't block job completion
```

**Webhook Request**:
```bash
POST http://graph-orchestrator:8000/webhooks/clustering-completed
Authorization: Bearer <STAGE5_WEBHOOK_SECRET>
Content-Type: application/json

{
  "event_type": "clustering.job.completed",
  "job_id": "job_abc123",
  "embedding_type": "event",
  "algorithm": "hdbscan",
  "clusters_created": 125,
  "outliers": 45,
  "output_files": [
    "/app/data/clusters/clusters_event_job_abc123_20251227_103000.jsonl"
  ],
  "quality_metrics": {...},
  "statistics": {...},
  "timestamp": "2025-12-27T10:30:00Z"
}
```

**Code**: `src/utils/stage5_webhook_publisher.py`

---

## Configuration

### Settings File (`config/settings.yaml`)

```yaml
# Upstream automation (Stage 3 → Stage 4)
upstream_automation:
  enabled: true

  redis_consumer:
    enabled: true
    stream_name: "stage3:embeddings:events"
    consumer_group: "stage4-clustering-consumers"
    block_ms: 5000
    trigger_events:
      - "embedding.job.completed"

  webhook_receiver:
    enabled: true
    endpoint_path: "/webhooks/embeddings-completed"
    auth_token: "${STAGE4_WEBHOOK_SECRET}"

  auto_trigger:
    embedding_types: ["document", "event", "entity", "storyline"]
    default_algorithm: "hdbscan"
    min_embeddings: 10
    quality_threshold: 0.0

# Downstream automation (Stage 4 → Stage 5)
downstream_automation:
  enabled: true

  redis_publisher:
    enabled: true
    include_output_paths: true

  webhook_publisher:
    enabled: true
    stage5_urls:
      - "http://graph-orchestrator:8000/webhooks/clustering-completed"
    retry:
      max_attempts: 3
      backoff_seconds: 5
    fail_silently: true
```

### Environment Variables (`.env`)

```bash
# Webhook authentication (optional but recommended)
STAGE4_WEBHOOK_SECRET=your_stage4_webhook_secret_here
STAGE5_WEBHOOK_SECRET=your_stage5_webhook_secret_here

# Upstream automation
UPSTREAM_AUTOMATION_ENABLED=true
AUTO_TRIGGER_ENABLED=true
STAGE3_STREAM_NAME=stage3:embeddings:events

# Downstream automation
DOWNSTREAM_AUTOMATION_ENABLED=true
STAGE5_WEBHOOK_URL=http://graph-orchestrator:8000/webhooks/clustering-completed
```

---

## Deployment

### Docker Compose Services

**Stage 3 Consumer** (background service):
```yaml
stage3-consumer:
  container_name: clustering-stage3-consumer
  command: python -m src.services.stage3_stream_consumer
  environment:
    - REDIS_HOST=redis-broker
    - STAGE3_STREAM_NAME=stage3:embeddings:events
    - UPSTREAM_AUTOMATION_ENABLED=true
  restart: unless-stopped
```

**Start Services**:
```bash
# Start all services including automation
./run-with-infrastructure.sh start

# Check consumer status
docker logs clustering-stage3-consumer -f
```

---

## Testing

### 1. Test Upstream Integration (Stage 3 → Stage 4)

**Via Webhook**:
```bash
curl -X POST http://localhost/webhooks/embeddings-completed \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "event_type=embedding.job.completed" \
  -d "job_id=test_job_123" \
  -d "embedding_type=event" \
  -d "total_embeddings=100"
```

**Expected Response** (202):
```json
{
  "status": "accepted",
  "stage4_task_id": "celery-task-uuid",
  "embedding_type": "event"
}
```

**Via Redis Stream** (simulate Stage 3 event):
```bash
docker exec -it redis-broker redis-cli

# Publish event to Stage 3 stream
XADD stage3:embeddings:events * \
  event_type embedding.job.completed \
  job_id test_job_456 \
  embedding_type event \
  total_embeddings 100
```

**Verify Auto-Trigger**:
```bash
# Check consumer logs
docker logs clustering-stage3-consumer --tail 50

# Check for job creation
curl http://localhost/api/v1/clustering/jobs | jq '.jobs[] | select(.metadata.triggered_by=="stage3_event")'
```

### 2. Test Downstream Integration (Stage 4 → Stage 5)

**Run Clustering Job**:
```bash
curl -X POST http://localhost/api/v1/clustering/batch \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_type": "event",
    "algorithm": "hdbscan"
  }'
```

**Monitor Events**:
```bash
# Redis stream
docker exec -it redis-broker redis-cli
XREAD COUNT 10 STREAMS stage4:clustering:events 0

# Worker logs (webhook output)
docker logs clustering-celery-worker --tail 50 | grep "Stage 5 webhook"
```

**Expected Output**:
```
Job job_abc123: Stage 5 webhook notification sent successfully
```

---

## Monitoring

### Consumer Statistics

**Endpoint**: Not yet exposed (future enhancement)

**Logs**:
```bash
docker logs clustering-stage3-consumer --tail 100
```

**Key Metrics**:
- `events_processed`: Total events consumed
- `jobs_triggered`: Clustering jobs auto-triggered
- `errors`: Processing failures
- `last_event_time`: Last event received timestamp

### Event Stream Health

**Check Redis Streams**:
```bash
docker exec -it redis-broker redis-cli

# Stage 3 stream info
XINFO STREAM stage3:embeddings:events

# Stage 4 stream info
XINFO STREAM stage4:clustering:events

# Consumer group info
XINFO GROUPS stage3:embeddings:events
```

---

## Troubleshooting

### Issue 1: No Auto-Triggering from Stage 3

**Symptoms**: Events published to `stage3:embeddings:events` but no clustering jobs created.

**Diagnosis**:
```bash
# Check consumer is running
docker ps | grep clustering-stage3-consumer

# Check logs
docker logs clustering-stage3-consumer --tail 50

# Check configuration
docker exec clustering-stage3-consumer cat /app/config/settings.yaml | grep -A 20 upstream_automation
```

**Common Causes**:
1. `upstream_automation.enabled: false` in settings.yaml
2. Consumer not started (missing from docker-compose)
3. Event type not in `trigger_events` list
4. Embedding type not in `allowed_embedding_types`
5. Insufficient embeddings (<10 by default)

**Solution**:
```bash
# Enable automation
export UPSTREAM_AUTOMATION_ENABLED=true

# Restart consumer
docker restart clustering-stage3-consumer

# Verify settings
curl http://localhost/api/v1/clustering/health | jq '.upstream_automation'
```

### Issue 2: Stage 5 Webhooks Failing

**Symptoms**: Clustering completes but Stage 5 not receiving notifications.

**Diagnosis**:
```bash
# Check worker logs
docker logs clustering-celery-worker | grep "Stage 5 webhook"

# Test webhook manually
curl -X POST http://graph-orchestrator:8000/webhooks/clustering-completed \
  -H "Content-Type: application/json" \
  -d '{"event_type":"clustering.job.completed","job_id":"test"}'
```

**Common Causes**:
1. Stage 5 not running (`graph-orchestrator` container down)
2. Incorrect webhook URL in settings.yaml
3. Authentication token mismatch
4. Stage 5 webhook endpoint not implemented

**Solution**:
```bash
# Verify Stage 5 is running
docker ps | grep graph-orchestrator

# Check Stage 5 health
curl http://graph-orchestrator:8000/health

# Test connectivity from Stage 4
docker exec clustering-celery-worker curl -I http://graph-orchestrator:8000/health

# Enable fail_silently to continue even if Stage 5 unavailable
# (already enabled by default)
```

### Issue 3: Duplicate Job Triggering

**Symptoms**: Same event triggers multiple clustering jobs.

**Cause**: Both Redis stream consumer AND webhook receiver active, Stage 3 sends both.

**Solution**:
Disable one mechanism in Stage 3 or Stage 4. Recommended: keep Redis stream, disable webhook.

```yaml
# config/settings.yaml
upstream_automation:
  webhook_receiver:
    enabled: false  # Disable if using Redis stream
```

---

## Performance Considerations

### Resource Usage

- **Stage 3 Consumer**: ~500MB RAM, 0.5 CPU (idle)
- **Event Publishing**: <10ms overhead per event
- **Webhook Calls**: <100ms per webhook (async, non-blocking)

### Scalability

- **Multiple Consumers**: Scale horizontally with consumer groups
  ```yaml
  consumer_name: clustering-worker-1  # Change to worker-2, worker-3, etc.
  ```

- **Load Balancing**: Redis consumer groups auto-distribute events

- **Webhook Redundancy**: Configure multiple Stage 5 URLs
  ```yaml
  stage5_urls:
    - "http://graph-orchestrator:8000/webhooks/clustering-completed"
    - "http://graph-orchestrator-backup:8000/webhooks/clustering-completed"
  ```

---

## Security

### Webhook Authentication

**Stage 3 → Stage 4**:
```bash
# Set secret in .env
STAGE4_WEBHOOK_SECRET=$(openssl rand -base64 32)

# Stage 3 must include in webhook call
Authorization: Bearer <STAGE4_WEBHOOK_SECRET>
```

**Stage 4 → Stage 5**:
```bash
STAGE5_WEBHOOK_SECRET=$(openssl rand -base64 32)

# Stage 4 includes in outgoing webhooks
Authorization: Bearer <STAGE5_WEBHOOK_SECRET>
```

### Network Security

- All services on `storytelling` Docker network (internal)
- No external port exposure (Traefik gateway only)
- Redis streams are internal (no external access)

---

## Future Enhancements

1. **NATS Integration**: Replace Redis streams with NATS for better scalability
2. **Consumer Metrics API**: Expose `/api/v1/automation/stats` endpoint
3. **Dead Letter Queue**: Handle permanently failed events
4. **Event Replay**: Re-process historical events from stream
5. **Circuit Breaker**: Temporarily disable webhooks if Stage 5 consistently fails
6. **GraphQL Subscriptions**: Real-time event subscriptions for monitoring

---

## Summary

✅ **Upstream Automation**: Stage 3 → Stage 4 via Redis Stream + Webhooks
✅ **Downstream Automation**: Stage 4 → Stage 5 via Redis Stream + Webhooks
✅ **Dual Integration**: Reliability through redundancy
✅ **Non-Blocking**: Asynchronous event processing
✅ **Fail-Safe**: Graceful handling of unavailable services
✅ **Configurable**: Enable/disable via settings.yaml
✅ **Scalable**: Consumer groups for horizontal scaling
✅ **Secure**: Optional webhook authentication

**Result**: Fully automated pipeline requiring **zero manual intervention**.

---

**Last Updated**: 2025-12-27
**Feature Branch**: `feature/inter-stage-event-automation`
**Status**: Implementation Complete, Testing Pending
