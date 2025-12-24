Create a branch off (to implement the following) of the dev branch first and then implement the following
To make the system more robust, scalable, and user-controllable, incorporate advanced batch lifecycle management and resource handling. Design these features to be non-blocking, interruptible, and efficient, leveraging tools like Celery for task orchestration, Redis for queuing/state management, and signals/hooks for control. Ensure the system can handle concurrent requests without downtime, while gracefully managing resources on the target hardware (e.g., releasing GPU/CPU/RAM when idle to prevent lockup).

Batch Lifecycle Controls: Provide comprehensive API/CLI endpoints and mechanisms for managing batches dynamically:
Start: Initiate a new batch processing job (e.g., via POST /v1/documents/batch), assigning a unique job_id and queuing it if the system is busy.
Stop/Pause: Temporarily halt a running batch without discarding progress (e.g., via PATCH /v1/jobs/{job_id}/pause). Use signals (e.g., Celery's revoke with terminate=False) to interrupt workers mid-process, saving partial results.
Continue/Resume: Restart a paused batch from its last checkpoint (e.g., via PATCH /v1/jobs/{job_id}/resume), reloading state from Redis or persistent storage.
Cancel: Fully abort a batch, discarding all processed data and artifacts (e.g., via DELETE /v1/jobs/{job_id}), including cleanup of temporary files, database entries, and cached items. Trigger resource release immediately.
Start New: Allow initiating a new batch while others are running or paused, adding it to a queue for sequential or parallel execution based on available resources.
Status Monitoring: Real-time querying (e.g., GET /v1/jobs/{job_id}) showing states like "queued", "running", "paused", "completed", "canceled", with progress metrics (e.g., docs processed/total).

Non-Blocking and Queuing Behavior: Ensure the system is fully asynchronous and queue-based (if not already):
Accept new batch requests even while processing others; enqueue them in Redis or RabbitMQ without blocking the API response (return 202 Accepted with job_id immediately).
Automatically start queued batches in FIFO order once the current one completes, unless explicitly paused or canceled.
Handle interruptions via OS signals or API calls (e.g., SIGINT for graceful shutdown), allowing safe termination of running tasks without data corruption.
Scale to multiple workers (e.g., via Celery's concurrency or Dask clusters) for parallel processing, limited by hardware (e.g., 22 workers max).

Progressive Persistence: Implement incremental saving to avoid data loss and enable resumability:
As each document in a batch completes processing, immediately persist it to configured backends (JSONL, PostgreSQL, Elasticsearch) using atomic operations.
Track per-document status in Redis (e.g., {job_id: {doc_id: "completed"}}) for checkpointing.
On pause/resume, skip already-persisted docs and resume from the next unprocessed one.
Support configurable persistence strategies (e.g., batch-size thresholds for bulk inserts) in settings.yaml to balance performance and reliability.

Cleanup Hooks for Heavy Models: Add lifecycle hooks to manage memory-intensive components:
On service startup/shutdown, batch completion, pause, or idle timeouts (e.g., 5min no activity), explicitly unload models (e.g., del model; torch.cuda.empty_cache() for vLLM/Mistral).
Use context managers or decorators for inference calls to ensure cleanup (e.g., with ModelLoader() as model: ...).
Monitor and log resource usage (e.g., via nvidia-smi in hooks) to detect leaks, integrating with observability tools like Prometheus.

Resource Lifecycle Management: Optimize for efficient resource utilization and release:
Dynamically allocate/release resources (GPU, CPU, RAM) based on workload—e.g., scale Dask workers up/down, or use continuous batching to minimize idle GPU time.
Implement idle detection: If no active/queued jobs, enter low-resource mode (e.g., unload models, reduce worker pools) to free up hardware for other processes.
Use tools like torch's autocast for mixed precision and quantization to reduce footprint.
Include health checks (/health) that report resource status (e.g., GPU utilization <10% when idle).
Ensure fault-tolerance: On errors (e.g., OOM), gracefully retry docs, release resources, and notify via logs/Sentry without crashing the service.

These features should integrate seamlessly with the existing architecture (e.g., Celery for batches, Redis for state), enhancing UX and scalability without overcomplicating the core flow. Be creative in implementation details (e.g., WebSockets for real-time progress updates), but prioritize reliability—test for scenarios like mid-batch cancellations and resource spikes. When running example data use the infrastructure docker compose


run the project using the infrastructure, and run the file as batch in the docker env  ./data/extracted_events_2025-12-15.jsonl on this stage address all errors and ensure no regressions. ensure all the implemented capabilities are fully functional and error free.

ENsure the stage is able to receive and send job as via webhooks redis-stream, nats etc check the config

create a repo dir up the current project and for stage 4  follow the same naming convention. use /permissions to pre-approve all commands