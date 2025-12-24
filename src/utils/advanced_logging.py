"""
Advanced Logging Module

Provides structured JSON logging with:
- Correlation ID tracking for distributed tracing
- Performance metrics (timing, throughput)
- GPU/CPU/memory metrics
- Batch progress logging
- Context managers for automatic timing
"""

import contextlib
import functools
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import psutil
import structlog
from structlog.types import EventDict, Processor


try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# Structured Logging Configuration
# =============================================================================


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    service_name: str = "stage4-clustering",
) -> None:
    """
    Configure structured logging with structlog.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "console")
        log_file: Optional log file path
        service_name: Service name for log context
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.root.addHandler(file_handler)

    # Configure structlog processors
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_service_context(service_name),
    ]

    # Add appropriate renderer
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def add_service_context(service_name: str) -> Processor:
    """
    Add service-level context to all log events.

    Args:
        service_name: Service name

    Returns:
        Processor function
    """

    def processor(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = service_name
        event_dict["stage"] = "4"
        return event_dict

    return processor


# =============================================================================
# Correlation ID Context
# =============================================================================


class LogContext:
    """
    Context manager for correlation ID tracking.

    Maintains correlation IDs for distributed tracing across
    service boundaries.
    """

    _correlation_id: Optional[str] = None

    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        cls._correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID."""
        return cls._correlation_id

    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID."""
        cls._correlation_id = None

    @classmethod
    @contextlib.contextmanager
    def correlation_context(cls, correlation_id: str):
        """
        Context manager for correlation ID.

        Example:
            with LogContext.correlation_context("job-123"):
                logger.info("processing")  # Includes correlation_id
        """
        previous_id = cls._correlation_id
        cls._correlation_id = correlation_id
        try:
            yield
        finally:
            cls._correlation_id = previous_id


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get logger with automatic correlation ID binding.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Bound logger instance
    """
    logger = structlog.get_logger(name)

    # Bind correlation ID if available
    correlation_id = LogContext.get_correlation_id()
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)

    return logger


# =============================================================================
# Performance Logger
# =============================================================================


class PerformanceLogger:
    """
    Context manager for automatic performance timing and logging.

    Tracks execution time and optional throughput metrics.
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[structlog.BoundLogger] = None,
        log_level: str = "info",
        item_count: Optional[int] = None,
        **extra_context: Any,
    ):
        """
        Initialize performance logger.

        Args:
            operation: Operation name for logging
            logger: Logger instance (creates new if None)
            log_level: Log level for output
            item_count: Number of items processed (for throughput)
            **extra_context: Additional context fields
        """
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.log_level = log_level
        self.item_count = item_count
        self.extra_context = extra_context
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(
            "operation_started",
            operation=self.operation,
            **self.extra_context,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and log results."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        log_data = {
            "operation": self.operation,
            "duration_seconds": round(duration, 3),
            **self.extra_context,
        }

        # Calculate throughput if item count provided
        if self.item_count is not None and self.item_count > 0:
            log_data["item_count"] = self.item_count
            log_data["items_per_second"] = round(self.item_count / duration, 2)

        # Log at appropriate level
        if exc_type is not None:
            log_data["error"] = str(exc_val)
            log_data["error_type"] = exc_type.__name__
            self.logger.error("operation_failed", **log_data)
        else:
            getattr(self.logger, self.log_level)("operation_completed", **log_data)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time (even if context not exited)."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def timed(
    operation: Optional[str] = None,
    log_level: str = "info",
) -> Callable:
    """
    Decorator for automatic timing of functions.

    Args:
        operation: Operation name (defaults to function name)
        log_level: Log level for output

    Example:
        @timed(operation="cluster_embeddings")
        def cluster(embeddings):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            with PerformanceLogger(op_name, log_level=log_level):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Batch Progress Logger
# =============================================================================


class BatchLogger:
    """
    Logger for batch processing with progress tracking.

    Reduces log spam by only logging every N items while tracking
    overall progress and throughput.
    """

    def __init__(
        self,
        total_items: int,
        operation: str,
        log_interval: int = 100,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        """
        Initialize batch logger.

        Args:
            total_items: Total number of items to process
            operation: Operation name
            log_interval: Log progress every N items
            logger: Logger instance
        """
        self.total_items = total_items
        self.operation = operation
        self.log_interval = log_interval
        self.logger = logger or get_logger(__name__)

        self.processed_items = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_count = 0

    def update(self, count: int = 1) -> None:
        """
        Update progress by count items.

        Args:
            count: Number of items processed
        """
        self.processed_items += count

        # Log if interval reached
        if (
            self.processed_items - self.last_log_count >= self.log_interval
            or self.processed_items >= self.total_items
        ):
            self._log_progress()
            self.last_log_count = self.processed_items
            self.last_log_time = time.time()

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = time.time() - self.start_time
        progress_pct = (self.processed_items / self.total_items) * 100 if self.total_items > 0 else 0

        # Calculate throughput
        items_per_sec = self.processed_items / elapsed if elapsed > 0 else 0

        # Estimate time remaining
        if items_per_sec > 0:
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items / items_per_sec
        else:
            eta_seconds = 0

        self.logger.info(
            "batch_progress",
            operation=self.operation,
            processed=self.processed_items,
            total=self.total_items,
            progress_pct=round(progress_pct, 1),
            items_per_second=round(items_per_sec, 2),
            elapsed_seconds=round(elapsed, 1),
            eta_seconds=round(eta_seconds, 1),
        )

    def complete(self) -> None:
        """Log completion statistics."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.processed_items / elapsed if elapsed > 0 else 0

        self.logger.info(
            "batch_completed",
            operation=self.operation,
            total_items=self.processed_items,
            duration_seconds=round(elapsed, 3),
            items_per_second=round(items_per_sec, 2),
        )


# =============================================================================
# Metrics Logger
# =============================================================================


class MetricsLogger:
    """
    Logger for system metrics (CPU, memory, GPU).

    Captures snapshots of system resource usage for performance monitoring.
    """

    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        """
        Initialize metrics logger.

        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.process = psutil.Process()

    def log_cpu_memory(self, context: Optional[str] = None) -> dict[str, Any]:
        """
        Log CPU and memory metrics.

        Args:
            context: Optional context label

        Returns:
            Metrics dictionary
        """
        metrics = {
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / (1024 * 1024),
            "memory_percent": self.process.memory_percent(),
        }

        if context:
            metrics["context"] = context

        self.logger.info("cpu_memory_metrics", **metrics)
        return metrics

    def log_gpu_metrics(self, context: Optional[str] = None) -> dict[str, Any]:
        """
        Log GPU metrics (if available).

        Args:
            context: Optional context label

        Returns:
            Metrics dictionary
        """
        if not GPU_AVAILABLE:
            return {}

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {}

            gpu = gpus[0]  # Use first GPU
            metrics = {
                "gpu_id": gpu.id,
                "gpu_name": gpu.name,
                "gpu_utilization_percent": gpu.load * 100,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal,
                "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                "gpu_temperature": gpu.temperature,
            }

            if context:
                metrics["context"] = context

            self.logger.info("gpu_metrics", **metrics)
            return metrics
        except Exception as e:
            self.logger.warning("gpu_metrics_error", error=str(e))
            return {}

    def log_all_metrics(self, context: Optional[str] = None) -> dict[str, Any]:
        """
        Log all available metrics.

        Args:
            context: Optional context label

        Returns:
            Combined metrics dictionary
        """
        metrics = {}
        metrics.update(self.log_cpu_memory(context=context))
        metrics.update(self.log_gpu_metrics(context=context))
        return metrics


# =============================================================================
# Utility Functions
# =============================================================================


@contextlib.contextmanager
def log_exceptions(
    logger: Optional[structlog.BoundLogger] = None,
    operation: Optional[str] = None,
    reraise: bool = True,
):
    """
    Context manager to automatically log exceptions.

    Args:
        logger: Logger instance
        operation: Operation name for context
        reraise: Whether to reraise exception after logging

    Example:
        with log_exceptions(operation="load_index"):
            load_faiss_index(path)
    """
    log = logger or get_logger(__name__)
    try:
        yield
    except Exception as e:
        log_data = {
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if operation:
            log_data["operation"] = operation

        log.error("exception_caught", **log_data, exc_info=True)

        if reraise:
            raise


def log_function_call(logger: Optional[structlog.BoundLogger] = None) -> Callable:
    """
    Decorator to log function calls with arguments.

    Args:
        logger: Logger instance

    Example:
        @log_function_call()
        def process_batch(batch_id, size):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger(__name__)

            # Log call
            log.debug(
                "function_called",
                function=func.__name__,
                args=args,
                kwargs=kwargs,
            )

            # Execute function
            try:
                result = func(*args, **kwargs)
                log.debug(
                    "function_returned",
                    function=func.__name__,
                )
                return result
            except Exception as e:
                log.error(
                    "function_failed",
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator
