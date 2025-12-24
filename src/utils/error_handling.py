"""
Error Handling Module

Provides comprehensive error handling infrastructure including:
- Custom exception hierarchy
- Retry decorator with exponential backoff
- Circuit breaker pattern
- Error tracking and alerting
"""

import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar, Union

import structlog
from pydantic import BaseModel


logger = structlog.get_logger(__name__)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class ClusteringServiceError(Exception):
    """Base exception for all clustering service errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# Configuration Errors
class ConfigurationError(ClusteringServiceError):
    """Error in service configuration."""
    pass


# Storage Errors
class StorageError(ClusteringServiceError):
    """Base class for storage-related errors."""
    pass


class DatabaseError(StorageError):
    """PostgreSQL database error."""
    pass


class RedisError(StorageError):
    """Redis connection or operation error."""
    pass


class FileStorageError(StorageError):
    """File system storage error."""
    pass


# Job Lifecycle Errors
class JobError(ClusteringServiceError):
    """Base class for job-related errors."""
    pass


class JobNotFoundError(JobError):
    """Job does not exist."""
    pass


class JobStateError(JobError):
    """Invalid job state transition."""
    pass


class JobCancelledError(JobError):
    """Job was cancelled."""
    pass


class JobTimeoutError(JobError):
    """Job exceeded timeout."""
    pass


class CheckpointError(JobError):
    """Error saving/loading checkpoint."""
    pass


# Clustering Errors
class ClusteringError(ClusteringServiceError):
    """Base class for clustering algorithm errors."""
    pass


class InvalidAlgorithmError(ClusteringError):
    """Unknown or unsupported clustering algorithm."""
    pass


class ClusteringFailedError(ClusteringError):
    """Clustering algorithm failed to converge."""
    pass


class InsufficientDataError(ClusteringError):
    """Not enough data points for clustering."""
    pass


# FAISS/Vector Errors
class VectorError(ClusteringServiceError):
    """Base class for vector operation errors."""
    pass


class IndexNotFoundError(VectorError):
    """FAISS index not found."""
    pass


class IndexLoadError(VectorError):
    """Failed to load FAISS index."""
    pass


class VectorSearchError(VectorError):
    """Error during vector search."""
    pass


# GPU Errors
class GPUError(ClusteringServiceError):
    """GPU-related error."""
    pass


class GPUMemoryError(GPUError):
    """GPU out of memory."""
    pass


class GPUNotAvailableError(GPUError):
    """GPU not available or not configured."""
    pass


# External Service Errors
class ExternalServiceError(ClusteringServiceError):
    """Error communicating with external service."""
    pass


class Stage3IntegrationError(ExternalServiceError):
    """Error communicating with Stage 3 embedding service."""
    pass


# Resource Management Errors
class ResourceError(ClusteringServiceError):
    """Resource management error."""
    pass


class ResourceExhaustedError(ResourceError):
    """Resource limit exceeded."""
    pass


class ResourceLockError(ResourceError):
    """Failed to acquire resource lock."""
    pass


# =============================================================================
# Retry Decorator with Exponential Backoff
# =============================================================================


T = TypeVar("T")


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retriable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        RedisError,
        DatabaseError,
        ExternalServiceError,
    )

    class Config:
        arbitrary_types_allowed = True


def retry(
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    jitter: Optional[bool] = None,
    retriable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        config: RetryConfig object (overrides individual params)
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff multiplier
        jitter: Add random jitter to delays
        retriable_exceptions: Tuple of exception types to retry

    Example:
        @retry(max_attempts=5, initial_delay=2.0)
        def risky_operation():
            ...
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 3,
            initial_delay=initial_delay or 1.0,
            max_delay=max_delay or 60.0,
            backoff_factor=backoff_factor or 2.0,
            jitter=jitter if jitter is not None else True,
            retriable_exceptions=retriable_exceptions or RetryConfig().retriable_exceptions,
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = config.initial_delay

            while attempt < config.max_attempts:
                try:
                    return func(*args, **kwargs)
                except config.retriable_exceptions as e:
                    attempt += 1

                    if attempt >= config.max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        raise

                    # Calculate delay with exponential backoff
                    current_delay = min(delay, config.max_delay)

                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        current_delay *= (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay=current_delay,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                    time.sleep(current_delay)
                    delay *= config.backoff_factor

            # This should never be reached, but satisfy type checker
            raise RuntimeError("Retry logic error")

        return wrapper

    return decorator


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Prevents cascading failures by opening circuit after N failures
    and periodically testing recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "circuit_breaker",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

        self._logger = structlog.get_logger(__name__)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            ResourceError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self._logger.info(
                    "circuit_breaker_half_open",
                    name=self.name,
                )
            else:
                raise ResourceError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    details={
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time,
                    },
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self._logger.info(
                "circuit_breaker_closed",
                name=self.name,
                previous_failures=self.failure_count,
            )

        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._logger.error(
                "circuit_breaker_opened",
                name=self.name,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._logger.info("circuit_breaker_reset", name=self.name)

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN


# =============================================================================
# Error Tracker
# =============================================================================


class ErrorTracker:
    """
    Track errors and generate alerts when thresholds exceeded.

    Thread-safe error tracking with time-window-based alerting.
    """

    def __init__(
        self,
        window_size: float = 300.0,  # 5 minutes
        alert_threshold: int = 10,
    ):
        """
        Initialize error tracker.

        Args:
            window_size: Time window in seconds
            alert_threshold: Number of errors before alert
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.errors: list[tuple[float, Exception]] = []
        self._logger = structlog.get_logger(__name__)

    def record(self, error: Exception) -> None:
        """
        Record an error occurrence.

        Args:
            error: Exception that occurred
        """
        current_time = time.time()
        self.errors.append((current_time, error))

        # Remove old errors outside window
        self._clean_old_errors(current_time)

        # Check if alert threshold exceeded
        if len(self.errors) >= self.alert_threshold:
            self._trigger_alert()

    def _clean_old_errors(self, current_time: float) -> None:
        """Remove errors outside time window."""
        cutoff_time = current_time - self.window_size
        self.errors = [
            (ts, err) for ts, err in self.errors if ts >= cutoff_time
        ]

    def _trigger_alert(self) -> None:
        """Trigger alert when threshold exceeded."""
        error_types = {}
        for _, error in self.errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1

        self._logger.error(
            "error_threshold_exceeded",
            error_count=len(self.errors),
            threshold=self.alert_threshold,
            window_size=self.window_size,
            error_types=error_types,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        current_time = time.time()
        self._clean_old_errors(current_time)

        error_types = {}
        for _, error in self.errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "window_size": self.window_size,
            "alert_threshold": self.alert_threshold,
        }

    def reset(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()
        self._logger.info("error_tracker_reset")


# =============================================================================
# Utility Functions
# =============================================================================


def handle_exceptions(
    *exception_types: Type[Exception],
    default_return: Optional[Any] = None,
    log_errors: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:
    """
    Decorator to catch and handle specific exceptions.

    Args:
        *exception_types: Exception types to catch
        default_return: Value to return on exception
        log_errors: Whether to log caught exceptions

    Example:
        @handle_exceptions(ValueError, KeyError, default_return=None)
        def parse_data(data):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_errors:
                    logger.error(
                        "exception_handled",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                return default_return

        return wrapper

    return decorator
