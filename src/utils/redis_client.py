"""
Common Redis Client Factory.

Centralized Redis client creation following DRY principles.
"""

import logging
import redis
from typing import Optional

logger = logging.getLogger(__name__)


def get_redis_client(
    url: Optional[str] = None,
    decode_responses: bool = True,
    **kwargs
) -> redis.Redis:
    """
    Create and return a Redis client.

    Args:
        url: Redis URL (defaults to config from settings)
        decode_responses: Whether to decode responses to strings
        **kwargs: Additional Redis client arguments

    Returns:
        Redis client instance
    """
    from src.config.config import get_config

    if url is None:
        config = get_config()
        url = config.get("celery.broker_url", "redis://redis-broker:6379/6")

    try:
        client = redis.from_url(url, decode_responses=decode_responses, **kwargs)
        # Test connection
        client.ping()
        logger.debug(f"Connected to Redis: {url}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {url}: {e}")
        raise


def get_cache_redis_client(**kwargs) -> redis.Redis:
    """
    Get Redis client for caching (DB 7 by default).

    Args:
        **kwargs: Additional Redis client arguments

    Returns:
        Redis client instance for cache
    """
    from src.config.config import get_config

    config = get_config()
    cache_url = config.get("redis.cache_url", "redis://redis-cache:6379/7")
    return get_redis_client(url=cache_url, **kwargs)


def get_broker_redis_client(**kwargs) -> redis.Redis:
    """
    Get Redis client for Celery broker (DB 6 by default).

    Args:
        **kwargs: Additional Redis client arguments

    Returns:
        Redis client instance for broker
    """
    from src.config.config import get_config

    config = get_config()
    broker_url = config.get("celery.broker_url", "redis://redis-broker:6379/6")
    return get_redis_client(url=broker_url, **kwargs)
