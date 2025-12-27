"""
Inter-stage communication services.

This module contains services for consuming events from upstream stages
(Stage 3) and publishing events to downstream stages (Stage 5).
"""

from src.services.stage3_stream_consumer import Stage3StreamConsumer

__all__ = ["Stage3StreamConsumer"]
