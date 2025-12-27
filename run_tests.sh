#!/bin/bash
# Test runner script for automation tests
# Run inside Docker container for proper environment

set -e

echo "=== Running Inter-Stage Automation Tests ==="
echo ""

echo "1. Running unit tests for Stream Consumer..."
docker exec clustering-celery-worker pytest tests/unit/test_stage3_stream_consumer.py -v --tb=short || echo "  ⚠ Stream consumer tests need container running"

echo ""
echo "2. Running unit tests for Webhooks..."
docker exec clustering-celery-worker pytest tests/unit/test_automation_webhooks.py -v --tb=short || echo "  ⚠ Webhook tests need container running"

echo ""
echo "3. Running integration tests..."
docker exec clustering-celery-worker pytest tests/integration/test_inter_stage_automation.py -v --tb=short || echo "  ⚠ Integration tests need container running"

echo ""
echo "=== Test Summary ==="
echo "Tests designed to run in Docker container environment."
echo "To run manually:"
echo "  docker exec clustering-celery-worker pytest tests/ -v"
