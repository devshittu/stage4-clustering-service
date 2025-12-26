# =============================================================================
# Stage 4 Clustering Service - Multi-Stage Dockerfile
# =============================================================================
# Optimized for production deployment with GPU support (NVIDIA RTX A4000)
# Python 3.11, FAISS-GPU, HDBSCAN, scikit-learn
# =============================================================================

# -----------------------------------------------------------------------------
# Base Stage: Common dependencies
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create app directory and user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /app/data /app/logs /app/config && \
    chown -R appuser:appuser /app

WORKDIR /app

# -----------------------------------------------------------------------------
# Dependencies Stage: Install Python packages
# -----------------------------------------------------------------------------
FROM base AS dependencies

# Copy requirements file
COPY requirements-simple.txt requirements.txt

# Install FAISS-GPU first (with fallback to CPU)
RUN pip install --no-cache-dir faiss-gpu==1.7.2 || \
    pip install --no-cache-dir faiss-cpu==1.7.4 || \
    echo "WARNING: FAISS installation failed, will retry with requirements.txt"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Production Stage: Final image
# -----------------------------------------------------------------------------
FROM base AS production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p \
    /app/data/indices \
    /app/data/output \
    /app/data/clusters \
    /app/logs \
    /app/config \
    /shared/stage3/data/vector_indices

# Set ownership
RUN chown -R appuser:appuser /app /shared

# Switch to non-root user
USER appuser

# Expose port (internal only, Traefik handles external routing)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (orchestrator)
CMD ["uvicorn", "src.api.orchestrator:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# -----------------------------------------------------------------------------
# Development Stage: For local development
# -----------------------------------------------------------------------------
FROM production AS development

# Switch back to root to install dev tools
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    debugpy \
    watchfiles

# Switch back to appuser
USER appuser

# Enable hot reload for development
CMD ["uvicorn", "src.api.orchestrator:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
