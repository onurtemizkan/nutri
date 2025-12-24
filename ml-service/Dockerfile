# =============================================================================
# Nutri ML Service - Production Dockerfile
# Optimized for AMD EPYC CPUs (Hetzner Cloud/Dedicated Servers)
#
# Build: docker build -t nutri-ml-service:latest .
# Run:   docker run -p 8000:8000 --env-file .env nutri-ml-service:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# Install dependencies and compile native extensions
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# Minimal production image optimized for AMD EPYC inference
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --system --gid 1001 mlservice && \
    useradd --system --uid 1001 --gid mlservice --home /home/mlservice --create-home mlservice

# Copy Python dependencies from builder
COPY --from=builder --chown=mlservice:mlservice /root/.local /home/mlservice/.local

# Update PATH for non-root user
ENV PATH=/home/mlservice/.local/bin:$PATH

# =============================================================================
# AMD EPYC CPU Optimizations
# =============================================================================
# These settings optimize PyTorch and ONNX Runtime for AMD EPYC processors
# Adjust OMP_NUM_THREADS based on your instance (CPX31=4, CCX23=4, etc.)

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # -------------------------------------------------------------------------
    # Thread Configuration (adjust based on vCPU count)
    # CPX21: 3 vCPU -> OMP_NUM_THREADS=3
    # CPX31: 4 vCPU -> OMP_NUM_THREADS=4
    # CCX13: 2 dedicated -> OMP_NUM_THREADS=2
    # CCX23: 4 dedicated -> OMP_NUM_THREADS=4
    # -------------------------------------------------------------------------
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    # -------------------------------------------------------------------------
    # PyTorch CPU Optimizations
    # -------------------------------------------------------------------------
    TORCH_NUM_THREADS=4 \
    # Disable GPU search (CPU-only deployment)
    CUDA_VISIBLE_DEVICES="" \
    # -------------------------------------------------------------------------
    # ONNX Runtime Optimizations
    # -------------------------------------------------------------------------
    ORT_DISABLE_ALL_TELEMETRY=1 \
    # -------------------------------------------------------------------------
    # Memory Optimizations
    # -------------------------------------------------------------------------
    MALLOC_TRIM_THRESHOLD_=131072 \
    # -------------------------------------------------------------------------
    # HuggingFace Cache (will be mounted as volume in production)
    # -------------------------------------------------------------------------
    HF_HOME=/home/mlservice/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/mlservice/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0 \
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    COMPUTE_DEVICE=cpu \
    FAST_MODE=true \
    ENVIRONMENT=production

# Copy application code
COPY --chown=mlservice:mlservice ./app /app/app

# Create directories for ML models and HuggingFace cache
RUN mkdir -p /app/app/ml_models /home/mlservice/.cache/huggingface && \
    chown -R mlservice:mlservice /app /home/mlservice/.cache

# Switch to non-root user
USER mlservice

# Expose port
EXPOSE 8000

# Health check disabled for deployment - Coolify has its own health monitoring
# HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
#     CMD curl -f http://localhost:8000/health/live || exit 1

# -----------------------------------------------------------------------------
# Run Configuration
# - workers=2: Good balance for 4 vCPU (each worker uses ~2 threads)
# - timeout=300: Allow time for model loading on cold start
# - For CPX21 (3 vCPU): use --workers 1
# - For CCX23 (4 dedicated): can try --workers 2
# -----------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "300"]
