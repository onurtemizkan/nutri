# =============================================================================
# Nutri ML Service - Production Dockerfile
# Multi-stage build for optimized Python ML service with security hardening
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# Install dependencies and compile any native extensions
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# Minimal production image with security hardening
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --system --gid 1001 mlservice && \
    useradd --system --uid 1001 --gid mlservice mlservice

# Copy Python dependencies from builder with proper ownership
COPY --from=builder --chown=mlservice:mlservice /root/.local /home/mlservice/.local

# Update PATH for non-root user
ENV PATH=/home/mlservice/.local/bin:$PATH

# Python production optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code with proper ownership
COPY --chown=mlservice:mlservice ./app /app/app

# Create directories for ML models with correct ownership
RUN mkdir -p /app/app/ml_models && chown -R mlservice:mlservice /app

# Switch to non-root user
USER mlservice

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
