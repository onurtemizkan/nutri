# Task ID: 19

**Title:** Enhance ML Service Dockerfile for Production

**Status:** in-progress

**Dependencies:** None

**Priority:** high

**Description:** Update the existing ML service Dockerfile to run as non-root user and optimize for production deployment with proper security configurations.

**Details:**

Update `ml-service/Dockerfile` to add non-root user support:

**Modifications to existing Dockerfile:**
```dockerfile
# Stage 2: Runtime (update existing)
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system --gid 1001 mlservice && \
    useradd --system --uid 1001 --gid mlservice mlservice

# Copy Python dependencies from builder (update ownership)
COPY --from=builder --chown=mlservice:mlservice /root/.local /home/mlservice/.local

# Update PATH for non-root user
ENV PATH=/home/mlservice/.local/bin:$PATH

# Copy application code
COPY --chown=mlservice:mlservice ./app /app/app

# Create directories for ML models with correct ownership
RUN mkdir -p /app/app/ml_models && chown -R mlservice:mlservice /app

# Switch to non-root user
USER mlservice

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Create `ml-service/.dockerignore`:**
```
__pycache__
*.pyc
*.pyo
.pytest_cache
.mypy_cache
venv
.env*
*.log
coverage
tests
*.md
.git
```

**Key changes:**
- Add non-root user 'mlservice'
- Set PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED
- Proper ownership of files
- Keep existing health check and multi-stage build

**Test Strategy:**

1. Build image: `cd ml-service && docker build -t nutri-ml:test .`
2. Verify non-root: `docker run --rm nutri-ml:test whoami` (should output 'mlservice')
3. Test health endpoint: `docker run -d -p 8000:8000 --name test-ml nutri-ml:test && sleep 5 && curl http://localhost:8000/health`
4. Check container health: `docker inspect --format='{{.State.Health.Status}}' test-ml`
5. Verify image size is reasonable (target <1.5GB due to ML dependencies)

## Subtasks

### 19.1. Modify Dockerfile to add non-root user with proper ownership

**Status:** pending  
**Dependencies:** None  

Update the existing ml-service/Dockerfile to create and configure a non-root user 'mlservice' with proper file ownership, PATH configuration, and security best practices.

**Details:**

Update the Dockerfile runtime stage to:
1. Create system group 'mlservice' (GID 1001) and user 'mlservice' (UID 1001)
2. Copy Python dependencies from builder with --chown=mlservice:mlservice to /home/mlservice/.local
3. Update PATH environment variable to include /home/mlservice/.local/bin
4. Copy application code with --chown=mlservice:mlservice
5. Create /app/app/ml_models directory with proper ownership (chown -R mlservice:mlservice /app)
6. Switch to non-root user with USER mlservice directive
7. Ensure existing health check, multi-stage build, and uvicorn CMD remain functional
8. Keep existing runtime dependencies (libpq5) installation

### 19.2. Create .dockerignore and add production environment variables

**Status:** pending  
**Dependencies:** 19.1  

Create a .dockerignore file to exclude unnecessary files from the Docker build context and add PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED environment variables for production security and performance.

**Details:**

1. Create ml-service/.dockerignore with exclusions:
   - Python cache files: __pycache__, *.pyc, *.pyo
   - Test and coverage directories: .pytest_cache, .mypy_cache, coverage, tests
   - Virtual environments and configs: venv, .env*, *.log
   - Documentation and version control: *.md, .git
2. Add environment variables to Dockerfile:
   - PYTHONDONTWRITEBYTECODE=1 (prevents .pyc file generation)
   - PYTHONUNBUFFERED=1 (ensures real-time logging)
3. Verify existing EXPOSE 8000 and HEALTHCHECK directives remain intact
4. Ensure CMD uvicorn with --workers 4 configuration is preserved
