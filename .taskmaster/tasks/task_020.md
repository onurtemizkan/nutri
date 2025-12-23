# Task ID: 20

**Title:** Create Production Docker Compose Configuration

**Status:** pending

**Dependencies:** 18 ✓, 19 ⧖

**Priority:** high

**Description:** Create docker-compose.prod.yml that mirrors production environment for local testing, including all services with production-like settings and .env.example documentation.

**Details:**

**Create `docker-compose.prod.yml`:**
```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./server
      dockerfile: Dockerfile
    container_name: nutri-backend
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
      - JWT_EXPIRES_IN=${JWT_EXPIRES_IN:-7d}
      - PORT=3000
      - REDIS_URL=${REDIS_URL}
      - ML_SERVICE_URL=http://ml-service:8000
    ports:
      - "3000:3000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "node", "-e", "require('http').get('http://localhost:3000/health', r => process.exit(r.statusCode === 200 ? 0 : 1))"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - nutri-network
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  ml-service:
    build:
      context: ./ml-service
      dockerfile: Dockerfile
    container_name: nutri-ml-service
    restart: unless-stopped
    environment:
      - DATABASE_URL=${ML_DATABASE_URL:-postgresql+asyncpg://postgres:postgres@postgres:5432/nutri_db}
      - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
      - ENVIRONMENT=production
      - DEBUG=false
      - LOG_LEVEL=INFO
    expose:
      - "8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - nutri-network
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    image: postgres:16-alpine
    container_name: nutri-postgres-prod
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB:-nutri_db}
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nutri-network

  redis:
    image: redis:7-alpine
    container_name: nutri-redis-prod
    restart: unless-stopped
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_prod_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - nutri-network

volumes:
  postgres_prod_data:
  redis_prod_data:

networks:
  nutri-network:
    driver: bridge
```

**Create/Update `.env.example`:**
```env
# Database (Required)
DATABASE_URL=postgresql://postgres:password@localhost:5432/nutri_db
ML_DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/nutri_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=nutri_db

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT (Required)
JWT_SECRET=your-256-bit-secret-key-change-in-production
JWT_EXPIRES_IN=7d

# Environment
NODE_ENV=production
```

**Test Strategy:**

1. Copy `.env.example` to `.env.prod` and fill in test values
2. Run: `docker-compose -f docker-compose.prod.yml --env-file .env.prod up --build`
3. Verify all services start: `docker-compose -f docker-compose.prod.yml ps`
4. Test backend: `curl http://localhost:3000/health`
5. Verify ML service is only internally accessible (not on host port)
6. Test backend can reach ML service internally
7. Check logs: `docker-compose -f docker-compose.prod.yml logs backend`

## Subtasks

### 20.1. Create docker-compose.prod.yml with backend service configuration

**Status:** pending  
**Dependencies:** None  

Create the production Docker Compose file with backend service including health checks, restart policies, environment variables, port mappings, and dependency configuration on postgres and redis services.

**Details:**

Create `docker-compose.prod.yml` in the project root with version 3.8. Configure the backend service with:
- Build context pointing to ./server with Dockerfile
- Container name: nutri-backend
- Restart policy: unless-stopped
- Environment variables: NODE_ENV=production, DATABASE_URL, JWT_SECRET, JWT_EXPIRES_IN (default 7d), PORT=3000, REDIS_URL, ML_SERVICE_URL=http://ml-service:8000
- Port mapping: 3000:3000
- Dependencies on postgres and redis with health check conditions
- Health check using Node.js HTTP request to /health endpoint (interval: 30s, timeout: 10s, retries: 3)
- Network: nutri-network
- JSON file logging with max-size 10m and max-file 3

### 20.2. Add ML service with internal networking and resource configuration

**Status:** pending  
**Dependencies:** 20.1  

Configure the ml-service in docker-compose.prod.yml with internal-only networking (no host port exposure), proper dependencies, health checks, and production-appropriate settings.

**Details:**

Add ml-service to docker-compose.prod.yml:
- Build context: ./ml-service with Dockerfile
- Container name: nutri-ml-service
- Restart policy: unless-stopped
- Environment: DATABASE_URL (ML_DATABASE_URL with default), REDIS_URL (with default), ENVIRONMENT=production, DEBUG=false, LOG_LEVEL=INFO
- Expose port 8000 internally only (no ports mapping to host)
- Dependencies on postgres and redis with health check conditions
- Health check using Python urllib to call /health endpoint (interval: 30s, timeout: 10s, retries: 3, start_period: 40s)
- Network: nutri-network
- JSON file logging with max-size 10m and max-file 3

Ensure ML service is only accessible from backend service via internal network.

### 20.3. Configure postgres and redis services with production volumes and health checks

**Status:** pending  
**Dependencies:** 20.1  

Set up postgres and redis services in docker-compose.prod.yml with named volumes for data persistence, health checks, restart policies, and production-optimized configurations.

**Details:**

Add postgres service:
- Image: postgres:16-alpine
- Container name: nutri-postgres-prod
- Restart policy: unless-stopped
- Environment: POSTGRES_USER (default postgres), POSTGRES_PASSWORD, POSTGRES_DB (default nutri_db)
- Volume: postgres_prod_data:/var/lib/postgresql/data
- Health check: pg_isready command (interval: 10s, timeout: 5s, retries: 5)
- Network: nutri-network

Add redis service:
- Image: redis:7-alpine
- Container name: nutri-redis-prod
- Restart policy: unless-stopped
- Command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
- Volume: redis_prod_data:/data
- Health check: redis-cli ping (interval: 10s, timeout: 5s, retries: 5)
- Network: nutri-network

Define volumes section with postgres_prod_data and redis_prod_data.
Define networks section with nutri-network using bridge driver.

### 20.4. Create/update .env.example with comprehensive documentation

**Status:** pending  
**Dependencies:** 20.1, 20.2, 20.3  

Create or update .env.example file with all required environment variables, sensible defaults, security notes, and clear documentation for production deployment configuration.

**Details:**

Create `.env.example` in project root with documented sections:

**Database (Required):**
- DATABASE_URL=postgresql://postgres:password@localhost:5432/nutri_db
- ML_DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/nutri_db
- POSTGRES_USER=postgres
- POSTGRES_PASSWORD=your-secure-password (with security note)
- POSTGRES_DB=nutri_db

**Redis:**
- REDIS_URL=redis://localhost:6379/0

**JWT (Required):**
- JWT_SECRET=your-256-bit-secret-key-change-in-production (with strong security warning)
- JWT_EXPIRES_IN=7d

**Environment:**
- NODE_ENV=production

Include comments explaining each variable's purpose, required vs optional status, and security implications. Add header with instructions to copy to .env.prod for local testing.
