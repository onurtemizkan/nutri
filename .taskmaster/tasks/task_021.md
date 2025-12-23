# Task ID: 21

**Title:** Implement Comprehensive Health Check Endpoints

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Enhance the backend health endpoint to check database and Redis connectivity, and add similar comprehensive checks to the ML service. Support both liveness and readiness probes.

**Details:**

**Update `server/src/index.ts` - Replace simple health check:**
```typescript
import { PrismaClient } from '@prisma/client';
import { Redis } from 'ioredis'; // or use existing redis client

const prisma = new PrismaClient();
const packageJson = require('../package.json');

// Liveness probe - just checks if server is running
app.get('/health/live', (_req, res) => {
  res.status(200).json({ status: 'ok' });
});

// Readiness probe - checks all dependencies
app.get('/health', async (_req, res) => {
  const checks: Record<string, { status: string; latency_ms?: number; error?: string }> = {};
  let overallStatus = 'healthy';

  // Database check
  const dbStart = Date.now();
  try {
    await prisma.$queryRaw`SELECT 1`;
    checks.database = { status: 'healthy', latency_ms: Date.now() - dbStart };
  } catch (error) {
    checks.database = { status: 'unhealthy', error: (error as Error).message };
    overallStatus = 'unhealthy';
  }

  // Redis check (if configured)
  if (process.env.REDIS_URL) {
    const redisStart = Date.now();
    try {
      // Ping redis
      checks.redis = { status: 'healthy', latency_ms: Date.now() - redisStart };
    } catch (error) {
      checks.redis = { status: 'unhealthy', error: (error as Error).message };
      overallStatus = 'degraded'; // Redis is optional
    }
  }

  // ML Service check (optional, don't fail if unavailable)
  if (process.env.ML_SERVICE_URL) {
    const mlStart = Date.now();
    try {
      const response = await fetch(`${process.env.ML_SERVICE_URL}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) {
        checks.ml_service = { status: 'healthy', latency_ms: Date.now() - mlStart };
      } else {
        checks.ml_service = { status: 'degraded', latency_ms: Date.now() - mlStart };
      }
    } catch (error) {
      checks.ml_service = { status: 'unavailable', error: (error as Error).message };
    }
  }

  const statusCode = overallStatus === 'healthy' ? 200 : overallStatus === 'degraded' ? 200 : 503;
  res.status(statusCode).json({
    status: overallStatus,
    version: packageJson.version,
    timestamp: new Date().toISOString(),
    checks,
  });
});
```

**Update ML Service `ml-service/app/main.py`:**
```python
from fastapi import FastAPI
from datetime import datetime
import time
from app.config import settings

@app.get('/health')
async def health_check():
    checks = {}
    overall_status = 'healthy'
    
    # Database check
    db_start = time.time()
    try:
        async with get_db() as db:
            await db.execute('SELECT 1')
        checks['database'] = {'status': 'healthy', 'latency_ms': int((time.time() - db_start) * 1000)}
    except Exception as e:
        checks['database'] = {'status': 'unhealthy', 'error': str(e)}
        overall_status = 'unhealthy'
    
    # Redis check
    redis_start = time.time()
    try:
        await redis_client.ping()
        checks['redis'] = {'status': 'healthy', 'latency_ms': int((time.time() - redis_start) * 1000)}
    except Exception as e:
        checks['redis'] = {'status': 'unhealthy', 'error': str(e)}
        overall_status = 'degraded'
    
    # Model check
    checks['model'] = {'status': 'healthy' if model_loaded else 'unavailable'}
    
    status_code = 200 if overall_status in ['healthy', 'degraded'] else 503
    return JSONResponse(
        status_code=status_code,
        content={
            'status': overall_status,
            'version': settings.VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'checks': checks
        }
    )

@app.get('/health/live')
async def liveness():
    return {'status': 'ok'}
```

**Test Strategy:**

1. Unit test health endpoint with mocked dependencies
2. Integration test with real DB/Redis: verify response structure
3. Test failure scenarios: stop postgres, verify status=unhealthy and 503
4. Test degraded state: stop redis, verify status=degraded and 200
5. Verify /health/live always returns 200
6. Test response latency <100ms under normal conditions
7. Verify version field matches package.json

## Subtasks

### 21.1. Enhance backend /health endpoint with database connectivity check

**Status:** pending  
**Dependencies:** None  

Update server/src/index.ts to add Prisma database connectivity check with latency measurement and proper error handling

**Details:**

Import PrismaClient and use $queryRaw to execute SELECT 1 query. Measure latency using Date.now() before and after query. Wrap in try-catch to handle errors. Return status 'healthy' on success with latency_ms, or 'unhealthy' with error message on failure. Set overall status to 'unhealthy' if database check fails. Import package.json to include version in response.

### 21.2. Add Redis connectivity check with optional dependency handling

**Status:** pending  
**Dependencies:** 21.1  

Implement Redis ping check in /health endpoint with proper error handling for when Redis is not configured or unavailable

**Details:**

Check if REDIS_URL environment variable exists. If configured, initialize Redis client (ioredis) and ping. Measure latency. On success, add redis check with 'healthy' status. On failure, add 'unhealthy' status but set overall status to 'degraded' (not unhealthy) since Redis is optional. Skip Redis check entirely if REDIS_URL not set.

### 21.3. Add ML service reachability check from backend

**Status:** pending  
**Dependencies:** 21.1  

Implement optional ML service health check using fetch with timeout to verify connectivity to ml-service

**Details:**

Check if ML_SERVICE_URL environment variable exists. Use fetch with AbortSignal.timeout(5000) to call ${ML_SERVICE_URL}/health. On successful response (response.ok), mark as 'healthy' with latency. On error or non-ok response, mark as 'unavailable' or 'degraded'. Don't fail overall health check if ML service is down (it's optional).

### 21.4. Implement /health/live liveness probe endpoint

**Status:** pending  
**Dependencies:** None  

Create simple liveness probe endpoint that returns immediate OK status without dependency checks

**Details:**

Add GET /health/live route in server/src/index.ts. Return simple JSON response {status: 'ok'} with 200 status code. No async checks, no database/redis/ml connectivity verification. This endpoint is for container orchestrators to verify the process is running.

### 21.5. Create comprehensive health checks for ML service in Python

**Status:** pending  
**Dependencies:** 21.4  

Implement /health and /health/live endpoints in ml-service FastAPI app with database, Redis, and model status checks

**Details:**

Update ml-service/app/main.py. Add /health/live endpoint returning {'status': 'ok'}. Enhance /health endpoint: 1) Database check using async db.execute('SELECT 1') with latency measurement, 2) Redis check using await redis_client.ping() with latency, 3) Model status check (healthy if model_loaded else unavailable). Set overall_status to 'healthy', 'degraded' (if Redis fails), or 'unhealthy' (if DB fails). Return 200 for healthy/degraded, 503 for unhealthy. Include version from settings.VERSION and timestamp.
