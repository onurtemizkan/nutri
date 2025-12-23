import express from 'express';
import cors from 'cors';
import { config } from './config/env';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger, correlationIdHeader } from './middleware/requestLogger';
import { logger } from './config/logger';
import authRoutes from './routes/authRoutes';
import mealRoutes from './routes/mealRoutes';
import healthMetricRoutes from './routes/healthMetricRoutes';
import activityRoutes from './routes/activityRoutes';
import foodAnalysisRoutes from './routes/foodAnalysisRoutes';
import supplementRoutes from './routes/supplementRoutes';
import foodRoutes from './routes/foodRoutes';
import foodFeedbackRoutes from './routes/foodFeedbackRoutes';
import adminRoutes from './routes/admin';
import prisma from './config/database';

// Import version from package.json
const packageJson = require('../package.json');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging (adds correlation ID to req.id)
app.use(requestLogger);
app.use(correlationIdHeader);

// =============================================================================
// Health Check Endpoints
// =============================================================================

interface HealthCheck {
  status: 'healthy' | 'unhealthy' | 'degraded' | 'unavailable';
  latency_ms?: number;
  error?: string;
}

// Liveness probe - just checks if server is running (for container orchestrators)
app.get('/health/live', (_req, res) => {
  res.status(200).json({ status: 'ok' });
});

// Readiness probe - comprehensive health check with all dependencies
app.get('/health', async (_req, res) => {
  const checks: Record<string, HealthCheck> = {};
  let overallStatus: 'healthy' | 'unhealthy' | 'degraded' = 'healthy';

  // Database check
  const dbStart = Date.now();
  try {
    await prisma.$queryRaw`SELECT 1`;
    checks.database = { status: 'healthy', latency_ms: Date.now() - dbStart };
  } catch (error) {
    checks.database = {
      status: 'unhealthy',
      latency_ms: Date.now() - dbStart,
      error: error instanceof Error ? error.message : 'Unknown database error',
    };
    overallStatus = 'unhealthy';
  }

  // Redis check (if configured)
  if (process.env.REDIS_URL) {
    const redisStart = Date.now();
    try {
      // For now, just mark as configured but not checked
      // Full Redis implementation would use ioredis client
      checks.redis = {
        status: 'healthy',
        latency_ms: Date.now() - redisStart,
      };
    } catch (error) {
      checks.redis = {
        status: 'degraded',
        latency_ms: Date.now() - redisStart,
        error: error instanceof Error ? error.message : 'Unknown redis error',
      };
      if (overallStatus === 'healthy') {
        overallStatus = 'degraded';
      }
    }
  }

  // ML Service check (optional, don't fail if unavailable)
  const mlServiceUrl = process.env.ML_SERVICE_URL;
  if (mlServiceUrl) {
    const mlStart = Date.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${mlServiceUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        checks.ml_service = { status: 'healthy', latency_ms: Date.now() - mlStart };
      } else {
        checks.ml_service = {
          status: 'degraded',
          latency_ms: Date.now() - mlStart,
          error: `HTTP ${response.status}`,
        };
      }
    } catch (error) {
      checks.ml_service = {
        status: 'unavailable',
        latency_ms: Date.now() - mlStart,
        error: error instanceof Error ? error.message : 'ML service unreachable',
      };
      // ML service being unavailable doesn't affect overall health
    }
  }

  const statusCode = overallStatus === 'unhealthy' ? 503 : 200;
  res.status(statusCode).json({
    status: overallStatus,
    version: packageJson.version,
    timestamp: new Date().toISOString(),
    checks,
  });
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/meals', mealRoutes);
app.use('/api/health-metrics', healthMetricRoutes);
app.use('/api/activities', activityRoutes);
app.use('/api/food', foodAnalysisRoutes);
app.use('/api/foods', foodRoutes);
app.use('/api/foods/feedback', foodFeedbackRoutes);
app.use('/api/supplements', supplementRoutes);
app.use('/api/admin', adminRoutes);

// Error handler (must be last)
app.use(errorHandler);

// Only start server if not in test mode
// In test mode, supertest will handle server lifecycle
if (process.env.NODE_ENV !== 'test') {
  const PORT = config.port;

  app.listen(PORT, () => {
    logger.info({
      port: PORT,
      env: config.nodeEnv,
      version: packageJson.version,
      healthCheck: `http://localhost:${PORT}/health`,
    }, 'Server started');
  });
}

export default app;
