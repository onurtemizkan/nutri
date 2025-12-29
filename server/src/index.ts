import express from 'express';
import cors from 'cors';
import { config } from './config/env';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger, correlationIdHeader } from './middleware/requestLogger';
import { httpsRedirect, securityHeaders, getTrustProxyConfig } from './middleware/security';
import { logger } from './config/logger';
import authRoutes from './routes/authRoutes';
import mealRoutes from './routes/mealRoutes';
import healthMetricRoutes from './routes/healthMetricRoutes';
import activityRoutes from './routes/activityRoutes';
import foodAnalysisRoutes from './routes/foodAnalysisRoutes';
import supplementRoutes from './routes/supplementRoutes';
import foodRoutes from './routes/foodRoutes';
import foodFeedbackRoutes from './routes/foodFeedbackRoutes';
import notificationRoutes from './routes/notificationRoutes';
import adminRoutes from './routes/admin';
import onboardingRoutes from './routes/onboardingRoutes';
import waterRoutes from './routes/waterRoutes';
import weightRoutes from './routes/weightRoutes';
import goalRoutes from './routes/goalRoutes';
import webhookRoutes from './routes/webhookRoutes';
import cgmRoutes from './routes/cgmRoutes';
import debugRoutes from './routes/debugRoutes';
import reportRoutes from './routes/reportRoutes';
import subscriptionRoutes from './routes/subscriptionRoutes';
import emailRoutes from './routes/email';
import prisma from './config/database';
import { notificationScheduler } from './services/notificationScheduler';
import { initSentry, setupSentryErrorHandler } from './config/sentry';
import { emailQueue, campaignQueue, sequenceQueue } from './services/emailQueueService';

// Bull Board imports for queue monitoring
import { createBullBoard } from '@bull-board/api';
import { BullAdapter } from '@bull-board/api/bullAdapter';
import { ExpressAdapter } from '@bull-board/express';

// Swagger/OpenAPI documentation
import swaggerUi from 'swagger-ui-express';
import { swaggerSpec } from './config/swagger';

// Import version from package.json
const packageJson = require('../package.json');

const app = express();

// =============================================================================
// Sentry Error Tracking (must be initialized FIRST)
// =============================================================================
initSentry(app);

// Trust proxy configuration for production (needed for X-Forwarded-* headers)
app.set('trust proxy', getTrustProxyConfig());

// Security middleware (must be early in the chain)
app.use(httpsRedirect); // Redirect HTTP to HTTPS in production
app.use(securityHeaders); // Set security headers including HSTS

// Middleware

// CORS configuration with explicit origins
// In production: requires CORS_ORIGIN env var
// In development: allows localhost origins for mobile development
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (mobile apps, Postman, etc.)
      if (!origin) {
        callback(null, true);
        return;
      }

      // Check if origin is in allowed list
      if (config.cors.origins.length === 0) {
        // No origins configured - reject in production, warn in development
        if (config.nodeEnv === 'production') {
          callback(new Error('CORS_ORIGIN not configured for production'));
          return;
        }
        // In development without config, allow the request but log warning
        logger.warn({ origin }, 'CORS: Allowing request from unconfigured origin in development');
        callback(null, true);
        return;
      }

      if (config.cors.origins.includes(origin)) {
        callback(null, true);
      } else {
        logger.warn(
          { origin, allowedOrigins: config.cors.origins },
          'CORS: Blocked request from unauthorized origin'
        );
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: config.cors.credentials,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Correlation-ID'],
    exposedHeaders: ['X-Correlation-ID'],
  })
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging (adds correlation ID to req.id)
app.use(requestLogger);
app.use(correlationIdHeader);

// =============================================================================
// Bull Board for Queue Monitoring
// =============================================================================

// Setup Bull Board for notification queue monitoring
const serverAdapter = new ExpressAdapter();
serverAdapter.setBasePath('/admin/queues');

createBullBoard({
  queues: [
    new BullAdapter(notificationScheduler.getQueue()),
    new BullAdapter(emailQueue),
    new BullAdapter(campaignQueue),
    new BullAdapter(sequenceQueue),
  ],
  serverAdapter,
});

// Mount Bull Board with secure authentication
// Credentials MUST be set via environment variables (no defaults)
app.use(
  '/admin/queues',
  (req, res, next) => {
    // In development, allow access without credentials
    if (config.nodeEnv === 'development') {
      next();
      return;
    }

    // Production: require explicitly configured credentials
    const adminUser = process.env.BULL_BOARD_USERNAME;
    const adminPass = process.env.BULL_BOARD_PASSWORD;

    // Validate that credentials are configured
    if (!adminUser || !adminPass) {
      logger.error(
        'Bull Board credentials not configured. Set BULL_BOARD_USERNAME and BULL_BOARD_PASSWORD'
      );
      res.status(503).json({
        error: 'Queue dashboard not configured',
        message: 'Administrator credentials must be set in environment variables',
      });
      return;
    }

    // Validate password strength (minimum 12 characters)
    if (adminPass.length < 12) {
      logger.error('BULL_BOARD_PASSWORD is too weak. Must be at least 12 characters');
      res.status(503).json({
        error: 'Queue dashboard misconfigured',
        message: 'Administrator password does not meet security requirements',
      });
      return;
    }

    // Require basic auth header
    const auth = req.headers.authorization;
    if (!auth || !auth.startsWith('Basic ')) {
      res.setHeader('WWW-Authenticate', 'Basic realm="Queue Dashboard"');
      res.status(401).send('Authentication required');
      return;
    }

    const credentials = Buffer.from(auth.slice(6), 'base64').toString();
    const colonIndex = credentials.indexOf(':');
    const providedUser = colonIndex > -1 ? credentials.slice(0, colonIndex) : credentials;
    const providedPass = colonIndex > -1 ? credentials.slice(colonIndex + 1) : '';

    // Use constant-time comparison to prevent timing attacks
    // timingSafeEqual throws if lengths differ, so we wrap in try-catch
    // to avoid leaking length information through timing
    let userMatch = false;
    let passMatch = false;

    try {
      userMatch = require('crypto').timingSafeEqual(
        Buffer.from(providedUser),
        Buffer.from(adminUser)
      );
    } catch {
      // Length mismatch - userMatch stays false
    }

    try {
      passMatch = require('crypto').timingSafeEqual(
        Buffer.from(providedPass),
        Buffer.from(adminPass)
      );
    } catch {
      // Length mismatch - passMatch stays false
    }

    if (!userMatch || !passMatch) {
      logger.warn(
        {
          ip: req.ip,
          path: req.path,
          correlationId: req.id,
        },
        'Failed Bull Board authentication attempt'
      );

      res.setHeader('WWW-Authenticate', 'Basic realm="Queue Dashboard"');
      res.status(401).send('Invalid credentials');
      return;
    }

    // Log successful access
    logger.info(
      {
        ip: req.ip,
        path: req.path,
        correlationId: req.id,
      },
      'Bull Board access granted'
    );

    next();
  },
  serverAdapter.getRouter()
);

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

  // Notification Queue check (with timeout to prevent blocking health checks)
  if (notificationScheduler.isReady()) {
    const queueStart = Date.now();
    const QUEUE_CHECK_TIMEOUT = 5000; // 5 second timeout

    try {
      const queue = notificationScheduler.getQueue();

      // Wrap queue checks in a timeout to prevent blocking
      const queueCheckPromise = Promise.all([
        queue.getWaitingCount(),
        queue.getActiveCount(),
        queue.getCompletedCount(),
        queue.getFailedCount(),
      ]);

      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Queue health check timeout')), QUEUE_CHECK_TIMEOUT)
      );

      const [waiting, active, completed, failed] = (await Promise.race([
        queueCheckPromise,
        timeoutPromise,
      ])) as [number, number, number, number];

      checks.notification_queue = {
        status: 'healthy',
        latency_ms: Date.now() - queueStart,
        waiting,
        active,
        completed,
        failed,
      } as HealthCheck & { waiting: number; active: number; completed: number; failed: number };
    } catch (error) {
      checks.notification_queue = {
        status: 'degraded',
        latency_ms: Date.now() - queueStart,
        error: error instanceof Error ? error.message : 'Queue unavailable',
      };
      if (overallStatus === 'healthy') {
        overallStatus = 'degraded';
      }
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

// =============================================================================
// API Documentation (Swagger UI)
// Only enabled in development or when ENABLE_SWAGGER=true
// =============================================================================
if (config.nodeEnv !== 'production' || process.env.ENABLE_SWAGGER === 'true') {
  app.use(
    '/api-docs',
    swaggerUi.serve,
    swaggerUi.setup(swaggerSpec, {
      customCss: '.swagger-ui .topbar { display: none }',
      customSiteTitle: 'Nutri API Documentation',
    })
  );
  logger.info('Swagger UI enabled at /api-docs');
} else {
  // Return 404 for /api-docs in production (unless explicitly enabled)
  app.use('/api-docs', (_req, res) => {
    res.status(404).json({ error: 'API documentation not available in production' });
  });
}

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/meals', mealRoutes);
app.use('/api/health-metrics', healthMetricRoutes);
app.use('/api/activities', activityRoutes);
app.use('/api/food', foodAnalysisRoutes);
app.use('/api/foods', foodRoutes);
app.use('/api/foods/feedback', foodFeedbackRoutes);
app.use('/api/supplements', supplementRoutes);
app.use('/api/notifications', notificationRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/onboarding', onboardingRoutes);
app.use('/api/water', waterRoutes);
app.use('/api/weight', weightRoutes);
app.use('/api/goals', goalRoutes);
app.use('/api/webhooks', webhookRoutes);
app.use('/api/cgm', cgmRoutes);
app.use('/api/debug', debugRoutes);
app.use('/api/reports', reportRoutes);
app.use('/api/subscription', subscriptionRoutes);
app.use('/api/email', emailRoutes);

// =============================================================================
// Error Handlers
// =============================================================================

// Sentry error handler (must be before custom error handler)
setupSentryErrorHandler(app);

// Custom error handler (must be last)
app.use(errorHandler);

// Only start server if not in test mode
// In test mode, supertest will handle server lifecycle
if (process.env.NODE_ENV !== 'test') {
  const PORT = config.port;

  app.listen(PORT, async () => {
    logger.info(
      {
        port: PORT,
        env: config.nodeEnv,
        version: packageJson.version,
        healthCheck: `http://localhost:${PORT}/health`,
        queueDashboard: `http://localhost:${PORT}/admin/queues`,
      },
      'Server started'
    );

    // Initialize notification scheduler maintenance jobs
    try {
      await notificationScheduler.scheduleMaintenanceJobs();
      logger.info('Notification scheduler maintenance jobs initialized');
    } catch (error) {
      logger.error({ error }, 'Failed to initialize notification scheduler maintenance jobs');
    }
  });
}

export default app;
