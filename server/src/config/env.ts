import dotenv from 'dotenv';

dotenv.config();

// Validate required environment variables
const requiredEnvVars = ['DATABASE_URL', 'JWT_SECRET'];
const missingEnvVars = requiredEnvVars.filter((envVar) => !process.env[envVar]);

if (missingEnvVars.length > 0 && process.env.NODE_ENV === 'production') {
  throw new Error(`Missing required environment variables: ${missingEnvVars.join(', ')}`);
}

// Warn in development
if (missingEnvVars.length > 0 && process.env.NODE_ENV !== 'production') {
  console.warn(`⚠️  Warning: Missing environment variables: ${missingEnvVars.join(', ')}`);
  console.warn('⚠️  Using default values. DO NOT use in production!');
}

// Validate JWT secret strength in production
if (
  process.env.NODE_ENV === 'production' &&
  process.env.JWT_SECRET &&
  process.env.JWT_SECRET.length < 32
) {
  throw new Error('JWT_SECRET must be at least 32 characters long in production');
}

/**
 * Parse CORS origins from environment variable
 * Accepts comma-separated list of origins
 * Example: CORS_ORIGIN=https://app.example.com,https://admin.example.com
 */
function parseCorsOrigins(): string[] {
  const origins = process.env.CORS_ORIGIN;
  if (!origins) {
    return [];
  }
  return origins
    .split(',')
    .map((origin) => origin.trim())
    .filter((origin) => origin.length > 0);
}

/**
 * Default allowed origins for development
 * Includes common Expo and React Native development ports
 */
const DEV_ALLOWED_ORIGINS = [
  'http://localhost:3000',
  'http://localhost:8081', // Metro bundler
  'http://localhost:19000', // Expo Go
  'http://localhost:19006', // Expo web
  'http://127.0.0.1:3000',
  'http://127.0.0.1:8081',
  'http://127.0.0.1:19000',
  'http://127.0.0.1:19006',
];

export const config = {
  port: parseInt(process.env.PORT || '3000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  databaseUrl: process.env.DATABASE_URL || '',
  jwt: {
    secret:
      process.env.JWT_SECRET ||
      (process.env.NODE_ENV === 'production'
        ? (() => {
            throw new Error('JWT_SECRET is required in production');
          })()
        : 'dev-secret-key-min-32-chars-long!!!'),
    expiresIn: process.env.JWT_EXPIRES_IN || '7d',
  },
  cors: {
    /**
     * Allowed origins for CORS
     * In production: uses CORS_ORIGIN environment variable (required)
     * In development: allows localhost origins + any from CORS_ORIGIN
     */
    origins:
      process.env.NODE_ENV === 'production'
        ? parseCorsOrigins()
        : [...DEV_ALLOWED_ORIGINS, ...parseCorsOrigins()],
    /**
     * Whether to allow credentials (cookies, authorization headers)
     */
    credentials: true,
  },
  /**
   * Sentry Configuration (optional)
   * Set SENTRY_DSN to enable error tracking
   */
  sentry: {
    dsn: process.env.SENTRY_DSN || '',
    /**
     * Sample rate for performance tracing
     * 0.1 = 10% of transactions in production
     * 1.0 = 100% in development
     */
    tracesSampleRate: parseFloat(process.env.SENTRY_TRACES_SAMPLE_RATE || '0.1'),
  },
  email: {
    /** Resend API key for sending emails */
    resendApiKey: process.env.RESEND_API_KEY || '',
    /** Webhook secret for verifying Resend webhooks */
    webhookSecret: process.env.RESEND_WEBHOOK_SECRET || '',
    /** From address for transactional emails */
    fromTransactional: process.env.EMAIL_FROM_TRANSACTIONAL || 'Nutri <hello@mail.nutriapp.com>',
    /** From address for marketing emails */
    fromMarketing: process.env.EMAIL_FROM_MARKETING || 'Nutri <updates@marketing.nutriapp.com>',
    /** Reply-to address for all emails */
    replyTo: process.env.EMAIL_REPLY_TO || 'support@nutriapp.com',
    /** Domain for transactional emails */
    domainTransactional: process.env.EMAIL_DOMAIN_TRANSACTIONAL || 'mail.nutriapp.com',
    /** Domain for marketing emails */
    domainMarketing: process.env.EMAIL_DOMAIN_MARKETING || 'marketing.nutriapp.com',
    /** Rate limit for email sending (per second) */
    rateLimitPerSecond: parseInt(process.env.EMAIL_RATE_LIMIT_PER_SECOND || '100', 10),
    /** Batch size for bulk email sending */
    batchSize: parseInt(process.env.EMAIL_BATCH_SIZE || '1000', 10),
    /** Base URL for email links (unsubscribe, etc.) */
    baseUrl: process.env.EMAIL_BASE_URL || process.env.APP_URL || 'https://nutriapp.com',
    /** Whether email sending is enabled */
    enabled: process.env.EMAIL_ENABLED !== 'false',
  },
  redis: {
    /** Redis URL for queues and caching */
    url: process.env.REDIS_URL || 'redis://localhost:6379',
  },
};
