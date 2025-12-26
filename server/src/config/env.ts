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
};
