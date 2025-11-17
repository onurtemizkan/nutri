import dotenv from 'dotenv';

dotenv.config();

// Validate required environment variables
const requiredEnvVars = ['DATABASE_URL', 'JWT_SECRET'];
const missingEnvVars = requiredEnvVars.filter((envVar) => !process.env[envVar]);

if (missingEnvVars.length > 0 && process.env.NODE_ENV === 'production') {
  throw new Error(
    `Missing required environment variables: ${missingEnvVars.join(', ')}`
  );
}

// Warn in development
if (missingEnvVars.length > 0 && process.env.NODE_ENV !== 'production') {
  console.warn(
    `⚠️  Warning: Missing environment variables: ${missingEnvVars.join(', ')}`
  );
  console.warn('⚠️  Using default values. DO NOT use in production!');
}

// Validate JWT secret strength in production
if (
  process.env.NODE_ENV === 'production' &&
  process.env.JWT_SECRET &&
  process.env.JWT_SECRET.length < 32
) {
  throw new Error(
    'JWT_SECRET must be at least 32 characters long in production'
  );
}

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
};
