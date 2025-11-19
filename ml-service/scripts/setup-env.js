#!/usr/bin/env node

/**
 * Setup Environment - Create .env file from .env.example if it doesn't exist
 */

const fs = require('fs');
const path = require('path');

const ENV_EXAMPLE = path.join(__dirname, '..', '.env.example');
const ENV_FILE = path.join(__dirname, '..', '.env');
const ENV_TEST = path.join(__dirname, '..', '.env.test');

console.log('🔧 Setting up environment files...\n');

// Create .env from .env.example
if (!fs.existsSync(ENV_FILE)) {
  console.log('📝 Creating .env from .env.example...');
  fs.copyFileSync(ENV_EXAMPLE, ENV_FILE);
  console.log('✅ .env created successfully\n');
} else {
  console.log('✅ .env already exists\n');
}

// Create .env.test for testing
if (!fs.existsSync(ENV_TEST)) {
  console.log('📝 Creating .env.test for testing...');

  const testEnv = `# Test Environment Configuration
# This file is used during automated testing

# Application
APP_NAME=Nutri ML Service (Test)
APP_VERSION=1.0.0
ENVIRONMENT=test
DEBUG=false

# Server
HOST=127.0.0.1
PORT=8000
WORKERS=1

# Database (SQLite in-memory for tests - configured in tests/conftest.py)
DATABASE_URL=sqlite+aiosqlite:///:memory:
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis (Optional - tests work without Redis)
REDIS_URL=redis://localhost:6379/1
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=5

# Cache TTL (shorter for tests)
CACHE_TTL_FEATURES=60        # 1 minute
CACHE_TTL_PREDICTIONS=300    # 5 minutes
CACHE_TTL_MODELS=600         # 10 minutes

# ML Models
MODEL_STORAGE_PATH=./models
MODEL_VERSION=v1.0.0-test

# Feature Engineering
FEATURE_VERSION=v1.2.3
MIN_DATA_POINTS_FOR_ML=30

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Logging
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Security
SECRET_KEY=test-secret-key-not-for-production
ALGORITHM=HS256
`;

  fs.writeFileSync(ENV_TEST, testEnv);
  console.log('✅ .env.test created successfully\n');
} else {
  console.log('✅ .env.test already exists\n');
}

console.log('🎉 Environment setup complete!\n');
console.log('Next steps:');
console.log('  1. Review and update .env with your settings');
console.log('  2. Install dependencies: npm run setup:deps');
console.log('  3. Start development: npm run dev\n');
