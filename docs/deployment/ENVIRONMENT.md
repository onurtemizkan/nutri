# Environment Variables Reference

Complete reference for all environment variables used in the Nutri application.

## Quick Reference

### Required Variables by Service

| Variable | Backend | ML Service | Where to Set |
|----------|---------|------------|--------------|
| `DATABASE_URL` | Required | Required | Coolify + GitHub Secrets |
| `REDIS_URL` | Required | Required | Coolify + GitHub Secrets |
| `JWT_SECRET` | Required | - | Coolify + GitHub Secrets |
| `NODE_ENV` | Required | - | Coolify |
| `ENVIRONMENT` | - | Required | Coolify |
| `PORT` | Optional | Optional | Coolify |

---

## Backend API (Express/Node.js)

### Core Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `NODE_ENV` | Application environment | Yes | `development` | `production` |
| `PORT` | HTTP server port | No | `3000` | `3000` |

**NODE_ENV Values:**
- `development` - Verbose logging, debug features enabled
- `production` - Optimized for production, minimal logging
- `test` - Test mode with test database

### Database Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Yes | - | See below |

**DATABASE_URL Format (Supabase):**
```
postgresql://postgres:[PASSWORD]@[PROJECT].supabase.co:6543/postgres
```

| Component | Description |
|-----------|-------------|
| `postgres` | Username (default for Supabase) |
| `[PASSWORD]` | Database password (URL-encoded) |
| `[PROJECT]` | Supabase project reference |
| `6543` | Connection pooler port (recommended) |
| `5432` | Direct connection port (higher latency) |

**Important:**
- Use port `6543` for production (connection pooling)
- URL-encode special characters in password
- Never commit to version control

### Redis Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `REDIS_URL` | Redis connection string | Yes | - | See below |

**REDIS_URL Format (Upstash):**
```
rediss://default:[PASSWORD]@[HOST].upstash.io:6379
```

**Note:** `rediss://` (double 's') indicates TLS connection.

### Authentication

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `JWT_SECRET` | Secret for signing JWTs | Yes | - | 64-char random string |
| `JWT_EXPIRES_IN` | Token expiration time | No | `7d` | `1d`, `12h`, `7d` |

**Generate JWT_SECRET:**
```bash
openssl rand -hex 32
# Output: 64-character hex string
```

### Service Integration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `ML_SERVICE_URL` | ML Service base URL | No | `http://localhost:8080` | `http://nutri-ml-service:8080` |

**In Coolify:** Use Docker service name for inter-service communication.

### CORS Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `CORS_ORIGIN` | Allowed origins | No | `*` (dev) | `https://app.nutri.com` |

**Production:** Always set to specific origins:
```
CORS_ORIGIN=https://app.nutri.com,https://www.nutri.com
```

### Logging

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `LOG_LEVEL` | Minimum log level | No | `info` (prod), `debug` (dev) | `info`, `warn`, `error`, `debug` |

**Log Levels (from least to most verbose):**
- `error` - Only errors
- `warn` - Warnings and errors
- `info` - General info, warnings, errors
- `debug` - All logs including debug

---

## ML Service (FastAPI/Python)

### Core Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `ENVIRONMENT` | Application environment | Yes | `development` | `production` |
| `PORT` | HTTP server port | No | `8080` | `8080` |
| `HOST` | Bind address | No | `0.0.0.0` | `0.0.0.0` |
| `WORKERS` | Uvicorn workers | No | `4` | `4` |
| `DEBUG` | Debug mode | No | `false` | `false` |

### Database Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Yes | - | Same as Backend |

### Redis Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `REDIS_URL` | Redis connection string | Yes | - | Same as Backend |

### ML Model Configuration

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `ML_MODEL_PATH` | Path to model files | No | `/app/models` | `/app/models` |
| `TORCH_DEVICE` | PyTorch device | No | `cpu` | `cpu`, `cuda` |
| `PREDICTION_CACHE_TTL` | Cache TTL for predictions | No | `3600` | `3600` (seconds) |

### Logging

| Variable | Description | Required | Default | Example |
|----------|-------------|----------|---------|---------|
| `LOG_LEVEL` | Minimum log level | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## GitHub Actions / CI/CD

### Required Secrets

Configure in: GitHub > Settings > Secrets and variables > Actions

| Secret | Description | Used In |
|--------|-------------|---------|
| `DATABASE_URL` | Production database URL | `migrate.yml`, `deploy.yml` |
| `REDIS_URL` | Production Redis URL | `deploy.yml` |
| `JWT_SECRET` | Production JWT secret | Reference only |
| `COOLIFY_BACKEND_WEBHOOK_URL` | Backend deploy webhook | `deploy.yml` |
| `COOLIFY_ML_SERVICE_WEBHOOK_URL` | ML Service deploy webhook | `deploy.yml` |
| `COOLIFY_WEBHOOK_SECRET` | Coolify API token | `deploy.yml` |
| `DISCORD_WEBHOOK_URL` | Discord notifications | `deploy.yml` (optional) |

### Where to Find Values

| Secret | Source |
|--------|--------|
| `DATABASE_URL` | Supabase > Settings > Database > Connection string |
| `REDIS_URL` | Upstash > Database > Details > REST URL |
| `COOLIFY_*_WEBHOOK_URL` | Coolify > Service > Webhooks |
| `COOLIFY_WEBHOOK_SECRET` | Coolify > Settings > API |
| `DISCORD_WEBHOOK_URL` | Discord > Server Settings > Integrations > Webhooks |

---

## Coolify Service Configuration

### Backend Service Environment

```env
# Core
NODE_ENV=production
PORT=3000

# Database
DATABASE_URL=postgresql://postgres:[PASSWORD]@[HOST]:6543/postgres

# Redis
REDIS_URL=rediss://default:[PASSWORD]@[HOST].upstash.io:6379

# Auth
JWT_SECRET=[64-char-secret]
JWT_EXPIRES_IN=7d

# Service Integration
ML_SERVICE_URL=http://nutri-ml-service:8080

# Logging
LOG_LEVEL=info

# CORS
CORS_ORIGIN=https://api.yourdomain.com
```

### ML Service Environment

```env
# Core
ENVIRONMENT=production
PORT=8080
HOST=0.0.0.0
WORKERS=4
DEBUG=false

# Database
DATABASE_URL=postgresql://postgres:[PASSWORD]@[HOST]:6543/postgres

# Redis
REDIS_URL=rediss://default:[PASSWORD]@[HOST].upstash.io:6379

# ML
ML_MODEL_PATH=/app/models
TORCH_DEVICE=cpu
PREDICTION_CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
```

---

## Local Development

### Backend (.env in server/)

```env
# Development settings
NODE_ENV=development
PORT=3000

# Local database (or Supabase)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nutri_dev

# Local Redis (or Upstash)
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET=dev-secret-not-for-production
JWT_EXPIRES_IN=7d

# ML Service
ML_SERVICE_URL=http://localhost:8080

# Logging
LOG_LEVEL=debug

# CORS
CORS_ORIGIN=*
```

### ML Service (.env in ml-service/)

```env
ENVIRONMENT=development
PORT=8080
HOST=0.0.0.0
DEBUG=true

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nutri_dev
REDIS_URL=redis://localhost:6379

LOG_LEVEL=DEBUG
```

---

## Security Best Practices

### Sensitive Variables

**Never commit these:**
- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `COOLIFY_WEBHOOK_SECRET`
- Any `*_API_KEY` or `*_SECRET`

### Storage Recommendations

| Variable Type | Store In |
|---------------|----------|
| Production secrets | GitHub Secrets + Coolify |
| Development secrets | Local `.env` (gitignored) |
| Non-sensitive config | Can commit in `.env.example` |

### Secret Rotation

**JWT_SECRET:**
1. Generate new secret
2. Update in Coolify and GitHub Secrets
3. Redeploy services
4. Users will need to re-authenticate

**DATABASE_URL (password change):**
1. Generate new password in Supabase
2. Update `DATABASE_URL` everywhere
3. Redeploy services

**REDIS_URL (password change):**
1. Generate new credentials in Upstash
2. Update `REDIS_URL` everywhere
3. Redeploy services

---

## Validation Checklist

Before deploying, verify:

### Backend
- [ ] `NODE_ENV` = `production`
- [ ] `DATABASE_URL` is Supabase production URL
- [ ] `DATABASE_URL` uses port `6543` (pooler)
- [ ] `REDIS_URL` is Upstash production URL
- [ ] `REDIS_URL` uses `rediss://` (TLS)
- [ ] `JWT_SECRET` is 64+ characters, unique
- [ ] `CORS_ORIGIN` is set to specific domains
- [ ] `LOG_LEVEL` is `info` or higher

### ML Service
- [ ] `ENVIRONMENT` = `production`
- [ ] `DATABASE_URL` is Supabase production URL
- [ ] `REDIS_URL` is Upstash production URL
- [ ] `DEBUG` = `false`

### GitHub Secrets
- [ ] All required secrets are set
- [ ] No spaces or extra characters in values
- [ ] Values match Coolify configuration

---

## Reference Files

- `server/.env.example` - Backend example configuration
- `ml-service/.env.example` - ML Service example configuration (if exists)
- `.env.prod.example` - Production template (root directory)

---

*Last updated: December 2025*
