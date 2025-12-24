# Beta Environment Setup Guide

This guide explains how to set up and manage the Nutri beta environment for testing before production deployment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                            │
├─────────────────────────────────────────────────────────────────┤
│  master branch  ──► Build (latest tag) ──► Deploy Production    │
│  develop branch ──► Build (beta tag)   ──► Deploy Beta          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Coolify                                  │
├────────────────────────┬────────────────────────────────────────┤
│     Production         │              Beta                       │
├────────────────────────┼────────────────────────────────────────┤
│ nutri-backend:latest   │  nutri-backend-beta:beta               │
│ nutri-ml:latest        │  nutri-ml-beta:beta                    │
└────────────────────────┴────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                            │
├────────────────────────┬────────────────────────────────────────┤
│     Production         │              Beta                       │
├────────────────────────┼────────────────────────────────────────┤
│ Supabase (prod)        │  Supabase (beta) or separate DB        │
│ Upstash Redis (prod)   │  Upstash Redis (beta) or DB index 1    │
└────────────────────────┴────────────────────────────────────────┘
```

## Prerequisites

- Coolify v4 running on Hetzner (or similar VPS)
- GitHub repository with Actions enabled
- Supabase account (free tier works)
- Upstash account (free tier works)

## Step 1: Set Up Beta Database (Supabase)

### Option A: Separate Supabase Project (Recommended)

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Create a new project named `nutri-beta`
3. Wait for the project to be provisioned
4. Get the connection strings from Settings > Database:
   - **Connection pooler URL** (Transaction mode): For `DATABASE_URL`
   - **Direct connection URL**: For `DIRECT_URL`

### Option B: Same Project, Different Database

```sql
-- Run in Supabase SQL Editor
CREATE DATABASE nutri_beta;
```

Then use the same connection strings but with `/nutri_beta` instead of `/postgres`.

## Step 2: Set Up Beta Redis (Upstash)

### Option A: Separate Redis Database (Recommended)

1. Go to [Upstash Console](https://console.upstash.com)
2. Create a new Redis database named `nutri-beta`
3. Copy the Redis URL (with TLS): `rediss://default:xxx@xxx.upstash.io:6379`

### Option B: Same Redis, Different Database Index

Use the same Redis URL but append `/1` for a different database:
```
rediss://default:xxx@xxx.upstash.io:6379/1
```

## Step 3: Set Up Coolify Applications

### 3.1 Create Beta Backend Application

1. In Coolify, go to your project
2. Click "New" > "Application"
3. Configure:
   - **Name**: `nutri-backend-beta`
   - **Source**: GitHub > your repo
   - **Branch**: `develop`
   - **Build Pack**: Dockerfile
   - **Dockerfile Location**: `./server/Dockerfile`
   - **Docker Image Tag**: `ghcr.io/YOUR_USER/nutri/backend:beta`

4. Set Environment Variables:
   ```
   DATABASE_URL=<beta-supabase-pooler-url>
   DIRECT_URL=<beta-supabase-direct-url>
   JWT_SECRET=<generate-new-secret>
   JWT_EXPIRES_IN=7d
   NODE_ENV=production
   ML_SERVICE_URL=http://nutri-ml-beta:8000
   ```

5. Configure Health Check:
   - **Path**: `/health`
   - **Port**: `3000`

6. Set up the webhook:
   - Go to "Webhooks" tab
   - Copy the webhook URL

### 3.2 Create Beta ML Service Application

1. Create another application:
   - **Name**: `nutri-ml-beta`
   - **Source**: GitHub > your repo
   - **Branch**: `develop`
   - **Build Pack**: Dockerfile
   - **Dockerfile Location**: `./ml-service/Dockerfile`
   - **Docker Image Tag**: `ghcr.io/YOUR_USER/nutri/ml-service:beta`

2. Set Environment Variables (see `ml-service/.env.beta.example`)

3. Configure Health Check:
   - **Path**: `/health`
   - **Port**: `8000`

4. Copy the webhook URL

## Step 4: Configure GitHub Secrets

Add these secrets to your GitHub repository:

### Beta-Specific Secrets

| Secret | Description |
|--------|-------------|
| `BETA_DATABASE_URL` | Supabase pooler URL for beta DB |
| `BETA_DIRECT_URL` | Supabase direct URL for beta DB |
| `BETA_API_URL` | Public URL to beta backend (for health checks) |
| `COOLIFY_BETA_BACKEND_WEBHOOK_URL` | Webhook URL for beta backend |
| `COOLIFY_BETA_ML_WEBHOOK_URL` | Webhook URL for beta ML service |

### Shared Secrets (if not already set)

| Secret | Description |
|--------|-------------|
| `COOLIFY_WEBHOOK_SECRET` | Shared webhook auth token |
| `DISCORD_WEBHOOK_URL` | Discord notifications (optional) |

## Step 5: Create GitHub Environment

1. Go to Settings > Environments
2. Create environment named `beta`
3. (Optional) Add protection rules:
   - Require reviewers
   - Wait timer
   - Restrict to `develop` branch

## Step 6: Configure Mobile App for Beta

### EAS Build Configuration

Create or update `eas.json`:

```json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal"
    },
    "beta": {
      "distribution": "internal",
      "ios": {
        "simulator": false
      },
      "env": {
        "APP_ENV": "beta",
        "BETA_API_URL": "https://beta-api.yourapp.com/api"
      }
    },
    "production": {
      "env": {
        "APP_ENV": "production"
      }
    }
  }
}
```

### Build and Deploy Beta App

```bash
# Build beta iOS app
eas build --platform ios --profile beta

# Submit to TestFlight (internal testing)
eas submit --platform ios --profile beta
```

## Deployment Workflow

### Automatic Deployment

```
develop branch push
       │
       ▼
┌──────────────────┐
│  Build and Push  │  ──► Images tagged with :beta
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Deploy Beta    │  ──► Runs migrations, triggers Coolify
└──────────────────┘
```

### Manual Deployment

```bash
# Trigger beta build manually
gh workflow run "Build and Push" --ref develop -f environment=beta

# Trigger beta deploy manually
gh workflow run "Deploy Beta"
```

## Environment URLs

| Environment | Backend URL | ML Service |
|-------------|-------------|------------|
| Production  | `https://z8cg...sslip.io/api` | Internal |
| Beta        | `https://beta-z8cg...sslip.io/api` | Internal |
| Development | `http://localhost:3000/api` | `http://localhost:8000` |

## Monitoring Beta

### Health Checks

```bash
# Check beta backend health
curl https://beta-api.yourapp.com/health

# Check beta ML service (via backend proxy)
curl https://beta-api.yourapp.com/api/food/health
```

### Logs

```bash
# View logs in Coolify dashboard
# Or via Docker on server:
ssh your-server
docker logs nutri-backend-beta -f
docker logs nutri-ml-beta -f
```

## Promoting Beta to Production

After testing in beta:

1. Create a PR from `develop` to `master`
2. Review and merge
3. Production deployment triggers automatically
4. Verify production health

```bash
# Quick verification
curl https://api.yourapp.com/health
```

## Rollback

### Beta Rollback

```bash
# In Coolify, roll back to previous deployment
# Or manually deploy specific tag:
docker pull ghcr.io/YOUR_USER/nutri/backend:SHA_HASH
```

### Database Rollback

```bash
# If migration needs rollback
cd server
DATABASE_URL=$BETA_DATABASE_URL npx prisma migrate resolve --rolled-back MIGRATION_NAME
```

## Cost Estimate (Free Tier)

| Service | Free Tier | Beta Usage |
|---------|-----------|------------|
| Supabase | 500MB DB, 2GB bandwidth | ✅ Sufficient |
| Upstash | 10k commands/day | ✅ Sufficient |
| Coolify | Self-hosted | Same server as prod |
| GitHub Actions | 2000 min/month | ~50 min/month for beta |

## Troubleshooting

### Common Issues

1. **Webhook timeout**: Coolify may timeout but still deploy. Check dashboard.

2. **Database connection errors**: Verify `DATABASE_URL` uses pooler URL for backend.

3. **Redis connection errors**: Ensure TLS (`rediss://`) for Upstash.

4. **Health check failures**: May be network issues from GitHub runners. Check Coolify dashboard.

### Debug Commands

```bash
# Test database connection
DATABASE_URL=$BETA_DATABASE_URL npx prisma db pull

# Test Redis connection
redis-cli -u $BETA_REDIS_URL ping

# Test webhook manually
curl -X POST -H "Authorization: Bearer $COOLIFY_WEBHOOK_SECRET" \
  "$COOLIFY_BETA_BACKEND_WEBHOOK_URL"
```

## Security Notes

1. **Different JWT Secrets**: Use unique secrets for beta and production
2. **Separate Databases**: Beta data should never mix with production
3. **Access Control**: Limit who can merge to `develop` branch
4. **Secrets Management**: Never commit secrets; use GitHub Secrets or Coolify env vars

---

*Last Updated: December 2024*
