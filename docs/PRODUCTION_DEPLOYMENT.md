# Nutri Production Deployment Guide

## Infrastructure Overview

| Component | Provider | Details |
|-----------|----------|---------|
| **VPS** | Hetzner | AMD EPYC, 4 vCPU, 8GB RAM |
| **Database** | Supabase | PostgreSQL 16 with pgbouncer |
| **Cache** | Upstash | Redis with SSL (rediss://) |
| **PaaS** | Coolify v4 | Self-hosted on Hetzner |
| **Proxy** | Traefik | Managed by Coolify |

## Service URLs

### Production Endpoints

| Service | URL | Health Check |
|---------|-----|--------------|
| **Backend API** | `http://zwogw8ccsw84w8ocws4sowok.195.201.228.58.sslip.io` | `/health` |
| **ML Service** | `http://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io` | `/health/live` |
| **Coolify Dashboard** | `http://195.201.228.58:8000` | - |

### Health Check Examples

```bash
# Backend
curl http://zwogw8ccsw84w8ocws4sowok.195.201.228.58.sslip.io/health

# ML Service
curl http://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io/health
```

## Deployment via Coolify UI

### Prerequisites
- Access to Coolify dashboard at `http://195.201.228.58:8000`
- Admin credentials

### Deploy Steps

1. **Navigate to Project**
   - Go to Projects → nutri → production
   - Select the service to deploy (backend or ml-service)

2. **Trigger Deployment**
   - Click "Redeploy" button
   - Wait for build to complete (Backend: ~2min, ML Service: ~10min)
   - Monitor deployment log for errors

3. **Verify Deployment**
   - Check container status shows "Running (healthy)"
   - Test health endpoint from external URL

## Manual Deployment via SSH

### SSH Access
```bash
ssh root@195.201.228.58
```

### Check Running Containers
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

### View Container Logs
```bash
# Backend
docker logs zwogw8ccsw84w8ocws4sowok-093433135795 --tail 100

# ML Service
docker logs z8cg8kkg4o0wg8044c8g0s0o-120429255658 --tail 100
```

### Restart a Service
```bash
# Find the Coolify application directory
ls /data/coolify/applications/

# Navigate and restart
cd /data/coolify/applications/<app-id>
docker compose down
docker compose up -d
```

### Force Pull Latest Image
```bash
cd /data/coolify/applications/<app-id>
docker compose pull
docker compose up -d
```

## Environment Variables

### Backend (Node.js)
| Variable | Description |
|----------|-------------|
| `NODE_ENV` | `production` |
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `REDIS_URL` | Upstash Redis URL (with SSL) |
| `JWT_SECRET` | JWT signing secret |
| `PORT` | `3000` |

### ML Service (Python)
| Variable | Description |
|----------|-------------|
| `ENVIRONMENT` | `production` |
| `DATABASE_URL` | PostgreSQL with asyncpg driver |
| `REDIS_URL` | Upstash Redis URL (with SSL) |
| `PORT` | `8000` |
| `FAST_MODE` | `true` (skip OWL-ViT for faster startup) |
| `HOST` | `0.0.0.0` |

## Troubleshooting

### ML Service Takes Long to Start
The ML service loads CLIP models on startup, which can take 5-10 minutes on CPU. This is normal.

**Solution**: Increase Coolify healthcheck start period to 300 seconds.

### "ModuleNotFoundError: No module named 'shap'"
The `shap` package was missing from production requirements.

**Solution**: Ensure `shap==0.44.1` is in `ml-service/requirements-prod.txt`

### "ModuleNotFoundError: No module named 'app.models.sensitivity'"
The sensitivity.py model file was missing.

**Solution**: Ensure `ml-service/app/models/sensitivity.py` exists with all required enums and SQLAlchemy models.

### Health Check Fails During Deployment
The rolling update healthcheck may fail if the service takes too long to start.

**Solution**: SSH into server and manually start the container:
```bash
cd /data/coolify/applications/<app-id>
docker compose up -d
```

### GitHub Actions Webhook 404
Coolify webhook URLs may become invalid if resources are recreated.

**Solution**:
1. Get new webhook URLs from Coolify dashboard
2. Update GitHub Secrets: `COOLIFY_BACKEND_WEBHOOK_URL`, `COOLIFY_ML_WEBHOOK_URL`

## Build Configuration

### Backend Dockerfile
- Multi-stage build
- Node.js 20 Alpine
- Production dependencies only
- Build-time Prisma generation

### ML Service Dockerfile
- Multi-stage build
- Python 3.11 slim-bookworm
- CPU-only PyTorch (`torch==2.4.1+cpu`)
- ONNX Runtime for optimized inference
- CLIP model for food classification

### Important Build Notes
- ML service Docker image is ~2.3GB due to ML dependencies
- Backend Docker image is ~300MB
- Both use Coolify's built-in Docker builder

## Monitoring

### Check Container Health
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

### View Resource Usage
```bash
docker stats --no-stream
```

### Application Logs
```bash
# Stream logs
docker logs -f <container-name>

# Last 100 lines
docker logs --tail 100 <container-name>
```

## Commits That Fixed Deployment Issues

| Commit | Fix |
|--------|-----|
| `14004e5` | Created missing `sensitivity.py` model file |
| `2f58a8a` | Fixed SULFITES enum spelling (SULPHITES → SULFITES) |
| `cbd5b81` | Added `shap` dependency to production requirements |

---

*Last Updated: 2025-12-24*
*Deployment Status: Backend ✅ ML Service ✅*
