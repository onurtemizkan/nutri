# Nutri Deployment Documentation

This directory contains comprehensive deployment documentation for the Nutri nutrition tracking application.

## Architecture Overview

```
                              GitHub Repository
                                     |
                        (push to master branch)
                                     |
                                     v
    +------------------------------------------------------------------+
    |                       GitHub Actions CI/CD                        |
    |  +------------------+  +------------------+  +------------------+ |
    |  |   Lint & Test    |  |  Build Images    |  |  Push to GHCR    | |
    |  |  (ci.yml)        |->|  (build.yml)     |->|                  | |
    |  +------------------+  +------------------+  +------------------+ |
    |                                                      |           |
    |  +------------------+  +------------------+           |           |
    |  | Security Scan    |  | Run Migrations   |          |           |
    |  | (Trivy)          |  | (migrate.yml)    |          |           |
    |  +------------------+  +------------------+           |           |
    +------------------------------------------------------------------+
                                     |
                            (webhook trigger)
                                     |
                                     v
    +------------------------------------------------------------------+
    |              Hetzner CX32 VPS (4 vCPU, 8GB RAM)                   |
    |  +------------------------------------------------------------+  |
    |  |                    Coolify (Self-hosted PaaS)               |  |
    |  |  +------------------+  +-------------+  +----------------+  |  |
    |  |  |     Traefik      |  |   Backend   |  |   ML Service   |  |  |
    |  |  |   (SSL/Proxy)    |  |  (Express)  |  |   (FastAPI)    |  |  |
    |  |  |   Port 80/443    |  |  Port 3000  |  |   Port 8000    |  |  |
    |  |  +------------------+  +-------------+  +----------------+  |  |
    |  +------------------------------------------------------------+  |
    +------------------------------------------------------------------+
              |                           |
              v                           v
    +-------------------+       +-------------------+
    |     Supabase      |       |   Upstash Redis   |
    |   (PostgreSQL)    |       |     (Cache)       |
    |    Free Tier      |       |    Free Tier      |
    +-------------------+       +-------------------+
```

## Services Overview

| Service | Technology | Port | Purpose |
|---------|------------|------|---------|
| Backend API | Node.js/Express | 3000 | REST API for mobile app |
| ML Service | Python/FastAPI | 8000 | ML predictions and analysis |
| Traefik | Reverse Proxy | 80, 443 | SSL termination, routing |
| PostgreSQL | Supabase | 5432 | Primary database |
| Redis | Upstash | 6379 | Caching, rate limiting |

## Cost Breakdown

| Service | Provider | Plan | Cost/Month |
|---------|----------|------|------------|
| VPS | Hetzner CX32 | 4 vCPU, 8GB RAM | ~$6 |
| Backups | Hetzner | 20% of VPS | ~$1.20 |
| Database | Supabase | Free Tier | $0 |
| Redis | Upstash | Free Tier | $0 |
| Domain | Various | Optional | ~$1/mo |
| **Total** | | | **~$8/month** |

### Free Tier Limits

**Supabase Free Tier:**
- 500 MB database storage
- 2 GB bandwidth
- 50,000 monthly active users
- 500 MB file storage

**Upstash Free Tier:**
- 10,000 commands/day
- 256 MB storage
- 1 database

> These limits are more than sufficient for ~200 daily active users.

## Quick Links

| Document | Description |
|----------|-------------|
| [SETUP.md](./SETUP.md) | Initial server and service setup |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common issues and solutions |
| [RUNBOOK.md](./RUNBOOK.md) | Daily operations and procedures |
| [ENVIRONMENT.md](./ENVIRONMENT.md) | Environment variables reference |

## Deployment Flow

### Automatic Deployment (CI/CD)

1. **Push to `master` branch**
2. **CI Pipeline** (`ci.yml`)
   - Runs linting, type checking, tests
   - Security scanning (npm audit, Trivy)
3. **Build Pipeline** (`build.yml`)
   - Builds Docker images
   - Pushes to GitHub Container Registry (GHCR)
   - Trivy vulnerability scanning on images
4. **Deploy Pipeline** (`deploy.yml`)
   - Runs database migrations
   - Triggers Coolify webhook
   - Health check verification
   - Discord notification

### Manual Deployment

```bash
# SSH to server
ssh root@<server-ip>

# Pull latest images
docker pull ghcr.io/onurtemizkan/nutri/backend:latest
docker pull ghcr.io/onurtemizkan/nutri/ml-service:latest

# Restart via Coolify dashboard
# Or use docker compose
cd /data/coolify/applications/<app-id>
docker compose pull && docker compose up -d
```

## Repository Structure

```
nutri/
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml              # Lint, test, security scan
│   ├── build.yml           # Build and push images
│   ├── deploy.yml          # Deploy to production
│   └── migrate.yml         # Database migrations
├── server/                  # Backend API
│   ├── Dockerfile
│   └── prisma/              # Database schema
├── ml-service/              # ML Service
│   └── Dockerfile
├── scripts/deploy/          # Deployment scripts
│   ├── setup-server.sh     # Server provisioning
│   └── migrate.sh          # Migration helper
├── docker-compose.yml       # Local development
└── docker-compose.prod.yml  # Production reference
```

## Key Technologies

- **Container Platform:** Docker + Docker Compose
- **PaaS:** Coolify (self-hosted)
- **CI/CD:** GitHub Actions
- **Container Registry:** GitHub Container Registry (GHCR)
- **Reverse Proxy:** Traefik (built into Coolify)
- **SSL:** Let's Encrypt (automatic via Traefik)

## Security Features

- **Firewall:** UFW with minimal open ports (22, 80, 443)
- **Intrusion Prevention:** fail2ban with SSH protection
- **SSH:** Key-only authentication, root via key only
- **Auto-updates:** Unattended security upgrades
- **Vulnerability Scanning:** Trivy on CI/CD
- **Secrets Management:** GitHub Secrets + Coolify env vars

## Monitoring

### Health Endpoints

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Backend | `/health` | Full health with dependencies |
| Backend | `/health/live` | Liveness probe |
| Backend | `/ready` | Readiness probe |
| ML Service | `/health` | Full health check |
| ML Service | `/health/live` | Liveness probe |
| ML Service | `/ready` | Readiness probe |

### Recommended Monitoring

- **UptimeRobot** (free tier): External uptime monitoring
- **Coolify Dashboard**: Container status, logs
- **Hetzner Console**: Server metrics, alerts

## Support

For deployment issues:
1. Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
2. Review GitHub Actions logs
3. Check Coolify dashboard logs
4. Review server logs: `docker logs <container>`

---

*Last updated: December 2025*
