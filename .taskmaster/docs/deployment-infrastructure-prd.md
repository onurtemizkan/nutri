# PRD: Hybrid Cloud Deployment Infrastructure

## Overview

Implement a production-ready, one-click deployment infrastructure for the Nutri application using the cost-optimized hybrid cloud approach:

- **Compute**: Hetzner CX32 VPS with Coolify (self-hosted PaaS)
- **Database**: Supabase (managed PostgreSQL)
- **Cache**: Upstash (serverless Redis)
- **Backups**: Automated via Hetzner snapshots + Supabase built-in

### Goals

1. **One-Click Deploy**: Push to `main` branch triggers automatic deployment
2. **Zero-Downtime**: Rolling deployments with health checks
3. **Frictionless**: Minimal manual intervention required
4. **Cost-Effective**: ~$8-15/month total infrastructure cost
5. **Observable**: Monitoring, logging, and alerting built-in
6. **Recoverable**: Automated backups with tested restore procedures

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│  (push to main) ──────────────────────────────────────────────► │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GitHub Actions CI/CD                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Lint & Test  │──│ Build Images │──│ Push to GHCR │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (webhook)
┌─────────────────────────────────────────────────────────────────┐
│              Hetzner CX32 (4 vCPU, 8GB RAM, €5.49/mo)           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Coolify (PaaS Layer)                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │   Traefik   │  │  Backend    │  │ ML Service  │      │   │
│  │  │ (SSL/Proxy) │  │  (Express)  │  │  (FastAPI)  │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                      │                    │
         │                      ▼                    │
         │    ┌────────────────────────────────┐    │
         │    │   Supabase (PostgreSQL)        │    │
         │    │   - Managed backups            │    │
         │    │   - Connection pooling         │    │
         │    └────────────────────────────────┘    │
         │                                          │
         │    ┌────────────────────────────────┐    │
         └───►│   Upstash (Redis)              │◄───┘
              │   - Serverless, auto-scaling   │
              │   - Global edge caching        │
              └────────────────────────────────┘
```

---

## Phase 1: Infrastructure Foundation

### 1.1 Hetzner VPS Provisioning

**Objective**: Set up the base compute infrastructure on Hetzner Cloud.

**Requirements**:
- Create Hetzner Cloud account and project
- Provision CX32 instance (4 vCPU, 8GB RAM, 80GB SSD)
- Configure firewall rules (SSH, HTTP, HTTPS only)
- Set up SSH key authentication (disable password auth)
- Configure automatic security updates
- Set up swap space (4GB) for memory overflow protection
- Document the server IP and access credentials securely

**Acceptance Criteria**:
- [ ] CX32 instance running Ubuntu 22.04 LTS
- [ ] SSH access working with key-based auth only
- [ ] Firewall allowing only ports 22, 80, 443
- [ ] Automatic security updates enabled
- [ ] Server accessible and responsive

### 1.2 Coolify Installation and Configuration

**Objective**: Install Coolify as the self-hosted PaaS layer for managing deployments.

**Requirements**:
- Install Coolify using official installation script
- Configure Coolify admin account with strong credentials
- Set up custom domain for Coolify dashboard (e.g., deploy.nutri.app)
- Configure SSL certificates via Let's Encrypt
- Set up Coolify backup to S3-compatible storage
- Configure resource limits for containers
- Set up Docker cleanup policies to prevent disk exhaustion

**Acceptance Criteria**:
- [ ] Coolify dashboard accessible via HTTPS
- [ ] Admin account configured with 2FA if available
- [ ] SSL certificates auto-renewing
- [ ] Docker system configured with log rotation
- [ ] Backup schedule configured

### 1.3 Supabase Project Setup

**Objective**: Configure Supabase as the managed PostgreSQL database.

**Requirements**:
- Create Supabase account and project
- Configure database connection pooling (PgBouncer)
- Set up database schema using Prisma migrations
- Configure Row Level Security (RLS) policies
- Set up database backups (Supabase handles this automatically on paid tiers)
- Document connection strings and credentials
- Configure connection limits appropriate for the application

**Acceptance Criteria**:
- [ ] Supabase project created in appropriate region
- [ ] Connection pooler URL documented
- [ ] Prisma schema deployed successfully
- [ ] Connection tested from local development
- [ ] Backup retention policy understood and documented

### 1.4 Upstash Redis Setup

**Objective**: Configure Upstash as the serverless Redis cache.

**Requirements**:
- Create Upstash account
- Create Redis database in region closest to Hetzner server
- Configure connection credentials
- Set up eviction policy (allkeys-lru recommended)
- Configure max memory limits
- Test connection from local development

**Acceptance Criteria**:
- [ ] Upstash Redis database created
- [ ] REST and Redis protocol URLs documented
- [ ] Connection tested successfully
- [ ] Eviction policy configured

---

## Phase 2: Application Containerization

### 2.1 Backend API Dockerfile

**Objective**: Create optimized Docker image for the Express.js backend.

**Requirements**:
- Create multi-stage Dockerfile for minimal image size
- Use Node.js 20 Alpine as base image
- Implement proper layer caching for node_modules
- Configure non-root user for security
- Add health check endpoint and HEALTHCHECK instruction
- Optimize for production (NODE_ENV=production)
- Include Prisma client generation in build stage
- Target image size under 200MB

**File**: `server/Dockerfile`

**Acceptance Criteria**:
- [ ] Dockerfile builds successfully
- [ ] Image size under 200MB
- [ ] Container runs as non-root user
- [ ] Health check endpoint responds correctly
- [ ] All environment variables documented

### 2.2 ML Service Dockerfile

**Objective**: Create optimized Docker image for the FastAPI ML service.

**Requirements**:
- Create multi-stage Dockerfile
- Use Python 3.11 slim as base image
- Implement proper layer caching for pip dependencies
- Configure non-root user for security
- Add health check endpoint and HEALTHCHECK instruction
- Optimize for production (disable debug mode)
- Pre-download/cache ML model weights if applicable
- Target image size under 1GB (ML dependencies are large)

**File**: `ml-service/Dockerfile`

**Acceptance Criteria**:
- [ ] Dockerfile builds successfully
- [ ] Container runs as non-root user
- [ ] Health check endpoint responds correctly
- [ ] ML model loads correctly on startup
- [ ] All environment variables documented

### 2.3 Docker Compose for Local Development

**Objective**: Create Docker Compose configuration that mirrors production environment.

**Requirements**:
- Create docker-compose.yml for local development
- Include all services (backend, ml-service, postgres, redis)
- Configure volume mounts for hot-reloading in development
- Set up networking between services
- Include health checks for all services
- Add docker-compose.prod.yml for production-like local testing
- Document all environment variables in .env.example

**Files**:
- `docker-compose.yml` (development)
- `docker-compose.prod.yml` (production testing)
- `.env.example` (documented environment variables)

**Acceptance Criteria**:
- [ ] `docker-compose up` starts all services
- [ ] Hot-reloading works for development
- [ ] Services can communicate with each other
- [ ] Health checks pass for all services
- [ ] Production compose file works correctly

---

## Phase 3: CI/CD Pipeline

### 3.1 GitHub Actions - Test Workflow

**Objective**: Create CI workflow that runs on all pull requests.

**Requirements**:
- Run on all pull requests to main branch
- Execute linting for all services (ESLint, Black, mypy)
- Run unit tests for backend (Jest)
- Run unit tests for ML service (pytest)
- Run TypeScript type checking
- Cache dependencies for faster builds
- Fail fast on any error
- Report test coverage

**File**: `.github/workflows/test.yml`

**Acceptance Criteria**:
- [ ] Workflow triggers on PR to main
- [ ] All linters run successfully
- [ ] All tests run successfully
- [ ] Coverage reports generated
- [ ] Build time under 5 minutes

### 3.2 GitHub Actions - Build and Push Workflow

**Objective**: Create workflow that builds and pushes Docker images on merge to main.

**Requirements**:
- Trigger on push to main branch (after PR merge)
- Build Docker images for backend and ML service
- Tag images with git SHA and 'latest'
- Push to GitHub Container Registry (GHCR)
- Use Docker layer caching for faster builds
- Run security scan on images (Trivy or similar)
- Output image digests for verification

**File**: `.github/workflows/build.yml`

**Acceptance Criteria**:
- [ ] Workflow triggers on push to main
- [ ] Images built successfully
- [ ] Images pushed to GHCR
- [ ] Security scan runs without critical vulnerabilities
- [ ] Build time under 10 minutes

### 3.3 GitHub Actions - Deploy Workflow

**Objective**: Create workflow that triggers Coolify deployment after successful build.

**Requirements**:
- Trigger after successful build workflow
- Call Coolify webhook to trigger deployment
- Wait for deployment to complete
- Verify deployment health checks pass
- Send notification on success/failure (Discord/Slack webhook)
- Support manual trigger for rollback scenarios
- Include deployment URL in notification

**File**: `.github/workflows/deploy.yml`

**Acceptance Criteria**:
- [ ] Workflow triggers after successful build
- [ ] Coolify deployment triggered successfully
- [ ] Health checks verified post-deployment
- [ ] Notifications sent correctly
- [ ] Manual rollback trigger works

### 3.4 Database Migration Workflow

**Objective**: Create safe database migration process integrated with deployment.

**Requirements**:
- Run Prisma migrations as part of deployment
- Create backup before migration (for paid Supabase tiers)
- Implement migration dry-run capability
- Handle migration failures gracefully
- Support rollback of failed migrations
- Log all migration activities

**Implementation Notes**:
- Migrations should run before new application version starts
- Use Prisma's migration system
- Consider using a separate migration job in Coolify

**Acceptance Criteria**:
- [ ] Migrations run automatically on deploy
- [ ] Failed migrations don't break the application
- [ ] Migration history tracked
- [ ] Rollback procedure documented and tested

---

## Phase 4: Coolify Application Configuration

### 4.1 Backend Service Configuration

**Objective**: Configure the backend Express.js service in Coolify.

**Requirements**:
- Create new application in Coolify from GHCR image
- Configure environment variables from secrets
- Set up health check endpoint (/api/health)
- Configure resource limits (CPU: 1 core, Memory: 1.5GB)
- Set up auto-restart on failure
- Configure logging to stdout/stderr
- Set up custom domain (api.nutri.app)
- Configure SSL certificate

**Acceptance Criteria**:
- [ ] Service running in Coolify
- [ ] Environment variables configured securely
- [ ] Health checks passing
- [ ] Custom domain working with HTTPS
- [ ] Logs accessible in Coolify dashboard

### 4.2 ML Service Configuration

**Objective**: Configure the ML FastAPI service in Coolify.

**Requirements**:
- Create new application in Coolify from GHCR image
- Configure environment variables from secrets
- Set up health check endpoint (/health)
- Configure resource limits (CPU: 2 cores, Memory: 4GB)
- Set up auto-restart on failure
- Configure internal networking (not exposed externally)
- Backend should access ML service via internal Docker network

**Acceptance Criteria**:
- [ ] Service running in Coolify
- [ ] Environment variables configured securely
- [ ] Health checks passing
- [ ] Service accessible from backend via internal network
- [ ] Not accessible from public internet

### 4.3 Coolify Webhook Configuration

**Objective**: Set up webhooks for automated deployments from GitHub.

**Requirements**:
- Generate Coolify deploy webhook URLs for each service
- Configure webhook secrets for security
- Store webhook URLs as GitHub secrets
- Test webhook triggers deployment
- Configure webhook to only deploy on successful builds

**Acceptance Criteria**:
- [ ] Webhook URLs generated and secured
- [ ] GitHub Actions can trigger deployments
- [ ] Unauthorized webhook calls rejected
- [ ] Deployment logs show webhook trigger source

### 4.4 Environment Secrets Management

**Objective**: Implement secure environment variable management.

**Requirements**:
- Document all required environment variables
- Store production secrets in Coolify's secret management
- Store CI/CD secrets in GitHub Secrets
- Implement secret rotation procedure
- Never log or expose secrets
- Use different secrets for each environment

**Environment Variables to Configure**:
```
# Backend
DATABASE_URL=postgresql://...
DIRECT_URL=postgresql://... (for migrations)
REDIS_URL=redis://...
JWT_SECRET=...
NODE_ENV=production

# ML Service
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
MODEL_PATH=/app/models
ENVIRONMENT=production
```

**Acceptance Criteria**:
- [ ] All secrets stored securely
- [ ] No secrets in source control
- [ ] Secret rotation procedure documented
- [ ] Environment variable documentation complete

---

## Phase 5: Monitoring and Observability

### 5.1 Health Check Endpoints

**Objective**: Implement comprehensive health check endpoints for all services.

**Requirements**:
- Backend: GET /api/health (checks DB and Redis connectivity)
- ML Service: GET /health (checks model loaded and DB connectivity)
- Return detailed status for each dependency
- Return appropriate HTTP status codes (200 OK, 503 Unavailable)
- Include version information in response
- Implement readiness vs liveness probes

**Response Format**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "checks": {
    "database": { "status": "healthy", "latency_ms": 5 },
    "redis": { "status": "healthy", "latency_ms": 2 },
    "ml_service": { "status": "healthy", "latency_ms": 15 }
  }
}
```

**Acceptance Criteria**:
- [ ] Health endpoints implemented for all services
- [ ] Database connectivity checked
- [ ] Redis connectivity checked
- [ ] Service dependencies checked
- [ ] Appropriate status codes returned

### 5.2 Logging Configuration

**Objective**: Implement structured logging for all services.

**Requirements**:
- Use JSON structured logging format
- Include correlation IDs for request tracing
- Log levels: error, warn, info, debug
- Configure log rotation to prevent disk exhaustion
- Sensitive data redaction (passwords, tokens)
- Include timestamp, service name, and request ID

**Acceptance Criteria**:
- [ ] Structured JSON logs configured
- [ ] Correlation IDs included in logs
- [ ] Sensitive data redacted
- [ ] Logs accessible in Coolify
- [ ] Log rotation configured

### 5.3 Uptime Monitoring Setup

**Objective**: Set up external uptime monitoring.

**Requirements**:
- Set up free uptime monitoring (UptimeRobot, Betterstack, or similar)
- Monitor health endpoints for all public services
- Configure alerting (email, Discord, or Slack)
- Set check interval (every 5 minutes)
- Configure status page (optional)

**Acceptance Criteria**:
- [ ] Uptime monitoring configured
- [ ] Alerts configured and tested
- [ ] Health endpoints monitored
- [ ] Response time tracking enabled

### 5.4 Error Tracking Integration

**Objective**: Implement error tracking for production debugging.

**Requirements**:
- Integrate Sentry (free tier) or similar error tracking
- Configure source maps for frontend errors
- Set up error grouping and deduplication
- Configure alerting for new errors
- Include environment and user context
- Set up performance monitoring (optional)

**Acceptance Criteria**:
- [ ] Error tracking integrated in backend
- [ ] Error tracking integrated in ML service
- [ ] Alerts configured for critical errors
- [ ] Source maps uploaded for debugging
- [ ] Performance monitoring enabled (optional)

---

## Phase 6: Backup and Recovery

### 6.1 Database Backup Strategy

**Objective**: Implement comprehensive database backup strategy.

**Requirements**:
- Supabase provides automatic daily backups (Pro tier)
- For free tier: implement pg_dump backup script
- Store backups in S3-compatible storage (Backblaze B2 or similar)
- Retain 7 daily, 4 weekly, and 3 monthly backups
- Test restore procedure quarterly
- Document backup and restore procedures

**Acceptance Criteria**:
- [ ] Backup strategy documented
- [ ] Backup script created (for free tier)
- [ ] Backup storage configured
- [ ] Retention policy implemented
- [ ] Restore procedure tested and documented

### 6.2 Hetzner Server Snapshots

**Objective**: Configure server-level backups via Hetzner snapshots.

**Requirements**:
- Enable Hetzner automatic backups (€1.20/mo for CX32)
- Configure weekly snapshot schedule
- Retain last 4 snapshots
- Document snapshot restore procedure
- Test restore procedure

**Acceptance Criteria**:
- [ ] Hetzner backups enabled
- [ ] Snapshot schedule configured
- [ ] Restore procedure documented
- [ ] Restore tested successfully

### 6.3 Application State Backup

**Objective**: Backup application configuration and state.

**Requirements**:
- Export Coolify configuration
- Backup environment variables (encrypted)
- Store in version control or secure storage
- Document recovery procedure for complete server loss
- Include recovery time estimates

**Acceptance Criteria**:
- [ ] Coolify configuration exportable
- [ ] Recovery procedure documented
- [ ] Recovery tested from scratch
- [ ] Recovery time estimate documented

---

## Phase 7: Deployment Scripts and Documentation

### 7.1 One-Click Deploy Script

**Objective**: Create scripts for initial infrastructure setup.

**Requirements**:
- Create setup script for new Hetzner server
- Automate Coolify installation
- Automate initial application deployment
- Include rollback script
- Support idempotent execution (safe to run multiple times)

**Files**:
- `scripts/deploy/setup-server.sh`
- `scripts/deploy/deploy-apps.sh`
- `scripts/deploy/rollback.sh`

**Acceptance Criteria**:
- [ ] Setup script provisions complete environment
- [ ] Scripts are idempotent
- [ ] Rollback script works correctly
- [ ] Scripts documented with usage examples

### 7.2 Deployment Documentation

**Objective**: Create comprehensive deployment documentation.

**Requirements**:
- Document complete deployment architecture
- Include step-by-step setup guide
- Document all environment variables
- Include troubleshooting guide
- Document rollback procedures
- Include cost breakdown
- Create runbook for common operations

**Files**:
- `docs/deployment/README.md`
- `docs/deployment/SETUP.md`
- `docs/deployment/TROUBLESHOOTING.md`
- `docs/deployment/RUNBOOK.md`

**Acceptance Criteria**:
- [ ] Architecture diagram included
- [ ] Setup guide complete and tested
- [ ] Environment variables documented
- [ ] Troubleshooting guide covers common issues
- [ ] Runbook covers daily operations

### 7.3 Local Development Parity

**Objective**: Ensure local development environment mirrors production.

**Requirements**:
- Document local setup using Docker Compose
- Create seed data scripts for development
- Ensure environment variable parity
- Document differences between local and production
- Create script to sync production schema to local

**Acceptance Criteria**:
- [ ] Local environment matches production architecture
- [ ] Seed data available for development
- [ ] Differences documented
- [ ] Schema sync script works

---

## Phase 8: Security Hardening

### 8.1 Server Security Configuration

**Objective**: Harden the Hetzner server security.

**Requirements**:
- Configure UFW firewall (allow only 22, 80, 443)
- Set up fail2ban for SSH brute force protection
- Disable root SSH login
- Configure SSH key-only authentication
- Enable automatic security updates
- Configure secure SSH settings (disable password auth)
- Set up intrusion detection (optional: OSSEC or similar)

**Acceptance Criteria**:
- [ ] Firewall configured correctly
- [ ] fail2ban active and configured
- [ ] Root login disabled
- [ ] Password authentication disabled
- [ ] Auto-updates enabled

### 8.2 Application Security

**Objective**: Implement application-level security measures.

**Requirements**:
- Enable HTTPS only (redirect HTTP to HTTPS)
- Configure secure headers (HSTS, CSP, etc.)
- Implement rate limiting
- Configure CORS properly
- Enable request validation
- Implement API authentication
- Set up secrets scanning in CI

**Acceptance Criteria**:
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Rate limiting active
- [ ] CORS configured properly
- [ ] Secrets scanning enabled

### 8.3 Dependency Security

**Objective**: Implement dependency vulnerability management.

**Requirements**:
- Enable Dependabot for automated dependency updates
- Configure security scanning in CI (npm audit, safety)
- Set up Docker image vulnerability scanning
- Create policy for addressing vulnerabilities
- Document update procedures

**Acceptance Criteria**:
- [ ] Dependabot enabled
- [ ] Security scanning in CI
- [ ] Docker image scanning enabled
- [ ] Vulnerability policy documented

---

## Success Metrics

### Deployment Metrics
- **Deploy Frequency**: Able to deploy multiple times per day
- **Lead Time**: < 10 minutes from merge to production
- **Deployment Success Rate**: > 95%
- **Rollback Time**: < 5 minutes

### Reliability Metrics
- **Uptime Target**: 99.5% (allows ~3.6 hours downtime/month)
- **Mean Time to Recovery**: < 30 minutes
- **Health Check Response Time**: < 100ms

### Cost Metrics
- **Infrastructure Cost**: < $15/month
- **Scaling Headroom**: 10x current load without architecture changes

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Infrastructure Foundation | 1-2 days | None |
| Phase 2: Application Containerization | 1-2 days | Phase 1 |
| Phase 3: CI/CD Pipeline | 1-2 days | Phase 2 |
| Phase 4: Coolify Configuration | 1 day | Phase 1, 2, 3 |
| Phase 5: Monitoring | 1 day | Phase 4 |
| Phase 6: Backup and Recovery | 1 day | Phase 4 |
| Phase 7: Documentation | 1 day | All phases |
| Phase 8: Security Hardening | 1 day | Phase 1, 4 |

**Total Estimated Time**: 8-12 days

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Hetzner outage | Document manual failover to alternative provider |
| Supabase free tier limits | Monitor usage, plan upgrade path |
| Coolify bugs | Keep backup of raw Docker deployment method |
| Data loss | Multiple backup strategies, tested restores |
| Security breach | Defense in depth, monitoring, quick response plan |

---

## Out of Scope (Future Improvements)

- Multi-region deployment
- Kubernetes migration
- Auto-scaling based on load
- Blue-green deployments
- Feature flags system
- A/B testing infrastructure
- CDN integration for static assets
