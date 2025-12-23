# Task ID: 29

**Title:** Create Deployment Documentation

**Status:** done

**Dependencies:** 28 ✓

**Priority:** medium

**Description:** Create comprehensive deployment documentation including architecture diagrams, setup guides, troubleshooting, and runbooks for daily operations.

**Details:**

**Create `docs/deployment/README.md`:**
```markdown
# Nutri Deployment Documentation

## Architecture Overview

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
                                    │ (webhook)
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              Hetzner CX32 (4 vCPU, 8GB RAM)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Coolify (PaaS Layer)                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │   Traefik   │  │  Backend    │  │ ML Service  │      │   │
│  │  │ (SSL/Proxy) │  │  (Express)  │  │  (FastAPI)  │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                      │
         ▼                      ▼
┌────────────────────┐  ┌────────────────────┐
│   Supabase         │  │   Upstash Redis    │
│   (PostgreSQL)     │  │   (Cache)          │
└────────────────────┘  └────────────────────┘
```

## Cost Breakdown

| Service | Provider | Cost/Month |
|---------|----------|------------|
| VPS (CX32) | Hetzner | €5.49 |
| Backups | Hetzner | €1.20 |
| Database | Supabase Free | $0 |
| Redis | Upstash Free | $0 |
| **Total** | | **~$8/mo** |

## Quick Links

- [Initial Setup Guide](./SETUP.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Operations Runbook](./RUNBOOK.md)
- [Environment Variables](./ENVIRONMENT.md)
```

**Create `docs/deployment/SETUP.md`:**
```markdown
# Initial Setup Guide

## Prerequisites

- Hetzner Cloud account
- Supabase account
- Upstash account
- GitHub repository with admin access
- Domain name (optional but recommended)

## Step 1: Provision Hetzner Server

1. Log into [Hetzner Cloud Console](https://console.hetzner.cloud/)
2. Create new project "nutri-production"
3. Add SSH key to project
4. Create server:
   - Location: Nearest to users (e.g., Helsinki, Frankfurt)
   - Image: Ubuntu 22.04
   - Type: CX32 (4 vCPU, 8GB RAM)
   - Networking: Public IPv4
   - SSH Key: Select your key
   - Backups: Enable ($1.20/mo)

## Step 2: Run Server Setup Script

```bash
ssh root@<server-ip>
curl -fsSL https://raw.githubusercontent.com/your-repo/nutri/master/scripts/deploy/setup-server.sh | bash
```

## Step 3: Configure Coolify

1. Access Coolify at `http://<server-ip>:8000`
2. Create admin account
3. Settings > General:
   - Set custom domain for Coolify dashboard
   - Enable SSL via Let's Encrypt
4. Add new source: GitHub (OAuth or Deploy Key)

[Continue with detailed steps...]
```

**Create `docs/deployment/TROUBLESHOOTING.md`:**
```markdown
# Troubleshooting Guide

## Common Issues

### Container won't start

**Symptoms:** Container status shows "Exited" or "Restarting"

**Solutions:**
1. Check logs: `docker logs <container-name>`
2. Verify environment variables in Coolify
3. Check health endpoint: `curl http://localhost:3000/health`
4. Verify database connection

### Database connection failed

**Symptoms:** "Connection refused" or timeout errors

**Solutions:**
1. Verify DATABASE_URL format
2. Check Supabase dashboard for connection limits
3. Ensure IP whitelist includes server IP
...
```

**Create `docs/deployment/RUNBOOK.md`:**
```markdown
# Operations Runbook

## Daily Operations

### Check System Health
```bash
curl https://api.nutri.app/health
```

### View Logs
```bash
# In Coolify dashboard, or:
docker logs -f nutri-backend --tail 100
```

## Emergency Procedures

### Rollback Deployment

1. Go to GitHub Actions
2. Find last successful deploy workflow
3. Click "Re-run jobs"

Or manually:
```bash
# SSH to server
docker pull ghcr.io/repo/backend:<previous-sha>
# Update image in Coolify
```
...
```

**Create `docs/deployment/ENVIRONMENT.md`:**
Document all environment variables with descriptions and examples.

**Test Strategy:**

1. Have team member follow SETUP.md on fresh account
2. Verify all links work
3. Test troubleshooting steps actually resolve issues
4. Review runbook procedures with ops team
5. Ensure architecture diagram matches actual deployment
6. Verify cost breakdown is accurate

## Subtasks

### 29.1. Create docs/deployment/README.md with architecture diagram and navigation

**Status:** pending  
**Dependencies:** None  

Create the main deployment documentation landing page with ASCII architecture diagram showing GitHub → CI/CD → Hetzner/Coolify → Supabase/Upstash flow, cost breakdown table, and quick links to other documentation files.

**Details:**

Create `docs/deployment/README.md` containing:
1. ASCII architecture diagram showing complete deployment pipeline (GitHub → GitHub Actions → Hetzner CX32 → Coolify → Services)
2. Cost breakdown table with Hetzner VPS (€5.49), Backups (€1.20), Supabase (free), Upstash (free), totaling ~$8/mo
3. Quick links section referencing SETUP.md, TROUBLESHOOTING.md, RUNBOOK.md, and ENVIRONMENT.md
4. Brief overview of the tech stack and deployment approach
5. Use clear markdown formatting with proper headers and code blocks

### 29.2. Write docs/deployment/SETUP.md with complete setup instructions

**Status:** pending  
**Dependencies:** 29.1  

Create comprehensive step-by-step setup guide covering Hetzner server provisioning, Coolify installation, service configuration, domain setup, SSL configuration, and GitHub integration.

**Details:**

Create `docs/deployment/SETUP.md` with:
1. Prerequisites section (accounts needed, access requirements, domain name)
2. Step 1: Hetzner server provisioning (location selection, CX32 specs, SSH key setup, backup enabling)
3. Step 2: Server setup script execution (SSH connection, running setup-server.sh)
4. Step 3: Coolify configuration (initial access, admin account, domain setup, SSL/Let's Encrypt)
5. Step 4: GitHub source integration (OAuth or Deploy Key setup)
6. Step 5: Supabase database setup (project creation, connection string, IP whitelist)
7. Step 6: Upstash Redis setup (database creation, connection URL)
8. Step 7: Service deployment in Coolify (environment variables, health checks, webhooks)
9. Step 8: DNS configuration and SSL verification
10. Include screenshots placeholders and troubleshooting tips inline

### 29.3. Create docs/deployment/TROUBLESHOOTING.md with common issues and solutions

**Status:** pending  
**Dependencies:** 29.1  

Document common deployment issues, their symptoms, root causes, and step-by-step solutions including container failures, database connection problems, health check failures, SSL certificate issues, and memory/resource problems.

**Details:**

Create `docs/deployment/TROUBLESHOOTING.md` with:
1. Container won't start (symptoms: Exited/Restarting status; solutions: check logs, verify env vars, test health endpoint, verify DB connection)
2. Database connection failed (symptoms: connection refused/timeout; solutions: verify DATABASE_URL format, check Supabase limits, IP whitelist, connection pooling)
3. Health check failures (symptoms: container marked unhealthy; solutions: verify /health endpoint, check dependencies, review logs)
4. SSL certificate issues (symptoms: HTTPS not working; solutions: Let's Encrypt rate limits, DNS propagation, Traefik config)
5. Out of memory errors (symptoms: container OOMKilled; solutions: check memory limits, optimize queries, review ML model sizes)
6. Coolify webhook not triggering (symptoms: no deployment after push; solutions: verify webhook URL, check GitHub Actions logs, test webhook manually)
7. Each issue includes: symptoms, possible causes, diagnostic commands, step-by-step solutions, prevention tips

### 29.4. Write docs/deployment/RUNBOOK.md with operations procedures

**Status:** pending  
**Dependencies:** 29.1  

Create operations runbook with daily health checks, log viewing, monitoring procedures, emergency rollback steps, scaling procedures, backup verification, and incident response workflows.

**Details:**

Create `docs/deployment/RUNBOOK.md` with:
1. Daily Operations: health check commands (curl health endpoints), log viewing (Coolify dashboard + docker logs), monitoring metrics to track
2. Emergency Procedures: rollback deployment (GitHub Actions re-run + manual docker image swap), service restart (Coolify UI + docker commands), database failover (Supabase dashboard)
3. Scaling Procedures: vertical scaling (Hetzner server resize), horizontal scaling (Coolify multi-container), database connection pool adjustments
4. Backup Verification: test backup restoration, verify automated backups running, document RTO/RPO
5. Incident Response: severity classification, notification procedures, escalation paths, post-mortem template
6. Maintenance Windows: planned downtime procedures, communication templates, zero-downtime deployment steps
7. Include actual commands with placeholders for environment-specific values

### 29.5. Document all environment variables in docs/deployment/ENVIRONMENT.md

**Status:** pending  
**Dependencies:** 29.1  

Create comprehensive environment variable reference documenting all required and optional variables for backend, ML service, and infrastructure with descriptions, examples, security notes, and validation rules.

**Details:**

Create `docs/deployment/ENVIRONMENT.md` with:
1. Backend API variables: NODE_ENV, DATABASE_URL, JWT_SECRET, JWT_EXPIRES_IN, PORT, REDIS_URL, ML_SERVICE_URL, CORS_ORIGIN
2. ML Service variables: DATABASE_URL, REDIS_URL, ML_MODEL_PATH, TORCH_DEVICE, PREDICTION_CACHE_TTL
3. Coolify/Infrastructure variables: COOLIFY_WEBHOOK_URL, GITHUB_TOKEN, DISCORD_WEBHOOK_URL
4. For each variable include: name, description, required/optional, example value (sanitized), default value, validation rules, security considerations
5. Security section: which variables are sensitive, how to rotate secrets, where secrets are stored (GitHub Secrets, Coolify env vars)
6. Environment-specific overrides: development vs staging vs production differences
7. Validation checklist: required variables by service, format validation commands
8. Reference existing .env.example and server/.env.example for accuracy
