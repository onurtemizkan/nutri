# Operations Runbook

Daily operations procedures, emergency responses, and maintenance tasks.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Emergency Procedures](#emergency-procedures)
3. [Scaling Procedures](#scaling-procedures)
4. [Backup & Recovery](#backup--recovery)
5. [Maintenance Windows](#maintenance-windows)
6. [Incident Response](#incident-response)

---

## Daily Operations

### Morning Health Check

Run these checks at the start of each day (or automate with monitoring):

```bash
# 1. Check all services are running
curl -s https://api.yourdomain.com/health | jq '.status'
# Expected: "healthy"

curl -s https://ml.yourdomain.com/health | jq '.status'
# Expected: "healthy"

# 2. Quick latency check
time curl -s https://api.yourdomain.com/health/live > /dev/null
# Expected: < 500ms
```

### View Application Logs

**Via Coolify Dashboard:**
1. Go to Coolify dashboard
2. Select service (Backend or ML Service)
3. Click "Logs" tab
4. Use search/filter as needed

**Via SSH:**

```bash
ssh root@<server-ip>

# Backend logs (last 100 lines)
docker logs nutri-backend --tail 100

# Follow logs in real-time
docker logs -f nutri-backend

# Filter for errors
docker logs nutri-backend 2>&1 | grep -i error

# ML Service logs
docker logs nutri-ml-service --tail 100

# Logs with timestamps
docker logs nutri-backend --timestamps --tail 50
```

### Check Resource Usage

```bash
# Container stats
docker stats --no-stream

# System memory
free -h

# Disk usage
df -h

# Top processes
htop
```

### Monitor Request Volume

Check structured logs for request counts and latency:

```bash
# Count requests in last hour
docker logs nutri-backend --since 1h 2>&1 | grep "request_completed" | wc -l

# Find slow requests (> 1000ms)
docker logs nutri-backend --since 1h 2>&1 | grep "request_completed" | jq 'select(.duration_ms > 1000)'
```

---

## Emergency Procedures

### Service Down - Complete Outage

**Severity: Critical**
**Response Time: Immediate**

1. **Verify outage:**
   ```bash
   curl -I https://api.yourdomain.com/health/live
   # If timeout or 5xx, proceed
   ```

2. **Check container status:**
   ```bash
   ssh root@<server-ip>
   docker ps -a
   ```

3. **Restart affected service:**
   ```bash
   docker restart nutri-backend
   # Wait 30 seconds
   curl https://api.yourdomain.com/health/live
   ```

4. **If restart fails, check logs:**
   ```bash
   docker logs nutri-backend --tail 200
   ```

5. **If still failing, rollback:**
   See [Rollback Deployment](#rollback-deployment)

### Rollback Deployment

**When to use:** New deployment causes issues

1. **Find previous working image:**
   ```bash
   # List recent images
   docker images ghcr.io/onurtemizkan/nutri/backend --format "{{.Tag}} {{.CreatedAt}}" | head -10
   ```

2. **Via GitHub Actions (Preferred):**
   - Go to Actions > "Deploy to Production"
   - Find last successful run before the problem
   - Click "Re-run all jobs"

3. **Via Coolify:**
   - Go to service settings
   - Change image tag to previous version
   - Redeploy

4. **Via Docker (Emergency):**
   ```bash
   # Pull previous image
   docker pull ghcr.io/onurtemizkan/nutri/backend:<previous-sha>

   # Stop current container
   docker stop nutri-backend

   # Start with old image
   docker run -d --name nutri-backend-rollback \
     --env-file /path/to/env \
     -p 3000:3000 \
     ghcr.io/onurtemizkan/nutri/backend:<previous-sha>
   ```

5. **Verify rollback:**
   ```bash
   curl https://api.yourdomain.com/health
   ```

### Database Emergency

**Symptoms:** Database connection errors, data issues

1. **Check Supabase status:**
   - Go to [Supabase Status](https://status.supabase.com/)
   - Check your project dashboard

2. **If project paused (free tier):**
   - Go to Supabase dashboard
   - Click "Restore project"
   - Wait for restoration (~2-5 minutes)

3. **If connection limit reached:**
   ```bash
   # Reduce active connections
   docker restart nutri-backend
   docker restart nutri-ml-service
   ```

4. **If data corruption suspected:**
   - Stop all writes immediately
   - Contact Supabase support
   - Restore from backup if needed

### Server Unresponsive

1. **Try SSH access:**
   ```bash
   ssh root@<server-ip>
   # If timeout, proceed to step 2
   ```

2. **Access via Hetzner Console:**
   - Log into Hetzner Cloud Console
   - Select server
   - Click "Console" for VNC access

3. **Force restart if needed:**
   - Hetzner Console > Server > Power > Reset

4. **After restart:**
   - Verify SSH access
   - Check Docker services started
   - Verify health endpoints

---

## Scaling Procedures

### Vertical Scaling (Upgrade Server)

**When:** Consistent high CPU/memory usage

1. **Choose new server type:**
   | Current | Upgrade To | Specs |
   |---------|------------|-------|
   | CX32 | CX42 | 8 vCPU, 16GB RAM |
   | CX42 | CX52 | 16 vCPU, 32GB RAM |

2. **Schedule maintenance window:**
   - Notify users if possible
   - Plan for ~15-30 minutes downtime

3. **Resize in Hetzner:**
   - Stop server (required for resize)
   - Click "Resize"
   - Select new server type
   - Start server

4. **Verify services:**
   ```bash
   ssh root@<server-ip>
   docker ps
   curl https://api.yourdomain.com/health
   ```

### Horizontal Scaling (Multiple Containers)

For stateless services (Backend API):

1. **In Coolify:**
   - Go to service settings
   - Set "Number of replicas" to desired count
   - Traefik handles load balancing automatically

2. **Verify load balancing:**
   ```bash
   # Multiple requests should hit different containers
   for i in {1..10}; do
     curl -s https://api.yourdomain.com/health | jq '.container_id'
   done
   ```

### Database Connection Pool Adjustment

**When:** Connection pool exhausted errors

```typescript
// In prisma configuration
datasources: {
  db: {
    url: env("DATABASE_URL"),
    connectionLimit: 10  // Adjust based on needs
  }
}
```

For Supabase, use connection pooler (port 6543) which supports more connections.

---

## Backup & Recovery

### Automatic Backups

**Hetzner Server Backups:**
- Enabled during setup ($1.20/mo)
- Daily snapshots, 7-day retention
- Restore via Hetzner Console

**Supabase Database Backups:**
- Free tier: Point-in-time recovery for 7 days
- Access via Supabase Dashboard > Database > Backups

### Manual Database Backup

```bash
# Export database
pg_dump "$DATABASE_URL" > backup-$(date +%Y%m%d).sql

# Compress
gzip backup-$(date +%Y%m%d).sql

# Upload to safe location
# (Use your preferred cloud storage)
```

### Verify Backups

Run monthly backup verification:

1. **Restore to test environment:**
   ```bash
   # Create test database in Supabase
   # Restore backup
   psql "$TEST_DATABASE_URL" < backup.sql
   ```

2. **Verify data integrity:**
   ```bash
   # Run test queries
   psql "$TEST_DATABASE_URL" -c "SELECT COUNT(*) FROM users;"
   ```

### Disaster Recovery

**RTO (Recovery Time Objective):** 1 hour
**RPO (Recovery Point Objective):** 24 hours (daily backups)

**Full Recovery Procedure:**

1. Provision new Hetzner server
2. Run `setup-server.sh`
3. Configure Coolify
4. Restore Supabase from backup
5. Deploy services
6. Update DNS

---

## Maintenance Windows

### Planned Maintenance

**Schedule:** Weekends, low-traffic hours (recommend Sunday 2-4 AM UTC)

**Communication:**
1. Announce 48 hours in advance
2. Post maintenance banner in app
3. Send notification to users (if applicable)

### Zero-Downtime Deployment

For routine deployments, use rolling updates:

1. **Coolify handles this automatically:**
   - New container starts
   - Health check passes
   - Traffic switches
   - Old container stops

2. **Verify no dropped requests:**
   ```bash
   # Monitor during deployment
   while true; do
     curl -s -w '%{time_total}s\n' https://api.yourdomain.com/health/live
     sleep 1
   done
   ```

### Database Migration Maintenance

For breaking schema changes:

1. **Put application in maintenance mode:**
   ```bash
   # Option: Enable maintenance middleware
   # Or: Scale replicas to 0 in Coolify
   ```

2. **Run migration:**
   ```bash
   cd server
   DATABASE_URL="$PROD_DATABASE_URL" npx prisma migrate deploy
   ```

3. **Deploy updated application:**

4. **Disable maintenance mode:**

5. **Verify:**
   ```bash
   curl https://api.yourdomain.com/health
   ```

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 - Critical | Complete outage | 15 minutes | Service down, data loss |
| P2 - High | Major feature broken | 1 hour | Auth failing, payments broken |
| P3 - Medium | Minor feature affected | 4 hours | Slow performance, minor bugs |
| P4 - Low | Cosmetic/minor | 24 hours | UI glitches, typos |

### Incident Workflow

1. **Detect:**
   - Automated monitoring alert
   - User report
   - Manual check

2. **Acknowledge:**
   - Assign incident owner
   - Communicate status

3. **Diagnose:**
   - Check health endpoints
   - Review logs
   - Identify root cause

4. **Mitigate:**
   - Apply fix or rollback
   - Verify service restored

5. **Communicate:**
   - Update stakeholders
   - Post to status page (if applicable)

6. **Post-mortem:**
   - Document incident
   - Identify improvements
   - Create action items

### Post-Mortem Template

```markdown
# Incident Post-Mortem: [Title]

**Date:** YYYY-MM-DD
**Duration:** X hours Y minutes
**Severity:** P1/P2/P3/P4
**Owner:** [Name]

## Summary
Brief description of what happened.

## Timeline
- HH:MM - First alert/report
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Service restored

## Root Cause
Detailed explanation of why this happened.

## Impact
- Users affected: X
- Requests failed: Y
- Revenue impact: $Z (if applicable)

## Resolution
What was done to fix the issue.

## Action Items
- [ ] Short-term fix
- [ ] Long-term improvement
- [ ] Monitoring improvement
- [ ] Documentation update

## Lessons Learned
What we learned from this incident.
```

---

## Quick Reference Commands

```bash
# ============== Health Checks ==============
curl https://api.yourdomain.com/health
curl https://api.yourdomain.com/health/live
curl https://ml.yourdomain.com/health

# ============== Container Management ==============
docker ps -a                              # List all containers
docker restart nutri-backend              # Restart backend
docker restart nutri-ml-service           # Restart ML service
docker logs nutri-backend --tail 100      # View logs
docker logs -f nutri-backend              # Follow logs
docker stats                              # Resource usage

# ============== System ==============
ssh root@<server-ip>                      # Connect to server
free -h                                   # Memory usage
df -h                                     # Disk usage
htop                                      # Process monitor

# ============== Emergency ==============
docker stop nutri-backend && docker start nutri-backend  # Hard restart
docker system prune -af                   # Clear Docker cache
systemctl restart docker                  # Restart Docker daemon

# ============== Database ==============
# Run migrations
cd /path/to/server && npx prisma migrate deploy

# Check migration status
npx prisma migrate status
```

---

*Last updated: December 2025*
