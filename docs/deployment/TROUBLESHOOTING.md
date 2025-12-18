# Troubleshooting Guide

Common deployment issues and their solutions.

## Quick Diagnostics

Before diving into specific issues, run these diagnostic commands:

```bash
# SSH to server
ssh root@<server-ip>

# Check all containers
docker ps -a

# Check recent logs
docker logs nutri-backend --tail 50
docker logs nutri-ml-service --tail 50

# Check system resources
free -h
df -h
htop  # Press 'q' to exit
```

---

## Container Issues

### Container Won't Start

**Symptoms:**
- Container status shows "Exited" or "Restarting"
- Coolify shows deployment failed

**Diagnostic Steps:**

```bash
# Check container status
docker ps -a | grep nutri

# View container logs
docker logs nutri-backend --tail 100

# Check for OOM kills
dmesg | grep -i "killed process"

# Inspect container
docker inspect nutri-backend | jq '.[0].State'
```

**Common Causes & Solutions:**

1. **Missing Environment Variables**
   ```bash
   # Check env vars in container
   docker exec nutri-backend env | sort
   ```
   Solution: Verify all required env vars in Coolify service settings

2. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :3000
   ```
   Solution: Stop conflicting service or change port

3. **Database Connection Failed**
   - See "Database Connection Failed" section below

4. **Insufficient Memory**
   ```bash
   docker stats --no-stream
   ```
   Solution: Increase container memory limit or upgrade server

### Container Keeps Restarting

**Symptoms:**
- Container restarts every few seconds
- Logs show application crash

**Diagnostic Steps:**

```bash
# Watch container status
watch docker ps

# Follow logs
docker logs -f nutri-backend

# Check restart count
docker inspect nutri-backend --format '{{.RestartCount}}'
```

**Solutions:**

1. Check for unhandled exceptions in logs
2. Verify health check endpoint is responding
3. Check for infinite loops in startup code
4. Review memory limits vs actual usage

---

## Database Issues

### Database Connection Failed

**Symptoms:**
- `ECONNREFUSED` or `Connection refused` errors
- `Connection timed out` errors
- Health check shows database unhealthy

**Diagnostic Steps:**

```bash
# Test connection from server
docker exec nutri-backend sh -c 'nc -zv <db-host> 6543'

# Check DNS resolution
docker exec nutri-backend sh -c 'nslookup <db-host>'

# Test with psql (if available)
psql "$DATABASE_URL" -c "SELECT 1"
```

**Common Causes & Solutions:**

1. **Wrong DATABASE_URL Format**

   Correct format for Supabase:
   ```
   postgresql://postgres:[PASSWORD]@[PROJECT-REF].supabase.co:6543/postgres
   ```

   - Use port `6543` for connection pooling (recommended)
   - Use port `5432` for direct connection

2. **IP Not Whitelisted**

   Go to Supabase > Settings > Database > Network
   - Add your Hetzner server IP
   - Or allow all IPs: `0.0.0.0/0`

3. **Connection Pool Exhausted**

   Signs: Intermittent connection failures

   Solutions:
   - Reduce `pool_size` in application
   - Use Supabase connection pooler (port 6543)
   - Check for connection leaks

4. **Supabase Project Paused**

   Free tier projects pause after 7 days of inactivity.
   - Go to Supabase dashboard
   - Click "Restore project"

### Migration Failed

**Symptoms:**
- `prisma migrate deploy` errors
- GitHub Actions migration job fails

**Diagnostic Steps:**

```bash
# Check migration status
cd server
npx prisma migrate status

# View migration history
npx prisma migrate diff --from-schema-datasource prisma/schema.prisma --to-schema-datamodel prisma/schema.prisma
```

**Common Causes & Solutions:**

1. **Pending Migrations**
   ```bash
   npx prisma migrate deploy
   ```

2. **Schema Drift**
   ```bash
   # Generate diff
   npx prisma migrate diff \
     --from-schema-datasource prisma/schema.prisma \
     --to-schema-datamodel prisma/schema.prisma \
     --script
   ```

3. **Lock File Issues**
   ```sql
   -- Connect to database and check locks
   SELECT * FROM _prisma_migrations;
   ```

---

## Redis Issues

### Redis Connection Failed

**Symptoms:**
- `ECONNREFUSED` on Redis
- Cache operations failing
- Rate limiting not working

**Diagnostic Steps:**

```bash
# Test Redis connection
docker exec nutri-backend sh -c 'nc -zv <redis-host> 6379'

# Check with redis-cli (if TLS)
redis-cli -u $REDIS_URL ping
```

**Common Causes & Solutions:**

1. **Wrong REDIS_URL Format**

   Upstash format (TLS):
   ```
   rediss://default:[PASSWORD]@[HOST].upstash.io:6379
   ```
   Note: `rediss://` (with double 's') for TLS

2. **TLS Issues**

   Ensure application uses TLS for Upstash connection

3. **Rate Limit Exceeded**

   Free tier: 10,000 commands/day
   - Check Upstash dashboard for usage
   - Reduce cache operations or upgrade

---

## Health Check Issues

### Health Check Failing

**Symptoms:**
- Container marked "unhealthy"
- Coolify shows service degraded
- Load balancer removing instance

**Diagnostic Steps:**

```bash
# Test health endpoint manually
curl -v http://localhost:3000/health/live
curl -v http://localhost:3000/health

# Check from inside container
docker exec nutri-backend curl localhost:3000/health/live
```

**Common Causes & Solutions:**

1. **Application Not Ready**

   Health check runs before app is ready.
   - Increase `start_period` in health check config
   - Add readiness check separate from liveness

2. **Dependency Down**

   `/health` checks database and Redis.
   - Use `/health/live` for basic liveness
   - Check individual dependencies

3. **Wrong Health Check Port**

   Verify port matches application config

---

## SSL/TLS Issues

### SSL Certificate Not Working

**Symptoms:**
- Browser shows "Not Secure"
- `ERR_CERT_AUTHORITY_INVALID`
- `curl: (60) SSL certificate problem`

**Diagnostic Steps:**

```bash
# Check certificate
openssl s_client -connect api.yourdomain.com:443 -servername api.yourdomain.com

# Check expiry
echo | openssl s_client -connect api.yourdomain.com:443 2>/dev/null | openssl x509 -noout -dates

# Test from server
curl -vI https://api.yourdomain.com/health/live
```

**Common Causes & Solutions:**

1. **Let's Encrypt Rate Limit**

   Rate limits: 50 certificates per registered domain per week

   Solutions:
   - Wait and retry
   - Use staging environment for testing
   - Check certificate already exists

2. **DNS Not Propagated**

   ```bash
   dig api.yourdomain.com +short
   ```
   Wait for DNS propagation (up to 48 hours)

3. **Wrong Domain in Coolify**

   Verify domain configuration in Coolify matches actual domain

4. **Traefik Certificate Issue**

   ```bash
   # Check Traefik logs
   docker logs $(docker ps -q -f name=traefik) --tail 100
   ```

### Mixed Content Errors

**Symptoms:**
- Some resources blocked
- Console shows mixed content warnings

**Solutions:**
- Ensure all URLs use HTTPS
- Check `CORS_ORIGIN` uses HTTPS
- Update hard-coded HTTP URLs

---

## Memory Issues

### Out of Memory (OOM)

**Symptoms:**
- Container killed unexpectedly
- `OOMKilled: true` in container status
- Swap usage high

**Diagnostic Steps:**

```bash
# Check container memory
docker stats --no-stream

# Check system memory
free -h

# Check OOM logs
dmesg | grep -i "killed process"

# Check container limits
docker inspect nutri-backend --format '{{.HostConfig.Memory}}'
```

**Common Causes & Solutions:**

1. **Memory Leak**
   - Check for growing memory usage over time
   - Review application for memory leaks
   - Implement memory profiling

2. **ML Models Too Large**
   - Reduce model size
   - Use lazy loading
   - Consider model quantization

3. **Too Many Concurrent Requests**
   - Implement request queuing
   - Add rate limiting
   - Scale horizontally

4. **Insufficient Server Memory**
   - Upgrade Hetzner server (CX32 → CX42)
   - Or add swap space

---

## CI/CD Issues

### Webhook Not Triggering Deployment

**Symptoms:**
- Push to master doesn't deploy
- No deployment activity in Coolify

**Diagnostic Steps:**

1. Check GitHub Actions: Did workflows run?
2. Check Coolify webhook settings
3. Verify webhook secret matches

**Common Causes & Solutions:**

1. **Webhook URL Changed**
   - Regenerate webhook in Coolify
   - Update GitHub Secrets

2. **Branch Filter**
   - Verify workflow triggers on correct branch

   ```yaml
   on:
     push:
       branches: [master]
   ```

3. **Workflow Failed**
   - Check GitHub Actions logs
   - Fix failing tests/builds

### Build Fails in CI

**Symptoms:**
- GitHub Actions build job fails
- Docker build errors

**Common Causes & Solutions:**

1. **Dependency Issues**
   ```bash
   # Clear npm cache
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **TypeScript Errors**
   ```bash
   npm run build
   # Fix reported errors
   ```

3. **Docker Build Context**
   - Ensure `.dockerignore` is correct
   - Check all required files are in build context

---

## Networking Issues

### Service Can't Reach Another Service

**Symptoms:**
- Backend can't reach ML service
- `ECONNREFUSED` between services

**Diagnostic Steps:**

```bash
# Check Docker network
docker network ls
docker network inspect <network-name>

# Test connectivity
docker exec nutri-backend curl nutri-ml-service:8080/health
```

**Solutions:**

1. Ensure services are on same Docker network
2. Use service names (not localhost) for inter-service communication
3. Check Coolify network configuration

### External API Calls Failing

**Symptoms:**
- Third-party API calls timeout
- DNS resolution failing

**Diagnostic Steps:**

```bash
# Test DNS
docker exec nutri-backend nslookup api.external.com

# Test connectivity
docker exec nutri-backend curl -v https://api.external.com
```

**Solutions:**

1. Check firewall rules (UFW)
2. Verify DNS configuration in Docker
3. Check if IP is blocked by external service

---

## Quick Fix Commands

```bash
# Restart all containers
cd /data/coolify && docker compose restart

# Force recreate containers
docker compose up -d --force-recreate

# Clear Docker cache
docker system prune -af

# Restart Docker daemon
systemctl restart docker

# Restart Coolify
docker restart $(docker ps -q -f name=coolify)

# Check disk space
df -h

# Clear old logs
journalctl --vacuum-time=3d

# Check for zombie processes
ps aux | grep defunct
```

---

## Getting Help

If you can't resolve the issue:

1. **Collect diagnostic info:**
   ```bash
   # Create debug bundle
   mkdir /tmp/nutri-debug
   docker ps -a > /tmp/nutri-debug/containers.txt
   docker logs nutri-backend --tail 500 > /tmp/nutri-debug/backend.log
   docker logs nutri-ml-service --tail 500 > /tmp/nutri-debug/ml.log
   free -h > /tmp/nutri-debug/memory.txt
   df -h > /tmp/nutri-debug/disk.txt
   ```

2. **Check recent changes:**
   - What was deployed recently?
   - Any configuration changes?
   - Infrastructure changes?

3. **Review documentation:**
   - [SETUP.md](./SETUP.md) - Initial setup
   - [RUNBOOK.md](./RUNBOOK.md) - Operations procedures
   - [ENVIRONMENT.md](./ENVIRONMENT.md) - Environment variables

---

*Last updated: December 2025*
