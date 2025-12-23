# Hetzner Deployment Guide

Deploy Nutri to Hetzner Cloud or Dedicated Servers with AMD EPYC optimization.

## Recommended Instances

| Stage | Instance | Specs | Cost | Expected Latency |
|-------|----------|-------|------|------------------|
| **MVP** | CPX31 | 4 vCPU, 8GB RAM | €9/mo | 150-200ms |
| **Growth** | CCX23 | 4 dedicated, 16GB RAM | €25/mo | 100-150ms |
| **Scale** | GEX44 | RTX 4000 Ada GPU | €184/mo | 15-30ms |

## Quick Start

### 1. Provision Server

```bash
# Create Hetzner Cloud server via CLI or Console
hcloud server create \
  --name nutri-prod \
  --type cpx31 \
  --image docker-ce \
  --location fsn1
```

### 2. SSH and Clone

```bash
ssh root@YOUR_SERVER_IP

# Install Docker Compose (if not included)
apt update && apt install -y docker-compose-plugin

# Clone repository
git clone https://github.com/YOUR_USERNAME/nutri.git
cd nutri
```

### 3. Configure Environment

```bash
# Copy and edit production environment
cp ml-service/.env.prod.example .env.prod

# Edit with your values
nano .env.prod
```

**Required changes in `.env.prod`:**
- `POSTGRES_PASSWORD` - Strong random password
- `JWT_SECRET` - At least 32 character random string
- Adjust `OMP_NUM_THREADS` based on your instance

### 4. Deploy

```bash
# Build and start all services
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build

# Watch ML service startup (wait for "ml_service_ready")
docker compose -f docker-compose.prod.yml logs -f ml-service

# Verify all services are healthy
docker compose -f docker-compose.prod.yml ps
```

### 5. Verify Deployment

```bash
# Test backend health
curl http://localhost:3000/health

# Test ML service (through backend)
curl -X POST http://localhost:3000/api/food/analyze \
  -F "image=@test_image.jpg"
```

## Instance-Specific Configuration

### CPX21 (3 vCPU, 4GB RAM) - €4.50/mo

```bash
# In .env.prod
OMP_NUM_THREADS=3
MKL_NUM_THREADS=3
TORCH_NUM_THREADS=3
```

```yaml
# In docker-compose.prod.yml, adjust ml-service resources:
deploy:
  resources:
    limits:
      cpus: '2.5'
      memory: 3G
```

### CPX31 (4 vCPU, 8GB RAM) - €9/mo (Recommended for MVP)

Default configuration works as-is.

### CCX23 (4 dedicated, 16GB RAM) - €25/mo

```yaml
# Can increase ML service resources:
deploy:
  resources:
    limits:
      cpus: '3.5'
      memory: 12G
```

## Production Checklist

- [ ] Strong `POSTGRES_PASSWORD` (use `openssl rand -hex 32`)
- [ ] Strong `JWT_SECRET` (use `openssl rand -hex 32`)
- [ ] Firewall configured (allow only 80, 443, 22)
- [ ] HTTPS configured (use Caddy or nginx-proxy)
- [ ] Backup strategy for PostgreSQL volume
- [ ] Monitoring setup (optional: Prometheus + Grafana)

## HTTPS with Caddy (Recommended)

```bash
# Add Caddy to docker-compose.prod.yml or run separately
docker run -d \
  --name caddy \
  --network nutri-prod-network \
  -p 80:80 -p 443:443 \
  -v caddy_data:/data \
  -v caddy_config:/config \
  caddy:2-alpine \
  caddy reverse-proxy --from your-domain.com --to backend:3000
```

## Monitoring Commands

```bash
# View all logs
docker compose -f docker-compose.prod.yml logs -f

# View specific service
docker compose -f docker-compose.prod.yml logs -f ml-service

# Check resource usage
docker stats

# Restart ML service (if needed)
docker compose -f docker-compose.prod.yml restart ml-service
```

## Updating

```bash
cd nutri
git pull

# Rebuild and restart
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build

# Or just rebuild ML service
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build ml-service
```

## Troubleshooting

### ML Service Won't Start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs ml-service

# Common issues:
# - Out of memory: Upgrade instance or reduce workers
# - Model download failed: Check internet connectivity
# - Database connection: Verify POSTGRES_PASSWORD matches
```

### Slow First Request

First request after restart takes ~2 minutes (model loading). This is normal.
Subsequent requests should be 150-200ms on CPX31.

### Out of Memory

```bash
# Check memory usage
docker stats --no-stream

# If ml-service is using too much:
# 1. Ensure FAST_MODE=true
# 2. Reduce workers in Dockerfile CMD
# 3. Upgrade to larger instance
```

## Cost Optimization

1. **Start with CPX31** (€9/mo) - sufficient for MVP
2. **Use FAST_MODE=true** - 18x faster, no quality loss for single foods
3. **HuggingFace cache volume** - saves 2 min on restarts
4. **Scale horizontally** before upgrading to GPU

## Performance Benchmarks

| Instance | FAST_MODE | Latency (p50) | Latency (p99) | Throughput |
|----------|-----------|---------------|---------------|------------|
| CPX31 | true | 170ms | 250ms | ~6 req/s |
| CCX23 | true | 120ms | 180ms | ~8 req/s |
| GEX44 | false | 25ms | 50ms | ~40 req/s |
