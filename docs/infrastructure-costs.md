# Nutri Application - Infrastructure Cost Estimation

**Last Updated:** December 2025

This document provides detailed cost estimates for running the Nutri application infrastructure at various scales of Daily Active Users (DAU).

---

## Executive Summary

| Scale | Monthly Cost | Cost per DAU | Phase |
|-------|-------------|--------------|-------|
| **100 DAU** | $65-95 | $0.65-0.95 | MVP |
| **500 DAU** | $110-150 | $0.22-0.30 | Early Growth |
| **1,000 DAU** | $215-290 | $0.22-0.29 | Growth |
| **5,000 DAU** | $595-810 | $0.12-0.16 | Scale |

---

## Infrastructure Components

### Services Overview

| Service | Technology | Purpose |
|---------|-----------|---------|
| **Backend API** | Node.js/Express | REST API, authentication, data management |
| **ML Service** | Python/FastAPI | Feature engineering, LSTM predictions, correlations |
| **Database** | PostgreSQL 16 | Primary data store |
| **Cache** | Redis 7 | ML prediction caching, feature caching |

### Data Volume Per User

| Data Type | Rows/Day | Rows/Year | Approx Size |
|-----------|----------|-----------|-------------|
| Meals | 4-5 | 1,460-1,825 | - |
| Health Metrics | 20-100 | 7,300-36,500 | - |
| Activities | 1-2 | 365-730 | - |
| ML Features | 8 | 2,920 | - |
| ML Predictions | 3-5 | 1,095-1,825 | - |
| **Total** | 36-120 | 13,000-48,000 | **5-15 MB/user/year** |

---

## Vendor Pricing Comparison

### PostgreSQL Database Providers

| Provider | Plan | Specs | Monthly Cost | Notes |
|----------|------|-------|--------------|-------|
| **AWS RDS** | db.t4g.micro | 2 vCPU, 1 GB | ~$12 | + storage costs |
| **AWS RDS** | db.t4g.small | 2 vCPU, 2 GB | ~$24 | + storage costs |
| **AWS RDS** | db.t4g.medium | 2 vCPU, 4 GB | ~$48 | + storage costs |
| **AWS RDS** | db.r6g.large | 2 vCPU, 16 GB | ~$190 | Memory-optimized |
| **DigitalOcean** | Basic | 1 vCPU, 1 GB, 10 GB | $15 | Flat pricing |
| **DigitalOcean** | Basic | 1 vCPU, 2 GB, 25 GB | $30 | Flat pricing |
| **DigitalOcean** | Basic | 2 vCPU, 4 GB, 38 GB | $60 | Flat pricing |
| **Google Cloud SQL** | db-f1-micro | Shared, 0.6 GB | ~$10 | + storage |
| **Google Cloud SQL** | db-custom-2-4096 | 2 vCPU, 4 GB | ~$75 | + storage |
| **Supabase** | Free | 500 MB storage | $0 | 50K MAUs limit |
| **Supabase** | Pro | 8 GB storage | $25 | + $10 compute credits |
| **Neon** | Free | 0.5 GB, 191 hrs | $0 | Serverless, scale-to-zero |
| **Neon** | Launch | 10 GB, 300 hrs | $19 | Great for dev/staging |
| **Neon** | Scale | 50 GB, 750 hrs | $69 | Production ready |
| **Railway** | Usage-based | Pay per use | ~$5-20 | Variable workloads |
| **Render** | Starter | 1 GB RAM, 1 GB disk | $7 | Basic tier |
| **Render** | Standard | 2 GB RAM, 16 GB disk | $25 | Production tier |
| **Fly.io** | Managed | 1 GB RAM | ~$38 | HA included |
| **Fly.io** | Self-managed | Single node | ~$2 | Dev only |

**Sources:** [AWS RDS Pricing](https://aws.amazon.com/rds/postgresql/pricing/), [DigitalOcean Pricing](https://www.digitalocean.com/pricing/managed-databases), [Neon Pricing](https://neon.tech/pricing), [Supabase Pricing](https://supabase.com/pricing)

---

### Redis/Cache Providers

| Provider | Plan | Specs | Monthly Cost | Notes |
|----------|------|-------|--------------|-------|
| **AWS ElastiCache** | cache.t4g.micro | 0.5 GB | $11.68 | Managed |
| **AWS ElastiCache** | cache.t4g.small | 1.37 GB | $23.36 | Managed |
| **AWS ElastiCache** | cache.t4g.medium | 3.09 GB | $46.72 | Managed |
| **AWS ElastiCache** | cache.r6g.large | 13.07 GB | ~$150 | Memory-optimized |
| **Upstash** | Free | 256 MB, 500K cmd/mo | $0 | Serverless |
| **Upstash** | Pay-as-you-go | 1 GB | ~$10-30 | Per-request billing |
| **Upstash** | Pro | 10 GB | $280 | With Prod Pack |
| **DigitalOcean** | Basic | 1 GB | $15 | Managed Redis |
| **DigitalOcean** | Basic | 2 GB | $30 | Managed Redis |
| **Redis Cloud** | Free | 30 MB | $0 | Limited |
| **Redis Cloud** | Essentials | 250 MB | $5 | Fixed pricing |
| **Railway** | Usage-based | Pay per use | ~$5-15 | Variable |
| **Render** | Starter | 25 MB | $0 | Free tier |
| **Render** | Standard | 100 MB | $10 | Paid tier |

**Sources:** [AWS ElastiCache Pricing](https://aws.amazon.com/elasticache/pricing/), [Upstash Pricing](https://upstash.com/pricing/redis), [DigitalOcean Redis](https://www.digitalocean.com/pricing/managed-databases)

---

### Compute Instances (API/ML Service)

| Provider | Instance | vCPU | Memory | Monthly Cost | Notes |
|----------|----------|------|--------|--------------|-------|
| **AWS EC2** | t4g.micro | 2 | 1 GB | $6.13 | Graviton, free tier eligible |
| **AWS EC2** | t4g.small | 2 | 2 GB | $12.26 | Graviton |
| **AWS EC2** | t4g.medium | 2 | 4 GB | $24.53 | Graviton |
| **AWS EC2** | t4g.large | 2 | 8 GB | $49.06 | Graviton |
| **AWS EC2** | m6g.large | 2 | 8 GB | $70.08 | General purpose |
| **DigitalOcean** | Basic | 1 | 1 GB | $6 | Droplet |
| **DigitalOcean** | Basic | 1 | 2 GB | $12 | Droplet |
| **DigitalOcean** | Basic | 2 | 4 GB | $24 | Droplet |
| **DigitalOcean** | CPU-Optimized | 2 | 4 GB | $42 | For ML workloads |
| **Google Cloud** | e2-micro | 0.25 | 1 GB | ~$6.11 | Shared-core |
| **Google Cloud** | e2-small | 0.5 | 2 GB | ~$12.23 | Shared-core |
| **Google Cloud** | e2-medium | 1 | 4 GB | ~$24.46 | Shared-core |
| **Google Cloud** | n2-standard-2 | 2 | 8 GB | ~$49 | General purpose |
| **Render** | Starter | 0.5 | 512 MB | $7 | Web service |
| **Render** | Standard | 1 | 2 GB | $25 | Web service |
| **Render** | Pro | 2 | 4 GB | $85 | Web service |
| **Railway** | Usage-based | Variable | Variable | ~$5-25 | Pay for actual use |
| **Fly.io** | shared-cpu-1x | 1 | 256 MB | ~$2 | Entry level |
| **Fly.io** | shared-cpu-1x | 1 | 1 GB | ~$5 | With more RAM |
| **Fly.io** | performance-2x | 2 | 4 GB | ~$62 | Dedicated |

**Sources:** [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/), [DigitalOcean Droplets](https://www.digitalocean.com/pricing/droplets), [Render Pricing](https://render.com/pricing), [Railway Pricing](https://railway.com/pricing)

---

### GPU Instances for ML Training

| Provider | Instance | GPU | vCPU | Memory | Hourly | Monthly (730h) |
|----------|----------|-----|------|--------|--------|----------------|
| **AWS** | g4dn.xlarge | 1x T4 | 4 | 16 GB | $0.526 | $384 |
| **AWS** | g4dn.2xlarge | 1x T4 | 8 | 32 GB | $0.752 | $549 |
| **AWS** | p3.2xlarge | 1x V100 | 8 | 61 GB | $3.06 | $2,234 |
| **AWS** | p3.8xlarge | 4x V100 | 32 | 244 GB | $12.24 | $8,935 |
| **AWS** | p4d.24xlarge | 8x A100 | 96 | 1.1 TB | $32.77* | $23,922* |
| **Google Cloud** | n1-standard-4 + T4 | 1x T4 | 4 | 15 GB | ~$0.45 | ~$328 |
| **Google Cloud** | a2-highgpu-1g | 1x A100 | 12 | 85 GB | $3.67 | $2,679 |
| **Lambda Labs** | GPU Cloud | 1x A10 | 30 | 200 GB | $0.75 | $548 |
| **Lambda Labs** | GPU Cloud | 1x A100 | 30 | 200 GB | $1.29 | $942 |
| **RunPod** | Community | 1x 3090 | 8 | 32 GB | $0.44 | $321 |
| **RunPod** | Secure | 1x A100 | 16 | 125 GB | $1.89 | $1,380 |
| **Vast.ai** | Spot | 1x 3090 | Variable | Variable | ~$0.20 | ~$146 |

*Note: AWS announced up to 45% price reductions on P4/P5 instances in June 2025.*

**Sources:** [AWS GPU Pricing](https://aws.amazon.com/ec2/instance-types/p4/), [Lambda Labs](https://lambdalabs.com/service/gpu-cloud), [RunPod](https://www.runpod.io/gpu-instance/pricing)

---

### Load Balancers

| Provider | Type | Hourly | Monthly Base | Notes |
|----------|------|--------|--------------|-------|
| **AWS ALB** | Application LB | $0.0225 | ~$16.43 | + $0.008/LCU-hour |
| **AWS NLB** | Network LB | $0.0225 | ~$16.43 | + data processing |
| **Google Cloud** | HTTP(S) LB | $0.025 | ~$18.25 | + rules + data |
| **DigitalOcean** | Load Balancer | - | $12 | Simple pricing |
| **Render** | Included | - | $0 | With paid services |
| **Railway** | Included | - | $0 | With usage |
| **Fly.io** | Included | - | $0 | Anycast routing |

**Typical AWS ALB Monthly Cost:** $20-30 for moderate traffic (includes ~$5-10 in LCU charges)

**Sources:** [AWS ELB Pricing](https://aws.amazon.com/elasticloadbalancing/pricing/), [DigitalOcean LB](https://www.digitalocean.com/pricing)

---

## Detailed Cost Breakdown by Scale

### 100 DAU - MVP Phase

**Recommended for:** Initial launch, beta testing, validation

#### Option A: AWS (Traditional)
| Service | Instance | Cost/Month |
|---------|----------|------------|
| PostgreSQL | RDS db.t4g.small + 20GB | $28 |
| Redis | ElastiCache cache.t4g.micro | $12 |
| ML Service | EC2 t4g.small | $12 |
| Backend API | EC2 t4g.micro | $6 |
| **Total** | | **$58** |

#### Option B: DigitalOcean (Simpler)
| Service | Plan | Cost/Month |
|---------|------|------------|
| PostgreSQL | Managed 1GB | $15 |
| Redis | Managed 1GB | $15 |
| ML Service | Droplet 2GB | $12 |
| Backend API | Droplet 1GB | $6 |
| **Total** | | **$48** |

#### Option C: Serverless (Most Cost-Effective for Low Traffic)
| Service | Provider | Cost/Month |
|---------|----------|------------|
| PostgreSQL | Neon Launch | $19 |
| Redis | Upstash Free | $0 |
| ML Service | Railway | ~$10 |
| Backend API | Railway | ~$5 |
| **Total** | | **$34** |

---

### 500 DAU - Early Growth

#### Option A: AWS
| Service | Instance | Cost/Month |
|---------|----------|------------|
| PostgreSQL | RDS db.t4g.medium + 50GB | $55 |
| Redis | ElastiCache cache.t4g.small | $23 |
| ML Service | EC2 t4g.small | $12 |
| Backend API | EC2 t4g.small | $12 |
| **Total** | | **$102** |

#### Option B: DigitalOcean
| Service | Plan | Cost/Month |
|---------|------|------------|
| PostgreSQL | Managed 4GB | $60 |
| Redis | Managed 2GB | $30 |
| ML Service | Droplet 4GB | $24 |
| Backend API | Droplet 2GB | $12 |
| **Total** | | **$126** |

#### Option C: Hybrid (Best Value)
| Service | Provider | Cost/Month |
|---------|----------|------------|
| PostgreSQL | Neon Scale | $69 |
| Redis | Upstash Pay-as-you-go | $15 |
| ML Service | Railway | ~$20 |
| Backend API | Render Standard | $25 |
| **Total** | | **$129** |

---

### 1,000 DAU - Growth Phase

#### Option A: AWS (Production-Ready)
| Service | Instance | Cost/Month |
|---------|----------|------------|
| PostgreSQL Primary | RDS db.t4g.medium + 100GB | $65 |
| PostgreSQL Replica | RDS db.t4g.small | $28 |
| Redis (HA) | 2x ElastiCache cache.t4g.small | $47 |
| ML Service | 2x EC2 t4g.small | $25 |
| Backend API | 2x EC2 t4g.small | $25 |
| Load Balancer | ALB | $25 |
| **Total** | | **$215** |

#### Option B: DigitalOcean (Simpler HA)
| Service | Plan | Cost/Month |
|---------|------|------------|
| PostgreSQL | Managed HA 4GB | $95 |
| Redis | Managed HA 2GB | $50 |
| ML Service | 2x Droplet 4GB | $48 |
| Backend API | 2x Droplet 2GB | $24 |
| Load Balancer | DO Load Balancer | $12 |
| **Total** | | **$229** |

---

### 5,000 DAU - Scale Phase

#### Option A: AWS (Full Production)
| Service | Instance | Cost/Month |
|---------|----------|------------|
| PostgreSQL Primary | RDS db.r6g.large + 200GB | $220 |
| PostgreSQL Replicas | 2x RDS db.r6g.large | $380 |
| Redis Cluster | 3x ElastiCache cache.t4g.medium | $140 |
| ML Service | 4x EC2 t4g.medium | $98 |
| Backend API | 4x EC2 t4g.medium | $98 |
| Load Balancer | ALB | $35 |
| Message Queue | SQS | $25 |
| Monitoring | CloudWatch | $30 |
| **Total** | | **$1,026** |

#### Option B: With GPU for ML Training
| Service | Instance | Cost/Month |
|---------|----------|------------|
| Base Infrastructure (above) | - | $1,026 |
| GPU Instance (Spot) | g4dn.xlarge @ 30% | ~$115 |
| **Total** | | **$1,141** |

---

## Cost Optimization Strategies

### Reserved Instance Savings

| Commitment | AWS Savings | GCP Savings | Notes |
|------------|-------------|-------------|-------|
| No commitment | 0% | 0% | On-demand pricing |
| 1-year | 30-40% | 25-30% | Partial upfront |
| 3-year | 50-60% | 45-52% | All upfront |
| Spot/Preemptible | 60-90% | 60-80% | Can be interrupted |

### Savings by Strategy

| Strategy | Potential Savings | Trade-off |
|----------|------------------|-----------|
| **Spot Instances** (ML training) | 60-90% | Job interruptions |
| **Reserved Instances** (1-year) | 30-40% | Upfront commitment |
| **Graviton/ARM** (AWS) | 20% | Already using t4g |
| **Serverless DB** (low traffic) | 40-60% | Cold start latency |
| **Aggressive Caching** | 15-20% | Stale predictions |
| **Off-peak Batch Processing** | 10-15% | Delayed insights |
| **Rightsizing** | 10-20% | Requires monitoring |

---

## Provider Recommendation by Phase

### MVP (100-500 DAU)
**Recommended: Railway + Neon + Upstash**
- Total: ~$35-70/month
- Pros: Zero ops, scale-to-zero, pay for actual usage
- Cons: Less control, potential cold starts

### Growth (500-2,000 DAU)
**Recommended: DigitalOcean or AWS**
- Total: ~$100-250/month
- Pros: Predictable costs, managed services, good support
- Cons: Less serverless flexibility

### Scale (2,000+ DAU)
**Recommended: AWS or GCP**
- Total: ~$500-1,200/month
- Pros: Full control, HA, auto-scaling, GPU options
- Cons: Complexity, requires DevOps expertise

---

## Scaling Bottlenecks

| DAU Range | Primary Bottleneck | Solution | Additional Cost |
|-----------|-------------------|----------|-----------------|
| 500-1,000 | ML model training (CPU-bound) | Async job queue (SQS) | +$20-30/mo |
| 1,000-2,000 | Database connections | PgBouncer connection pooling | +$20/mo |
| 2,000-5,000 | LSTM training time | GPU instances (g4dn.xlarge) | +$115-400/mo |
| 5,000+ | Data volume (35GB+) | Database sharding, CDN | +$300-500/mo |

---

## Additional Costs to Consider

| Item | Estimated Cost | Vendors |
|------|---------------|---------|
| **Domain** | $10-15/year | Namecheap, Cloudflare |
| **SSL Certificate** | $0 | Let's Encrypt, Cloudflare |
| **Email Service** | $0-35/mo | SendGrid (free tier), AWS SES ($0.10/1K) |
| **Error Tracking** | $0-26/mo | Sentry (free tier), Bugsnag |
| **Analytics** | $0-100/mo | Mixpanel (free), Amplitude (free), PostHog |
| **CI/CD** | $0-50/mo | GitHub Actions (free), CircleCI |
| **APM** | $0-100/mo | New Relic (free tier), Datadog |
| **Backups (S3)** | $2-10/mo | AWS S3 ($0.023/GB) |
| **CDN** | $0-20/mo | Cloudflare (free), AWS CloudFront |
| **Push Notifications** | $0-25/mo | Firebase (free), OneSignal |

---

## Total Cost Summary by Vendor

### 100 DAU Comparison

| Vendor Stack | Monthly Cost | Pros | Cons |
|--------------|-------------|------|------|
| **Serverless** (Railway+Neon+Upstash) | $34 | Cheapest, zero-ops | Cold starts |
| **DigitalOcean** | $48 | Simple, predictable | Manual scaling |
| **AWS** | $58 | Full control | Most complex |
| **Render** | $57 | Developer-friendly | Limited flexibility |

### 1,000 DAU Comparison

| Vendor Stack | Monthly Cost | Pros | Cons |
|--------------|-------------|------|------|
| **AWS** | $215 | HA, auto-scaling, GPU path | Complex |
| **DigitalOcean** | $229 | Simpler HA | No GPU options |
| **GCP** | $225 | Good ML tools | Learning curve |
| **Mixed** (Neon+Railway+AWS) | $200 | Best of each | Integration work |

---

## Appendix: Instance Reference

### AWS EC2 Graviton (t4g) Instances

| Type | vCPU | Memory | On-Demand $/hr | Monthly |
|------|------|--------|----------------|---------|
| t4g.nano | 2 | 0.5 GB | $0.0042 | $3.07 |
| t4g.micro | 2 | 1 GB | $0.0084 | $6.13 |
| t4g.small | 2 | 2 GB | $0.0168 | $12.26 |
| t4g.medium | 2 | 4 GB | $0.0336 | $24.53 |
| t4g.large | 2 | 8 GB | $0.0672 | $49.06 |
| t4g.xlarge | 4 | 16 GB | $0.1344 | $98.11 |

### AWS RDS PostgreSQL Instances

| Type | vCPU | Memory | On-Demand $/hr | Monthly |
|------|------|--------|----------------|---------|
| db.t4g.micro | 2 | 1 GB | $0.016 | $11.68 |
| db.t4g.small | 2 | 2 GB | $0.032 | $23.36 |
| db.t4g.medium | 2 | 4 GB | $0.065 | $47.45 |
| db.t4g.large | 2 | 8 GB | $0.129 | $94.17 |
| db.r6g.large | 2 | 16 GB | $0.26 | $189.80 |
| db.r6g.xlarge | 4 | 32 GB | $0.52 | $379.60 |

### AWS ElastiCache Redis Instances

| Type | vCPU | Memory | On-Demand $/hr | Monthly |
|------|------|--------|----------------|---------|
| cache.t4g.micro | 2 | 0.5 GB | $0.016 | $11.68 |
| cache.t4g.small | 2 | 1.37 GB | $0.032 | $23.36 |
| cache.t4g.medium | 2 | 3.09 GB | $0.064 | $46.72 |
| cache.r6g.large | 2 | 13.07 GB | $0.206 | $150.38 |
| cache.r6g.xlarge | 4 | 26.32 GB | $0.412 | $300.76 |

---

## Monitoring & Scaling Triggers

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU utilization | >60% | >80% | Scale up/out |
| Memory utilization | >70% | >85% | Upgrade instance |
| DB connections | >70% | >85% | Add PgBouncer |
| Response time p95 | >300ms | >500ms | Investigate & scale |
| Error rate | >0.5% | >1% | Investigate immediately |
| Disk usage | >70% | >85% | Expand storage |
| Cache hit ratio | <70% | <50% | Review cache strategy |

---

## References & Tools

### Pricing Calculators
- [AWS Pricing Calculator](https://calculator.aws/)
- [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)
- [DigitalOcean Pricing Calculator](https://www.digitalocean.com/pricing/calculator)
- [Vantage Instance Comparison](https://instances.vantage.sh/)

### Cost Monitoring Tools
- [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)
- [Infracost](https://www.infracost.io/) - Infrastructure cost estimates in CI/CD
- [CloudHealth](https://www.cloudhealthtech.com/)
- [Spot.io](https://spot.io/) - Spot instance management

---

*Last updated: December 2025. Prices are approximate and vary by region. Always verify current pricing with vendors.*

*This document should be reviewed quarterly and updated based on actual usage patterns.*
