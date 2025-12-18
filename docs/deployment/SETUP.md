# Initial Setup Guide

Complete guide for setting up the Nutri production environment from scratch.

## Prerequisites

Before starting, ensure you have:

- [ ] **Hetzner Cloud account** - [Sign up](https://console.hetzner.cloud/)
- [ ] **Supabase account** - [Sign up](https://supabase.com/)
- [ ] **Upstash account** - [Sign up](https://upstash.com/)
- [ ] **GitHub repository access** - Admin or maintain permissions
- [ ] **SSH key pair** - For server access
- [ ] **Domain name** (optional) - For custom URLs

## Step 1: Provision Hetzner Server

### 1.1 Create Hetzner Project

1. Log into [Hetzner Cloud Console](https://console.hetzner.cloud/)
2. Click **"+ New Project"**
3. Name it `nutri-production`
4. Click **"Add Project"**

### 1.2 Add SSH Key

1. Go to **Security** > **SSH Keys**
2. Click **"Add SSH Key"**
3. Paste your public key (`~/.ssh/id_rsa.pub` or `~/.ssh/id_ed25519.pub`)
4. Give it a descriptive name

### 1.3 Create Server

1. Click **"Add Server"**
2. Configure:

| Setting | Value |
|---------|-------|
| Location | Choose nearest to users (Helsinki, Frankfurt, etc.) |
| Image | Ubuntu 22.04 |
| Type | CX32 (4 vCPU, 8 GB RAM) |
| Networking | Public IPv4 (checked) |
| SSH Keys | Select your key |
| Backups | Enable (~$1.20/mo) |
| Name | `nutri-prod-1` |

3. Click **"Create & Buy now"**
4. **Save the IP address** - you'll need it for DNS and Coolify

### 1.4 Verify SSH Access

```bash
ssh root@<server-ip>
# Should connect without password
exit
```

## Step 2: Run Server Setup Script

### 2.1 Execute Setup

```bash
ssh root@<server-ip>

# Option A: Run directly from GitHub
curl -fsSL https://raw.githubusercontent.com/onurtemizkan/nutri/master/scripts/deploy/setup-server.sh | bash

# Option B: Download and review first
curl -fsSL https://raw.githubusercontent.com/onurtemizkan/nutri/master/scripts/deploy/setup-server.sh -o setup-server.sh
chmod +x setup-server.sh
cat setup-server.sh  # Review the script
./setup-server.sh
```

### 2.2 What the Script Does

- Updates system packages
- Installs essential tools
- Configures UFW firewall (ports 22, 80, 443, 8000)
- Sets up fail2ban for SSH protection
- Hardens SSH configuration
- Enables automatic security updates
- Creates 4GB swap file
- Installs Docker
- Sets up Docker cleanup cron
- Installs Coolify

### 2.3 Verify Setup

After the script completes:

```bash
# Check firewall
ufw status verbose

# Check fail2ban
fail2ban-client status sshd

# Check Docker
docker --version
docker compose version

# Check swap
free -h
```

## Step 3: Configure Coolify

### 3.1 Initial Access

1. Open browser: `http://<server-ip>:8000`
2. Create admin account:
   - Email: Your email
   - Password: Strong password (save in password manager)
3. Complete initial setup wizard

### 3.2 Configure SSL for Dashboard

1. Go to **Settings** > **Configuration**
2. Set **FQDN**: `coolify.yourdomain.com` (or use server IP)
3. Enable **Let's Encrypt** if using domain
4. Save and restart if prompted

### 3.3 Add GitHub Source

1. Go to **Sources** > **Add New**
2. Select **GitHub App** (recommended) or **Deploy Key**

**For GitHub App:**
1. Click **"Create GitHub App"**
2. Follow the OAuth flow
3. Install app on your repository

**For Deploy Key:**
1. Select **Deploy Key**
2. Copy the generated public key
3. Add to GitHub repo: Settings > Deploy keys > Add deploy key
4. Check "Allow write access"

## Step 4: Set Up Supabase Database

### 4.1 Create Project

1. Go to [Supabase Dashboard](https://app.supabase.com/)
2. Click **"New Project"**
3. Configure:
   - Name: `nutri-production`
   - Database Password: Generate strong password (save this!)
   - Region: Choose closest to Hetzner server
4. Click **"Create new project"**
5. Wait for project to be ready (~2 minutes)

### 4.2 Get Connection String

1. Go to **Settings** > **Database**
2. Find **Connection string** section
3. Copy the **URI** (Connection pooling recommended for production)
4. Format: `postgresql://postgres:[PASSWORD]@[HOST]:6543/postgres`

### 4.3 Configure Network Access

Supabase allows all IPs by default on free tier. For additional security:
1. Go to **Settings** > **Database** > **Network**
2. Add your Hetzner server IP if restricting

## Step 5: Set Up Upstash Redis

### 5.1 Create Database

1. Go to [Upstash Console](https://console.upstash.com/)
2. Click **"Create Database"**
3. Configure:
   - Name: `nutri-production`
   - Region: Choose closest to Hetzner server
   - TLS: Enabled (recommended)
4. Click **"Create"**

### 5.2 Get Connection Details

1. Click on your database
2. Find **REST API** section
3. Copy:
   - **UPSTASH_REDIS_REST_URL**
   - **UPSTASH_REDIS_REST_TOKEN**

Or for native Redis connection:
- Connection string: `rediss://default:[PASSWORD]@[HOST]:6379`

## Step 6: Configure GitHub Secrets

### 6.1 Required Secrets

Go to GitHub repo > **Settings** > **Secrets and variables** > **Actions**

Add these secrets:

| Secret | Description | Example |
|--------|-------------|---------|
| `DATABASE_URL` | Supabase connection string | `postgresql://postgres:xxx@xxx.supabase.co:6543/postgres` |
| `REDIS_URL` | Upstash connection string | `rediss://default:xxx@xxx.upstash.io:6379` |
| `JWT_SECRET` | Random 64-char string | `openssl rand -hex 32` |
| `COOLIFY_BACKEND_WEBHOOK_URL` | From Coolify (Step 7) | `https://coolify.example.com/webhooks/...` |
| `COOLIFY_ML_SERVICE_WEBHOOK_URL` | From Coolify (Step 7) | `https://coolify.example.com/webhooks/...` |
| `COOLIFY_WEBHOOK_SECRET` | From Coolify settings | Your Coolify API token |
| `DISCORD_WEBHOOK_URL` | Optional, for notifications | `https://discord.com/api/webhooks/...` |

### 6.2 Generate JWT Secret

```bash
openssl rand -hex 32
# Copy output for JWT_SECRET
```

## Step 7: Deploy Services in Coolify

### 7.1 Create Backend Service

1. Go to **Projects** > **Add New Project** or use default
2. Click **Add New Resource** > **Docker Compose**
3. Select your GitHub source
4. Choose repository: `onurtemizkan/nutri`
5. Set:
   - **Build Path**: `server`
   - **Dockerfile**: `Dockerfile`
   - **Name**: `nutri-backend`

### 7.2 Configure Backend Environment Variables

In the service settings, add:

```env
NODE_ENV=production
PORT=3000
DATABASE_URL=<your-supabase-url>
REDIS_URL=<your-upstash-url>
JWT_SECRET=<your-jwt-secret>
JWT_EXPIRES_IN=7d
ML_SERVICE_URL=http://nutri-ml-service:8080
LOG_LEVEL=info
CORS_ORIGIN=https://your-app-domain.com
```

### 7.3 Configure Backend Health Check

- **Path**: `/health/live`
- **Port**: `3000`
- **Interval**: `30s`

### 7.4 Create ML Service

1. Add another resource
2. Select repository
3. Set:
   - **Build Path**: `ml-service`
   - **Dockerfile**: `Dockerfile`
   - **Name**: `nutri-ml-service`

### 7.5 Configure ML Service Environment Variables

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=<your-supabase-url>
REDIS_URL=<your-upstash-url>
```

### 7.6 Get Webhook URLs

For each service:
1. Go to service settings
2. Find **Webhooks** section
3. Enable **GitHub Webhook**
4. Copy the webhook URL
5. Add to GitHub Secrets (Step 6)

### 7.7 Configure Domains

For each service:
1. Go to **Domains** tab
2. Add domain:
   - Backend: `api.yourdomain.com`
   - ML Service: `ml.yourdomain.com` (or internal only)
3. Enable **SSL** via Let's Encrypt

## Step 8: DNS Configuration

### 8.1 Add DNS Records

In your domain registrar/DNS provider, add:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | api | `<server-ip>` | 3600 |
| A | coolify | `<server-ip>` | 3600 |
| A | ml | `<server-ip>` | 3600 (optional) |

### 8.2 Verify DNS

```bash
# Check DNS propagation
dig api.yourdomain.com +short
# Should return server IP
```

### 8.3 Verify SSL

```bash
# Check SSL certificate
curl -vI https://api.yourdomain.com/health/live 2>&1 | grep -A 5 "Server certificate"
```

## Step 9: Run Initial Migration

### 9.1 Via GitHub Actions

1. Go to **Actions** > **Database Migration**
2. Click **Run workflow**
3. Select:
   - Branch: `master`
   - Environment: `production`
   - Dry run: `true` (first time to verify)
4. Review output
5. Run again with Dry run: `false`

### 9.2 Manually (if needed)

```bash
# Clone repo locally
git clone https://github.com/onurtemizkan/nutri.git
cd nutri/server

# Set DATABASE_URL
export DATABASE_URL="<your-supabase-url>"

# Run migrations
npm ci
npx prisma generate
npx prisma migrate deploy
```

## Step 10: Verify Deployment

### 10.1 Health Checks

```bash
# Backend health
curl https://api.yourdomain.com/health
# Should return JSON with status: "healthy"

# Backend liveness
curl https://api.yourdomain.com/health/live
# Should return {"status": "ok"}

# ML Service health
curl https://ml.yourdomain.com/health
# Should return health status
```

### 10.2 Check Logs

In Coolify:
1. Go to service
2. Click **Logs** tab
3. Verify no errors

### 10.3 Test API

```bash
# Test API endpoint
curl https://api.yourdomain.com/
# Should return service info
```

## Post-Setup Checklist

- [ ] Backend health check passes
- [ ] ML Service health check passes
- [ ] SSL certificates valid
- [ ] Database connection works
- [ ] Redis connection works
- [ ] GitHub webhook triggers deployment
- [ ] Discord notifications working (if configured)
- [ ] Backups enabled on Hetzner
- [ ] SSH key backup saved securely
- [ ] All secrets documented in password manager

## Next Steps

1. Configure [UptimeRobot](https://uptimerobot.com/) for monitoring
2. Test deployment by pushing a commit
3. Review [RUNBOOK.md](./RUNBOOK.md) for daily operations
4. Familiarize with [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

*Estimated setup time: 30-60 minutes*
