# Fix Coolify Webhook URLs

## Problem
GitHub Actions deploy workflow is failing with:
```
HTTP Status: 404
Response: {"message":"No resources found."}
```

This means the webhook URLs in GitHub Secrets point to Coolify resources that no longer exist (likely the resources were recreated with new UUIDs).

## Solution

### Step 1: Get New Webhook URLs from Coolify

1. Open Coolify dashboard: http://195.201.228.58:8000
2. Log in with your admin credentials

**For Backend Service:**
1. Navigate to: Projects → (your project) → nutri-backend
2. Go to the **Settings** or **Webhooks** tab
3. Find the **Webhook URL** (or Deploy Webhook)
4. Copy the full URL

**For ML Service:**
1. Navigate to: Projects → (your project) → nutri-ml-service
2. Go to the **Settings** or **Webhooks** tab
3. Find the **Webhook URL**
4. Copy the full URL

### Step 2: Update GitHub Secrets

1. Go to: https://github.com/onurtemizkan/nutri/settings/secrets/actions
2. Update these secrets:
   - **COOLIFY_BACKEND_WEBHOOK_URL**: Paste the backend webhook URL
   - **COOLIFY_ML_WEBHOOK_URL**: Paste the ML service webhook URL

### Step 3: Trigger New Deployment

Either:
- Push a new commit to master branch, OR
- Go to Actions → Deploy → Run workflow (manual trigger)

## Alternative: Manual Deployment

If webhooks don't work, SSH to the server:

```bash
ssh root@195.201.228.58

# Pull latest images
docker pull ghcr.io/onurtemizkan/nutri/backend:latest
docker pull ghcr.io/onurtemizkan/nutri/ml-service:latest

# Find Coolify application directories
ls /data/coolify/applications/

# For each service, run:
cd /data/coolify/applications/<backend-app-id>
docker compose pull && docker compose up -d

cd /data/coolify/applications/<ml-service-app-id>
docker compose pull && docker compose up -d
```

## Verify Deployment

```bash
# Backend health
curl http://ko4ss4wkc8k8cswgs0k4gks0.195.201.228.58.sslip.io/health

# ML service health
curl http://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io/health/live
```

## Using Coolify API (Alternative)

If you have your Coolify API token, you can deploy directly:

```bash
# List applications to find UUIDs
curl -H "Authorization: Bearer YOUR_COOLIFY_TOKEN" \
  http://195.201.228.58:8000/api/v1/applications

# Deploy by UUID
curl -H "Authorization: Bearer YOUR_COOLIFY_TOKEN" \
  "http://195.201.228.58:8000/api/v1/deploy?uuid=BACKEND_UUID,ML_SERVICE_UUID"
```

---

*Created: 2024-12-24*
*Issue: GitHub Actions deploy failing with "No resources found"*
