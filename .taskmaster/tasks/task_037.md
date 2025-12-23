# Task ID: 37

**Title:** Create Rollback Script and Procedures

**Status:** pending

**Dependencies:** 24 ✓

**Priority:** medium

**Description:** Create a rollback script that can quickly revert to a previous deployment version with both manual and automated options.

**Details:**

**Create `scripts/deploy/rollback.sh`:**
```bash
#!/bin/bash
set -e

# =============================================================================
# Nutri Rollback Script
# Rolls back to a previous deployment version
# =============================================================================

echo "=== Nutri Rollback ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Configuration
REGISTRY="ghcr.io"
REPO="${GITHUB_REPOSITORY:-your-org/nutri}"
BACKEND_IMAGE="$REGISTRY/$REPO/backend"
ML_IMAGE="$REGISTRY/$REPO/ml-service"

# Parse arguments
TARGET_TAG="$1"
SERVICE="${2:-all}"  # backend, ml, or all

if [ -z "$TARGET_TAG" ]; then
    echo "Usage: $0 <image-tag> [service]"
    echo ""
    echo "Arguments:"
    echo "  image-tag   Git SHA or tag to rollback to"
    echo "  service     'backend', 'ml', or 'all' (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0 abc1234              # Rollback all services to abc1234"
    echo "  $0 abc1234 backend      # Rollback only backend"
    echo ""
    echo "Recent tags:"
    echo "--- Backend ---"
    curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
        "https://api.github.com/orgs/$(dirname $REPO)/packages/container/$(basename $REPO)%2Fbackend/versions" \
        2>/dev/null | jq -r '.[0:5] | .[].metadata.container.tags[]' 2>/dev/null || echo "Unable to fetch tags"
    exit 1
fi

# Confirm rollback
echo "Rolling back to: $TARGET_TAG"
echo "Services: $SERVICE"
echo ""
read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Function to trigger Coolify deployment
trigger_coolify_deploy() {
    local webhook_url="$1"
    local service_name="$2"
    
    echo "Triggering $service_name deployment..."
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Authorization: Bearer $COOLIFY_WEBHOOK_SECRET" \
        "$webhook_url")
    
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo "✓ $service_name deployment triggered"
    else
        echo "✗ $service_name deployment failed (HTTP $http_code)"
        return 1
    fi
}

# Function to wait for health check
wait_for_health() {
    local url="$1"
    local max_attempts="${2:-30}"
    
    echo "Waiting for $url to be healthy..."
    for i in $(seq 1 $max_attempts); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "✓ Service healthy after $i attempts"
            return 0
        fi
        sleep 5
    done
    echo "✗ Service not healthy after $max_attempts attempts"
    return 1
}

# Perform rollback
if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "backend" ]; then
    echo ""
    echo "=== Rolling back Backend ==="
    
    # Pull the target image
    docker pull "$BACKEND_IMAGE:$TARGET_TAG" || {
        echo "Error: Could not pull $BACKEND_IMAGE:$TARGET_TAG"
        exit 1
    }
    
    # Tag as latest for Coolify
    docker tag "$BACKEND_IMAGE:$TARGET_TAG" "$BACKEND_IMAGE:latest"
    
    # Trigger Coolify deployment
    if [ -n "$COOLIFY_BACKEND_WEBHOOK_URL" ]; then
        trigger_coolify_deploy "$COOLIFY_BACKEND_WEBHOOK_URL" "Backend"
    else
        echo "Warning: COOLIFY_BACKEND_WEBHOOK_URL not set"
        echo "Manually restart the container in Coolify"
    fi
    
    # Wait for health
    sleep 10
    wait_for_health "${PRODUCTION_API_URL:-http://localhost:3000}/health"
fi

if [ "$SERVICE" = "all" ] || [ "$SERVICE" = "ml" ]; then
    echo ""
    echo "=== Rolling back ML Service ==="
    
    docker pull "$ML_IMAGE:$TARGET_TAG" || {
        echo "Error: Could not pull $ML_IMAGE:$TARGET_TAG"
        exit 1
    }
    
    docker tag "$ML_IMAGE:$TARGET_TAG" "$ML_IMAGE:latest"
    
    if [ -n "$COOLIFY_ML_WEBHOOK_URL" ]; then
        trigger_coolify_deploy "$COOLIFY_ML_WEBHOOK_URL" "ML Service"
    else
        echo "Warning: COOLIFY_ML_WEBHOOK_URL not set"
    fi
fi

echo ""
echo "=== Rollback Complete ==="
echo "Rolled back to: $TARGET_TAG"
echo ""
echo "Verify deployment:"
echo "  curl ${PRODUCTION_API_URL:-http://localhost:3000}/health"
```

**Add to deploy workflow (`.github/workflows/deploy.yml`):**
Already included workflow_dispatch with image_tag input for manual rollback.

**Document in `docs/deployment/ROLLBACK.md`:**
```markdown
# Rollback Procedures

## Quick Rollback via GitHub Actions

1. Go to Actions > Deploy workflow
2. Click "Run workflow"
3. Enter the image tag/SHA to rollback to
4. Click "Run workflow"

## Manual Rollback via Script

```bash
# Set required environment variables
export COOLIFY_WEBHOOK_SECRET="..."
export COOLIFY_BACKEND_WEBHOOK_URL="..."
export PRODUCTION_API_URL="https://api.nutri.app"

# Rollback all services to specific commit
./scripts/deploy/rollback.sh abc1234

# Rollback only backend
./scripts/deploy/rollback.sh abc1234 backend
```

## Finding Previous Tags

```bash
# Via GitHub API
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/orgs/your-org/packages/container/nutri%2Fbackend/versions" \
  | jq '.[].metadata.container.tags'

# Via Docker
docker pull ghcr.io/your-org/nutri/backend:latest
docker history ghcr.io/your-org/nutri/backend:latest
```

## Rollback Time Target

- Time to initiate: < 2 minutes
- Deployment time: < 5 minutes
- **Total rollback time: < 7 minutes**
```

**Test Strategy:**

1. Deploy a new version to staging
2. Run rollback script with previous tag
3. Verify correct image is running: `docker inspect <container> | grep Image`
4. Test health endpoint responds correctly
5. Test rollback with invalid tag (should fail gracefully)
6. Test GitHub Actions manual rollback trigger
7. Time the entire rollback process (target <5 min)
8. Test rollback of individual services
