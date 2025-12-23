# Task ID: 24

**Title:** Create GitHub Actions Deploy Workflow

**Status:** done

**Dependencies:** 23 ✓

**Priority:** high

**Description:** Create a deployment workflow that triggers Coolify webhook after successful build, verifies deployment health, and sends notifications.

**Details:**

**Create `.github/workflows/deploy.yml`:**
```yaml
name: Deploy

on:
  workflow_run:
    workflows: ["Build and Push"]
    types: [completed]
    branches: [master]
  workflow_dispatch:  # Manual trigger for rollback
    inputs:
      image_tag:
        description: 'Image tag to deploy (default: latest)'
        required: false
        default: 'latest'

jobs:
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' || github.event_name == 'workflow_dispatch' }}

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Trigger Coolify Backend Deployment
        id: deploy-backend
        run: |
          response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Authorization: Bearer ${{ secrets.COOLIFY_WEBHOOK_SECRET }}" \
            "${{ secrets.COOLIFY_BACKEND_WEBHOOK_URL }}")
          http_code=$(echo "$response" | tail -n1)
          body=$(echo "$response" | sed '$d')
          echo "HTTP Status: $http_code"
          echo "Response: $body"
          if [ "$http_code" != "200" ] && [ "$http_code" != "201" ]; then
            echo "Backend deployment trigger failed!"
            exit 1
          fi

      - name: Trigger Coolify ML Service Deployment
        id: deploy-ml
        run: |
          response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Authorization: Bearer ${{ secrets.COOLIFY_WEBHOOK_SECRET }}" \
            "${{ secrets.COOLIFY_ML_WEBHOOK_URL }}")
          http_code=$(echo "$response" | tail -n1)
          body=$(echo "$response" | sed '$d')
          echo "HTTP Status: $http_code"
          echo "Response: $body"
          if [ "$http_code" != "200" ] && [ "$http_code" != "201" ]; then
            echo "ML service deployment trigger failed!"
            exit 1
          fi

      - name: Wait for deployment
        run: sleep 60  # Wait for containers to start

      - name: Verify Backend Health
        id: health-backend
        run: |
          for i in {1..10}; do
            response=$(curl -s -o /dev/null -w "%{http_code}" \
              --connect-timeout 5 \
              --max-time 10 \
              "${{ secrets.PRODUCTION_API_URL }}/health" || echo "000")
            echo "Attempt $i: HTTP $response"
            if [ "$response" = "200" ]; then
              echo "Backend is healthy!"
              exit 0
            fi
            sleep 10
          done
          echo "Backend health check failed!"
          exit 1

      - name: Send Success Notification
        if: success()
        run: |
          curl -X POST "${{ secrets.DISCORD_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '{
              "embeds": [{
                "title": "✅ Deployment Successful",
                "description": "Nutri has been deployed to production",
                "color": 5763719,
                "fields": [
                  {"name": "Commit", "value": "'"${{ github.sha }}"'", "inline": true},
                  {"name": "Branch", "value": "master", "inline": true}
                ],
                "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
              }]
            }' || true

      - name: Send Failure Notification
        if: failure()
        run: |
          curl -X POST "${{ secrets.DISCORD_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '{
              "embeds": [{
                "title": "❌ Deployment Failed",
                "description": "Nutri deployment to production failed",
                "color": 15548997,
                "fields": [
                  {"name": "Commit", "value": "'"${{ github.sha }}"'", "inline": true},
                  {"name": "Action", "value": "[View Logs]('"${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"')", "inline": true}
                ],
                "timestamp": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
              }]
            }' || true
```

**Required GitHub Secrets:**
- COOLIFY_WEBHOOK_SECRET
- COOLIFY_BACKEND_WEBHOOK_URL
- COOLIFY_ML_WEBHOOK_URL
- PRODUCTION_API_URL
- DISCORD_WEBHOOK_URL (optional)

**Test Strategy:**

1. Set up test Coolify webhook URLs
2. Add required secrets to GitHub repository
3. Trigger workflow manually with workflow_dispatch
4. Verify Coolify receives webhook calls
5. Test health check retry logic
6. Verify Discord notifications are sent
7. Test failure notification by using invalid webhook URL
8. Test manual rollback trigger with specific image tag

## Subtasks

### 24.1. Create deploy.yml with workflow triggers and inputs

**Status:** pending  
**Dependencies:** None  

Create `.github/workflows/deploy.yml` with workflow_run trigger listening to 'Build and Push' workflow completion, and workflow_dispatch for manual deployments with image_tag input parameter.

**Details:**

Set up the workflow file with two trigger types: 1) workflow_run that triggers on successful completion of 'Build and Push' workflow on master branch, 2) workflow_dispatch with optional image_tag input (default: 'latest'). Configure the deploy job to run on ubuntu-latest and only execute if the workflow_run was successful OR if manually triggered. This establishes the foundation for the deployment workflow.

### 24.2. Implement Coolify webhook triggers with authentication

**Status:** pending  
**Dependencies:** 24.1  

Add steps to trigger Coolify webhooks for both backend and ML service deployments with proper authentication headers and response validation.

**Details:**

Create two separate steps using curl to POST to Coolify webhook URLs with Authorization Bearer token from COOLIFY_WEBHOOK_SECRET. Capture both HTTP status code and response body using `-s -w "\n%{http_code}"`. Validate that responses are 200 or 201, otherwise fail the workflow. Add echo statements for debugging. Structure: COOLIFY_BACKEND_WEBHOOK_URL first, then COOLIFY_ML_WEBHOOK_URL. Include proper error messages if triggers fail.

### 24.3. Add deployment wait period and health check with retry logic

**Status:** pending  
**Dependencies:** 24.2  

Implement a 60-second wait period followed by backend health check verification with 10 retry attempts and exponential backoff.

**Details:**

Add sleep 60 step to allow containers to start. Create health check step that polls PRODUCTION_API_URL/health endpoint up to 10 times with 10-second intervals. Use curl with --connect-timeout 5 and --max-time 10 flags. Handle connection failures by defaulting to '000' status code. Exit with success (0) on first 200 response, or fail (1) after all retries exhausted. Include iteration counter in echo output for debugging.

### 24.4. Implement Discord notification integration for deployment status

**Status:** pending  
**Dependencies:** 24.3  

Add success and failure notification steps that send rich embed messages to Discord webhook with deployment details and links.

**Details:**

Create two conditional notification steps: 1) Success notification (if: success()) with green embed (color: 5763719) showing ✅ title, commit SHA, branch, and timestamp, 2) Failure notification (if: failure()) with red embed (color: 15548997) showing ❌ title, commit SHA, and link to workflow run logs. Use curl to POST JSON payloads to DISCORD_WEBHOOK_URL. Include `|| true` to prevent notification failures from affecting workflow status. Format timestamp using `date -u +%Y-%m-%dT%H:%M:%SZ`.

### 24.5. Configure required GitHub repository secrets

**Status:** pending  
**Dependencies:** None  

Document and configure all required GitHub secrets: COOLIFY_WEBHOOK_SECRET, COOLIFY_BACKEND_WEBHOOK_URL, COOLIFY_ML_WEBHOOK_URL, PRODUCTION_API_URL, and DISCORD_WEBHOOK_URL.

**Details:**

Create a secrets checklist and add placeholder values to GitHub repository settings. Required secrets: 1) COOLIFY_WEBHOOK_SECRET - Bearer token for Coolify authentication, 2) COOLIFY_BACKEND_WEBHOOK_URL - Full webhook URL for backend deployment, 3) COOLIFY_ML_WEBHOOK_URL - Full webhook URL for ML service, 4) PRODUCTION_API_URL - Base URL for health checks (e.g., https://api.nutri.com), 5) DISCORD_WEBHOOK_URL - Discord webhook endpoint (optional but recommended). Document these in deployment docs and ensure they're referenced correctly in workflow file.

### 24.6. End-to-end workflow testing with actual Coolify deployment

**Status:** pending  
**Dependencies:** 24.1, 24.2, 24.3, 24.4, 24.5  

Test the complete deployment workflow with real Coolify webhooks, verify health checks, monitor deployment process, and validate notifications.

**Details:**

Execute full workflow test: 1) Merge a test PR to trigger 'Build and Push' workflow, 2) Verify deploy workflow triggers automatically on completion, 3) Monitor Coolify for webhook reception and deployment start, 4) Watch health check retries in action logs, 5) Verify Discord success notification, 6) Test manual rollback using workflow_dispatch with previous image tag, 7) Test failure scenario by temporarily breaking health endpoint or using invalid webhook URL, 8) Confirm failure notification is sent. Document any issues and iterations needed.
