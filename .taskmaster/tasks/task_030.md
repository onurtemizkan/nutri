# Task ID: 30

**Title:** Set Up Uptime Monitoring

**Status:** pending

**Dependencies:** 21

**Priority:** medium

**Description:** Configure external uptime monitoring for production endpoints using a free monitoring service with alerting capabilities.

**Details:**

**Option 1: UptimeRobot (Recommended - Free tier)**

1. Create account at https://uptimerobot.com
2. Add monitors:

**Monitor 1 - Backend Health:**
- Monitor Type: HTTP(s)
- Friendly Name: Nutri Backend
- URL: https://api.nutri.app/health
- Monitoring Interval: 5 minutes
- Alert contacts: Your email/Discord

**Monitor 2 - ML Service (via Backend):**
- Monitor Type: HTTP(s)
- Friendly Name: Nutri ML Health
- URL: https://api.nutri.app/api/food/health
- Monitoring Interval: 5 minutes

**Configure Alerts:**
1. Alert Contacts > Add Contact
2. Add Discord webhook:
   - Type: Webhook
   - URL: Your Discord webhook URL
   - POST value:
   ```json
   {"content": "*alertTypeFriendlyName* - *monitorFriendlyName* is *alertDetails*"}
   ```

**Create Status Page (Optional):**
1. UptimeRobot > My Settings > Status Pages
2. Create page with both monitors
3. Custom domain: status.nutri.app

**Option 2: Better Stack (Alternative)**

```bash
# Create heartbeat endpoint
curl -X POST "https://uptime.betterstack.com/api/v2/heartbeats" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nutri Backend",
    "period": 300,
    "grace": 60,
    "call": true,
    "email": true
  }'
```

**Create `scripts/deploy/setup-monitoring.sh`:**
```bash
#!/bin/bash
# Documentation script for monitoring setup

echo "=== Nutri Monitoring Setup ==="
echo ""
echo "1. Go to https://uptimerobot.com and create account"
echo "2. Add the following monitors:"
echo ""
echo "   Backend Health Check:"
echo "   - URL: https://api.nutri.app/health"
echo "   - Interval: 5 minutes"
echo "   - Expected status: 200"
echo ""
echo "3. Configure Discord notifications:"
echo "   - Create webhook in Discord server"
echo "   - Add webhook URL to UptimeRobot alert contacts"
echo ""
echo "4. (Optional) Create public status page"
echo ""
```

**Add monitoring endpoint docs to `docs/deployment/MONITORING.md`:**
```markdown
# Monitoring Setup

## Endpoints to Monitor

| Endpoint | Expected Status | Interval |
|----------|-----------------|----------|
| /health | 200 | 5 min |
| /health/live | 200 | 1 min |

## Alert Escalation

1. Discord notification (immediate)
2. Email notification (after 5 min down)
3. SMS (optional, after 15 min down)

## Response Procedures

See [Runbook](./RUNBOOK.md) for response procedures.
```

**Test Strategy:**

1. Configure monitors in UptimeRobot
2. Intentionally break health endpoint, verify alert fires
3. Verify alert reaches Discord within 5 minutes
4. Test alert recovery notification
5. Check response time graphs show expected latency
6. Verify status page (if created) shows correct status

## Subtasks

### 30.1. Configure UptimeRobot monitors for Backend and ML Service health endpoints

**Status:** pending  
**Dependencies:** None  

Set up UptimeRobot account and create HTTP monitors for both the Backend API (/health and /health/live) and ML Service health endpoints with appropriate check intervals.

**Details:**

1. Create free account at https://uptimerobot.com
2. Add Backend API monitors:
   - Monitor 1: 'Nutri Backend Health' pointing to https://api.nutri.app/health (5-min interval) - this comprehensive endpoint checks database and ML service connectivity
   - Monitor 2: 'Nutri Backend Live' pointing to https://api.nutri.app/health/live (1-min interval) - lightweight liveness probe
3. Add ML Service monitors:
   - Monitor 3: 'Nutri ML Health' pointing to https://ml.nutri.app/health (5-min interval) - checks database, Redis, and overall ML service health
   - Monitor 4: 'Nutri ML Queue Status' pointing to https://ml.nutri.app/queue/status (5-min interval) - monitors inference queue health and circuit breaker state
4. Configure expected HTTP status codes: 200 for healthy responses, alert on 503 (unhealthy) or timeouts
5. Set timeout threshold to 30 seconds (health endpoints should respond within 5 seconds normally)
6. Enable SSL certificate monitoring for HTTPS endpoints

### 30.2. Set up Discord webhook alerts for monitoring notifications

**Status:** pending  
**Dependencies:** 30.1  

Configure Discord webhook integration to receive real-time notifications when monitors detect downtime or recovery, following the notification pattern in the existing RUNBOOK.md.

**Details:**

1. Create Discord webhook in the designated alerts channel:
   - Server Settings > Integrations > Webhooks > New Webhook
   - Name it 'Nutri Uptime Monitor'
   - Copy webhook URL
2. Add webhook to UptimeRobot:
   - Go to My Settings > Alert Contacts > Add Alert Contact
   - Type: Webhook (as per task details JSON format)
   - Webhook URL: [Discord webhook URL]
   - POST value (JSON): {"content": "**Alert**: *alertTypeFriendlyName* - *monitorFriendlyName* is *alertDetails*. Response time: *alertDuration*ms"}
   - Enable for Down, Up, and SSL expiry alerts
3. Add email backup alert:
   - Add Alert Contact > Email
   - Configure to receive alerts after 5 minutes of downtime (avoid alert fatigue)
4. Link alert contacts to all monitors created in subtask 1
5. Configure alert notification delays: Immediate for Discord, 5-min delay for email to reduce noise

### 30.3. Create public status page for service availability transparency

**Status:** pending  
**Dependencies:** 30.1  

Configure UptimeRobot status page to provide public visibility into service availability, response times, and historical uptime data.

**Details:**

1. Create status page in UptimeRobot:
   - Go to My Settings > Status Pages > Add Status Page
   - Name: 'Nutri Service Status'
   - Select monitors to include: Backend Health, Backend Live, ML Health
   - Enable 'Show response time' and 'Show uptime ratio'
2. Customize status page appearance:
   - Add Nutri branding/logo if available
   - Set appropriate colors (green/yellow/red states)
   - Configure incident history display (last 30 days)
3. (Optional) Configure custom domain:
   - Add CNAME record: status.nutri.app -> stats.uptimerobot.com
   - Enable SSL in UptimeRobot dashboard
4. Add status page link to application footer/help section
5. Enable RSS feed for automated status updates
6. Configure announcement feature for planned maintenance windows

### 30.4. Document monitoring setup and response procedures in deployment docs

**Status:** pending  
**Dependencies:** 30.1, 30.2, 30.3  

Create comprehensive MONITORING.md documentation covering endpoint monitoring configuration, alert escalation procedures, and response runbook integration.

**Details:**

1. Create docs/deployment/MONITORING.md with sections:
   - Overview of monitoring architecture
   - Endpoints monitored (table with URL, expected status, interval as shown in task details)
   - UptimeRobot configuration reference (how to add/modify monitors)
   - Alert escalation matrix (Discord immediate, email 5-min, SMS optional 15-min)
   - Integration with existing RUNBOOK.md emergency procedures
2. Create scripts/deploy/setup-monitoring.sh documentation script:
   - Echo step-by-step UptimeRobot setup instructions
   - Include webhook configuration template
   - Reference production URLs for all health endpoints
3. Update docs/deployment/RUNBOOK.md:
   - Add reference to MONITORING.md in incident detection section
   - Link to UptimeRobot dashboard in emergency procedures
   - Document how to acknowledge alerts and update status page during incidents
4. Add monitoring section to scripts/deploy/setup-server.sh output (already mentions UptimeRobot at line 463)
5. Include troubleshooting section for common monitoring issues (false positives, SSL cert warnings)
