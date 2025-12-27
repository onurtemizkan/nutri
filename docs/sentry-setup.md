# Sentry Error Tracking Setup Guide

This guide explains how to set up Sentry error tracking for the Nutri application.

## Overview

Sentry is configured for both services:
- **Backend API** (Node.js/Express) - `@sentry/node` v10+
- **ML Service** (Python/FastAPI) - `sentry-sdk[fastapi]` v2+

## Prerequisites

1. Create a Sentry account at https://sentry.io
2. Create an organization (or use an existing one)
3. Create two projects:
   - `nutri-backend` - Platform: Node.js
   - `nutri-ml-service` - Platform: Python/FastAPI

## Configuration

### 1. Get Your DSN

For each project:
1. Go to **Settings > Projects > [Project] > Client Keys (DSN)**
2. Copy the DSN (format: `https://[key]@[org].ingest.sentry.io/[project-id]`)

### 2. Environment Variables

#### Backend API (`server/.env`)
```env
# Required for Sentry
SENTRY_DSN=https://your-key@org.ingest.sentry.io/project-id
```

#### ML Service (`ml-service/.env`)
```env
# Required for Sentry
SENTRY_DSN=https://your-key@org.ingest.sentry.io/project-id
```

### 3. GitHub Actions (Source Maps)

For source map upload in CI/CD, add these as repository secrets/variables:

**Secrets:**
- `SENTRY_AUTH_TOKEN` - Generate at https://sentry.io/settings/account/api/auth-tokens/
  - Required scopes: `project:releases`, `project:write`, `org:read`

**Variables:**
- `SENTRY_ORG` - Your organization slug (from Settings > General Settings)
- `SENTRY_BACKEND_PROJECT` - Backend project slug (default: `nutri-backend`)

## Features Enabled

### Error Tracking
- Automatic capture of unhandled exceptions
- Manual capture with `captureException()`
- Request context (URL, method, headers)
- User context (when authenticated)

### Performance Monitoring
- Automatic transaction tracing
- Express/FastAPI route instrumentation
- Database query tracing (SQLAlchemy, Prisma)
- HTTP request tracing

### Security
Sensitive data is automatically scrubbed:
- Authorization headers removed
- Cookies removed
- Passwords redacted `[REDACTED]`
- API keys redacted
- Emails partially masked (`j***@example.com`)

## Verification

### Test Endpoints (Non-Production Only)

**Backend:**
```bash
# Manual test error capture
curl http://localhost:3000/api/debug/sentry-test

# Trigger unhandled error
curl http://localhost:3000/api/debug/sentry-throw

# Check Sentry status
curl http://localhost:3000/api/debug/sentry-status
```

**ML Service:**
```bash
# Manual test error capture
curl http://localhost:8000/debug/sentry-test

# Trigger unhandled error
curl http://localhost:8000/debug/sentry-throw

# Check Sentry status
curl http://localhost:8000/debug/sentry-status
```

### Verification Checklist

- [ ] `SENTRY_DSN` configured for both services
- [ ] Test error appears in Sentry dashboard
- [ ] Stack trace shows original source code (source maps working)
- [ ] Sensitive data is scrubbed (no auth tokens visible)
- [ ] User context appears on errors (when authenticated)
- [ ] Performance traces visible in Sentry Performance tab

## Alert Configuration

Configure alerts in Sentry dashboard:

### Recommended Alerts

1. **New Issue Alert**
   - Trigger: First occurrence of any new error
   - Action: Notify via Slack/Email

2. **Error Rate Alert**
   - Trigger: >10 errors in 5 minutes
   - Action: Notify via Slack/Email

3. **Performance Alert**
   - Trigger: P95 response time >2s
   - Action: Notify via Slack

### Setting Up Alerts

1. Go to **Alerts > Create Alert Rule**
2. Select alert type (Issue, Transaction)
3. Configure conditions
4. Choose notification channel

## Troubleshooting

### Errors Not Appearing

1. Verify `SENTRY_DSN` is set correctly
2. Check for network connectivity to Sentry
3. Verify error isn't in `ignoreErrors` list
4. Check Sentry dashboard filters

### Source Maps Not Working

1. Verify `SENTRY_AUTH_TOKEN` has correct permissions
2. Check GitHub Actions workflow ran successfully
3. Verify release version matches (`GITHUB_SHA`)
4. Check source map artifacts in Sentry release

### Performance Traces Missing

1. Verify `tracesSampleRate` is not 0
2. Check transaction isn't filtered out
3. Verify OpenTelemetry integration (ML service)

## Best Practices

1. **Use meaningful release versions** - Use git SHA for precise debugging
2. **Set appropriate sample rates** - 10% for production, 100% for development
3. **Add user context** - Call `setUser()` after authentication
4. **Configure alerts** - Set up alerts for new issues and error spikes
5. **Review weekly** - Check Sentry dashboard for new issues regularly

## Resources

- [Sentry Node.js Documentation](https://docs.sentry.io/platforms/javascript/guides/express/)
- [Sentry Python Documentation](https://docs.sentry.io/platforms/python/guides/fastapi/)
- [Sentry CLI Documentation](https://docs.sentry.io/product/cli/)
- [Source Maps Documentation](https://docs.sentry.io/platforms/javascript/sourcemaps/)
