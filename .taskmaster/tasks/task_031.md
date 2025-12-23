# Task ID: 31

**Title:** Integrate Error Tracking with Sentry

**Status:** pending

**Dependencies:** 26 ✓, 27 ✓

**Priority:** medium

**Description:** Set up Sentry error tracking for both backend and ML service to capture, group, and alert on production errors with source maps.

**Details:**

**Backend Integration (Express.js):**

1. Install Sentry:
```bash
cd server && npm install @sentry/node @sentry/profiling-node
```

2. Create `server/src/config/sentry.ts`:
```typescript
import * as Sentry from '@sentry/node';
import { ProfilingIntegration } from '@sentry/profiling-node';
import { Express } from 'express';

export function initSentry(app: Express): void {
  if (!process.env.SENTRY_DSN) {
    console.log('Sentry DSN not configured, skipping initialization');
    return;
  }

  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    release: process.env.npm_package_version,
    integrations: [
      new Sentry.Integrations.Http({ tracing: true }),
      new Sentry.Integrations.Express({ app }),
      new ProfilingIntegration(),
    ],
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: 0.1,
    beforeSend(event) {
      // Scrub sensitive data
      if (event.request?.headers) {
        delete event.request.headers.authorization;
        delete event.request.headers.cookie;
      }
      return event;
    },
  });
}

export { Sentry };
```

3. Update `server/src/index.ts`:
```typescript
import { initSentry, Sentry } from './config/sentry';

const app = express();

// Initialize Sentry FIRST
initSentry(app);

// Sentry request handler (must be first middleware)
if (process.env.SENTRY_DSN) {
  app.use(Sentry.Handlers.requestHandler());
  app.use(Sentry.Handlers.tracingHandler());
}

// ... other middleware and routes ...

// Sentry error handler (before your error handler)
if (process.env.SENTRY_DSN) {
  app.use(Sentry.Handlers.errorHandler());
}

app.use(errorHandler);
```

**ML Service Integration (Python):**

1. Add to `ml-service/requirements.txt`:
```
sentry-sdk[fastapi]>=1.40.0
```

2. Create `ml-service/app/core/sentry.py`:
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from app.config import settings

def init_sentry():
    if not settings.SENTRY_DSN:
        return
    
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        release=settings.VERSION,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1 if settings.ENVIRONMENT == 'production' else 1.0,
        profiles_sample_rate=0.1,
        send_default_pii=False,  # Don't send personally identifiable info
        before_send=scrub_sensitive_data,
    )

def scrub_sensitive_data(event, hint):
    if 'request' in event and 'headers' in event['request']:
        headers = event['request']['headers']
        if 'authorization' in headers:
            headers['authorization'] = '[REDACTED]'
    return event
```

3. Update `ml-service/app/main.py`:
```python
from app.core.sentry import init_sentry

# Initialize Sentry on startup
init_sentry()
```

**GitHub Actions - Upload Source Maps:**
Add to `.github/workflows/build.yml`:
```yaml
      - name: Upload source maps to Sentry
        if: env.SENTRY_AUTH_TOKEN != ''
        env:
          SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}
        run: |
          npm install -g @sentry/cli
          cd server && npm run build
          sentry-cli releases new ${{ github.sha }}
          sentry-cli releases files ${{ github.sha }} upload-sourcemaps ./dist
          sentry-cli releases finalize ${{ github.sha }}
```

**Environment Variables to Add:**
```
SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
SENTRY_AUTH_TOKEN=xxx (for source map upload)
```

**Test Strategy:**

1. Create Sentry project and get DSN
2. Trigger test error: `throw new Error('Test Sentry integration')`
3. Verify error appears in Sentry dashboard
4. Check source maps resolve correctly
5. Verify sensitive data is scrubbed (no auth tokens)
6. Test alert notification for new errors
7. Verify performance monitoring shows traces
8. Test ML service integration separately

## Subtasks

### 31.1. Integrate Sentry SDK with Express.js Backend API

**Status:** pending  
**Dependencies:** None  

Install and configure @sentry/node and @sentry/profiling-node for the Express.js backend to capture errors, performance traces, and profiling data with proper initialization order.

**Details:**

1. Install Sentry packages: `cd server && npm install @sentry/node @sentry/profiling-node`

2. Create `server/src/config/sentry.ts` with:
   - Sentry.init() with DSN from env, environment detection, release version from package.json
   - HTTP integration for request tracing
   - Express integration for route instrumentation
   - ProfilingIntegration for performance profiling
   - tracesSampleRate: 0.1 production, 1.0 development
   - profilesSampleRate: 0.1
   - beforeSend hook for data scrubbing (implemented in subtask 3)

3. Update `server/src/index.ts`:
   - Import and call initSentry(app) BEFORE any middleware registration (after line 14)
   - Add Sentry.Handlers.requestHandler() as FIRST middleware (after cors())
   - Add Sentry.Handlers.tracingHandler() after requestHandler
   - Add Sentry.Handlers.errorHandler() BEFORE the custom errorHandler middleware (line 137)
   - All Sentry handlers wrapped in conditional: `if (process.env.SENTRY_DSN)`

4. Update `server/src/config/env.ts`:
   - Add SENTRY_DSN to config object (optional, no validation needed)
   - Export sentryDsn config value

5. Update `server/src/middleware/errorHandler.ts`:
   - Import Sentry from config/sentry
   - Call Sentry.captureException(err) before logging for unhandled errors

### 31.2. Integrate Sentry SDK with FastAPI ML Service

**Status:** pending  
**Dependencies:** None  

Install and configure sentry-sdk[fastapi] for the Python ML service with FastAPI and SQLAlchemy integrations to capture errors and performance traces.

**Details:**

1. Add to `ml-service/requirements.txt` (after line 78, in Monitoring section):
   ```
   sentry-sdk[fastapi]==1.40.0
   ```

2. Create `ml-service/app/core/sentry.py`:
   - Import sentry_sdk, FastApiIntegration, SqlalchemyIntegration
   - Create init_sentry() function checking settings.sentry_dsn
   - Configure: dsn, environment, release (app_version), traces_sample_rate (0.1 prod, 1.0 dev)
   - Add FastApiIntegration() and SqlalchemyIntegration()
   - Set send_default_pii=False
   - Add before_send callback for scrubbing (implemented in subtask 3)

3. Update `ml-service/app/config.py`:
   - Add `sentry_dsn: Optional[str] = None` to Settings class (after line 77)
   - Add `sentry_traces_sample_rate: float = 0.1` for configurability

4. Update `ml-service/app/main.py`:
   - Import init_sentry from app.core.sentry (after line 20)
   - Call init_sentry() at the TOP of lifespan function (line 104, before any other initialization)

5. Update exception handler at line 402:
   - Import sentry_sdk
   - Call sentry_sdk.capture_exception(exc) before logging the error

### 31.3. Implement Sensitive Data Scrubbing for Both Services

**Status:** pending  
**Dependencies:** 31.1, 31.2  

Create beforeSend hooks for both backend and ML service to scrub sensitive data including authorization tokens, cookies, passwords, and API keys from error reports.

**Details:**

1. Update `server/src/config/sentry.ts` beforeSend hook:
   ```typescript
   beforeSend(event, hint) {
     // Scrub request headers
     if (event.request?.headers) {
       delete event.request.headers.authorization;
       delete event.request.headers.cookie;
       delete event.request.headers['x-api-key'];
     }
     // Scrub request body for sensitive fields
     if (event.request?.data) {
       const sensitiveFields = ['password', 'token', 'secret', 'apiKey', 'creditCard'];
       for (const field of sensitiveFields) {
         if (event.request.data[field]) {
           event.request.data[field] = '[REDACTED]';
         }
       }
     }
     // Scrub user data
     if (event.user?.email) {
       // Keep partial email for debugging: 'j***@example.com'
       const [local, domain] = event.user.email.split('@');
       event.user.email = `${local[0]}***@${domain}`;
     }
     return event;
   }
   ```

2. Update `ml-service/app/core/sentry.py` scrub_sensitive_data function:
   ```python
   def scrub_sensitive_data(event, hint):
       # Scrub request headers
       if 'request' in event and 'headers' in event['request']:
           headers = event['request']['headers']
           for header in ['authorization', 'cookie', 'x-api-key']:
               if header in headers:
                   headers[header] = '[REDACTED]'
       
       # Scrub request body
       if 'request' in event and 'data' in event['request']:
           sensitive_fields = ['password', 'token', 'secret', 'api_key', 'credit_card']
           for field in sensitive_fields:
               if field in event['request']['data']:
                   event['request']['data'][field] = '[REDACTED]'
       
       # Scrub user email (partial)
       if 'user' in event and 'email' in event.get('user', {}):
           email = event['user']['email']
           if '@' in email:
               local, domain = email.split('@')
               event['user']['email'] = f"{local[0]}***@{domain}"
       
       return event
   ```

3. Add denyUrls and ignoreErrors config to filter out expected errors:
   - Ignore 401/403 authentication errors (expected behavior)
   - Ignore rate limiting responses
   - Ignore known third-party script errors

### 31.4. Configure GitHub Actions for Source Map Upload

**Status:** pending  
**Dependencies:** 31.1  

Add Sentry CLI source map upload step to build.yml workflow for both backend TypeScript and any bundled assets to enable proper stack trace deobfuscation in production.

**Details:**

1. Update `.github/workflows/build.yml` - Add to build-backend job (after line 64):
   ```yaml
   - name: Build TypeScript with source maps
     run: |
       cd server
       npm run build
     
   - name: Create Sentry release and upload source maps
     if: env.SENTRY_AUTH_TOKEN != ''
     env:
       SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}
       SENTRY_ORG: ${{ secrets.SENTRY_ORG }}
       SENTRY_PROJECT: nutri-backend
     run: |
       npm install -g @sentry/cli
       cd server
       sentry-cli releases new ${{ github.sha }}
       sentry-cli releases files ${{ github.sha }} upload-sourcemaps ./dist --ext ts --ext js --ext map
       sentry-cli releases set-commits ${{ github.sha }} --auto
       sentry-cli releases finalize ${{ github.sha }}
       sentry-cli releases deploys ${{ github.sha }} new -e production
   ```

2. Update `server/tsconfig.json` to generate source maps:
   - Ensure `"sourceMap": true` is set in compilerOptions
   - Add `"inlineSources": true` for better stack traces

3. Add documentation for required GitHub secrets:
   - SENTRY_AUTH_TOKEN: Generated from Sentry Settings > API Keys
   - SENTRY_ORG: Organization slug from Sentry
   - SENTRY_PROJECT: Project slug (nutri-backend, nutri-ml-service)

4. Update Sentry config in both services to include release tag:
   - Backend: `release: process.env.GITHUB_SHA || process.env.npm_package_version`
   - ML: `release: os.environ.get('GITHUB_SHA', settings.app_version)`

### 31.5. Configure Sentry Alerts and Verify End-to-End Integration

**Status:** pending  
**Dependencies:** 31.1, 31.2, 31.3, 31.4  

Set up Sentry alert rules for error rate thresholds and new issues, add test endpoints for verification, and create documentation for the team.

**Details:**

1. Create test error endpoints for verification:
   
   Backend - Add to a new route or existing test route:
   ```typescript
   // GET /api/debug/sentry-test (only in non-production)
   if (process.env.NODE_ENV !== 'production') {
     app.get('/api/debug/sentry-test', (req, res) => {
       throw new Error('Test Sentry integration - Backend');
     });
   }
   ```
   
   ML Service - Add test endpoint:
   ```python
   @app.get('/debug/sentry-test', tags=['Debug'])
   async def test_sentry():
       if settings.environment == 'production':
           raise HTTPException(status_code=404)
       raise ValueError('Test Sentry integration - ML Service')
   ```

2. Document Sentry project setup in `docs/sentry-setup.md`:
   - Create two Sentry projects: nutri-backend, nutri-ml-service
   - Configure alert rules via Sentry dashboard:
     * New Issue Alert: Notify on first occurrence of new errors
     * Error Rate Alert: >10 errors in 5 minutes
     * Performance Alert: P95 response time >2s
   - Configure integrations: Slack/Email for alerts
   - Team member notification preferences

3. Update `.env.example` and `.env.prod.example` files:
   - Add SENTRY_DSN placeholder with comment
   - Add SENTRY_AUTH_TOKEN for CI/CD
   - Add SENTRY_ORG and SENTRY_PROJECT vars

4. Add environment variables documentation:
   - Update CLAUDE.md environment variables section
   - Document required vs optional Sentry config

5. Create verification checklist:
   - [ ] Trigger test error in dev
   - [ ] Verify error appears in Sentry with correct stack trace
   - [ ] Verify source maps resolve to original code
   - [ ] Verify sensitive data is scrubbed
   - [ ] Verify alerts fire on test errors
   - [ ] Test performance monitoring captures traces
