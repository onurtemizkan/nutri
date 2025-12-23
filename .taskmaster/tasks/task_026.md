# Task ID: 26

**Title:** Implement Structured Logging for Backend

**Status:** done

**Dependencies:** None

**Priority:** medium

**Description:** Add JSON structured logging to the Express backend with correlation IDs, proper log levels, and sensitive data redaction.

**Details:**

**Install dependencies in server/:**
```bash
npm install pino pino-http @types/pino-http
```

**Create `server/src/config/logger.ts`:**
```typescript
import pino from 'pino';
import { randomUUID } from 'crypto';

const isProduction = process.env.NODE_ENV === 'production';

export const logger = pino({
  level: process.env.LOG_LEVEL || (isProduction ? 'info' : 'debug'),
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  redact: {
    paths: [
      'req.headers.authorization',
      'req.body.password',
      'req.body.token',
      'res.headers["set-cookie"]',
      '*.password',
      '*.token',
      '*.jwt',
      '*.secret',
    ],
    censor: '[REDACTED]',
  },
  transport: isProduction ? undefined : {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
    },
  },
});

export const generateCorrelationId = (): string => randomUUID();

export type Logger = typeof logger;
```

**Create `server/src/middleware/requestLogger.ts`:**
```typescript
import pinoHttp from 'pino-http';
import { logger, generateCorrelationId } from '../config/logger';
import { Request, Response } from 'express';

export const requestLogger = pinoHttp({
  logger,
  genReqId: (req: Request) => {
    const existingId = req.headers['x-correlation-id'];
    return (existingId as string) || generateCorrelationId();
  },
  customProps: (req: Request) => ({
    correlationId: req.id,
    service: 'nutri-backend',
    environment: process.env.NODE_ENV,
  }),
  customLogLevel: (_req: Request, res: Response, err?: Error) => {
    if (res.statusCode >= 500 || err) return 'error';
    if (res.statusCode >= 400) return 'warn';
    return 'info';
  },
  customSuccessMessage: (req: Request, res: Response) => {
    return `${req.method} ${req.url} completed with ${res.statusCode}`;
  },
  customErrorMessage: (_req: Request, res: Response, err: Error) => {
    return `Request failed: ${err.message}`;
  },
  serializers: {
    req: (req) => ({
      method: req.method,
      url: req.url,
      query: req.query,
      params: req.params,
      // Don't log body by default - too verbose
    }),
    res: (res) => ({
      statusCode: res.statusCode,
    }),
  },
});
```

**Update `server/src/index.ts`:**
```typescript
import { requestLogger } from './middleware/requestLogger';
import { logger } from './config/logger';

// Add after cors middleware
app.use(requestLogger);

// Update server start log
if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, () => {
    logger.info({ port: PORT, env: config.nodeEnv }, 'Server started');
  });
}
```

**Update error handler to use logger:**
```typescript
import { logger } from '../config/logger';

// In errorHandler middleware:
logger.error({
  err,
  correlationId: req.id,
  statusCode,
  path: req.path,
  method: req.method,
}, 'Request error');
```

**Add pino-pretty for dev (devDependency):**
```bash
npm install -D pino-pretty
```

**Test Strategy:**

1. Start server and verify JSON log output in production mode
2. Verify correlation ID appears in logs and response headers
3. Test password/token redaction - should show [REDACTED]
4. Verify log levels work (DEBUG shows more than INFO)
5. Test error logging includes stack trace
6. Verify pino-pretty works in development
7. Test high-volume requests don't cause log overflow
8. Verify logs are compatible with common log aggregators (JSON format)

## Subtasks

### 26.1. Install pino dependencies and configure logger with redaction

**Status:** pending  
**Dependencies:** None  

Install pino, pino-http, @types/pino-http, and pino-pretty (dev). Create server/src/config/logger.ts with structured logging configuration including sensitive data redaction, correlation ID generation, and environment-based formatting.

**Details:**

Run `npm install pino pino-http @types/pino-http` and `npm install -D pino-pretty` in server/ directory. Create server/src/config/logger.ts implementing: pino instance with level based on LOG_LEVEL env var (default 'info' production, 'debug' development), ISO timestamp formatting, redaction paths for sensitive fields (authorization headers, password, token, jwt, secret fields), correlation ID generator using randomUUID from crypto, pino-pretty transport for development (colorized, human-readable), and export logger instance with TypeScript type. Test locally that logger initializes without errors.

### 26.2. Create requestLogger middleware with custom log levels

**Status:** pending  
**Dependencies:** 26.1  

Implement pino-http middleware in server/src/middleware/requestLogger.ts that integrates structured logging with Express, handles correlation IDs from headers, and applies custom log levels based on HTTP status codes.

**Details:**

Create server/src/middleware/requestLogger.ts using pinoHttp from pino-http. Configure genReqId to extract x-correlation-id header or generate new UUID, add customProps with correlationId/service/environment metadata, implement customLogLevel function (500+ or error → 'error', 400-499 → 'warn', else 'info'), add customSuccessMessage and customErrorMessage formatters, configure serializers for req (method, url, query, params only) and res (statusCode only) to avoid verbose body logging. Export requestLogger middleware function.

### 26.3. Integrate requestLogger into Express app and replace console.log

**Status:** pending  
**Dependencies:** 26.2  

Update server/src/index.ts to use the requestLogger middleware and replace all console.log statements with structured logger calls throughout the backend codebase.

**Details:**

In server/src/index.ts: import requestLogger from './middleware/requestLogger' and logger from './config/logger', add app.use(requestLogger) after CORS middleware but before routes, replace server start console.log with logger.info({ port: PORT, env: config.nodeEnv }, 'Server started'). Search codebase for all console.log/console.error statements in server/src and replace with appropriate logger.info/logger.error/logger.warn/logger.debug calls. Verify NODE_ENV=production produces JSON logs and NODE_ENV=development produces colorized pino-pretty output.

### 26.4. Update errorHandler middleware to use structured logging

**Status:** pending  
**Dependencies:** 26.2  

Modify the existing error handling middleware in server/src/middleware/errorHandler.ts to use structured logging with correlation IDs, proper error serialization, and contextual request information.

**Details:**

In server/src/middleware/errorHandler.ts: import logger from '../config/logger', replace any console.error calls with logger.error(), include error object, correlationId (req.id), statusCode, path (req.path), method (req.method), and error message in log context. Ensure error stack traces are included in development but sanitized in production. Test with various error scenarios (validation errors, auth errors, 500 errors) to verify proper logging with correlation IDs. Verify sensitive data (passwords, tokens) are redacted even in error logs.
