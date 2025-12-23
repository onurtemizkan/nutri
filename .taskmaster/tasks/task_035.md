# Task ID: 35

**Title:** Configure Application Security Headers

**Status:** pending

**Dependencies:** None

**Priority:** medium

**Description:** Implement security headers (HSTS, CSP, X-Frame-Options, etc.) in the backend API and configure HTTPS-only access with proper CORS settings.

**Details:**

**Install helmet for Express:**
```bash
cd server && npm install helmet
```

**Create `server/src/middleware/security.ts`:**
```typescript
import helmet from 'helmet';
import { Express, Request, Response, NextFunction } from 'express';

export function configureSecurityHeaders(app: Express): void {
  // Basic helmet protection
  app.use(helmet());

  // Strict Transport Security (HTTPS only)
  app.use(
    helmet.hsts({
      maxAge: 31536000, // 1 year in seconds
      includeSubDomains: true,
      preload: true,
    })
  );

  // Content Security Policy
  app.use(
    helmet.contentSecurityPolicy({
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"], // React may need inline styles
        imgSrc: ["'self'", 'data:', 'https:'],
        connectSrc: ["'self'", 'https://api.nutri.app'],
        fontSrc: ["'self'"],
        objectSrc: ["'none'"],
        mediaSrc: ["'self'"],
        frameSrc: ["'none'"],
      },
    })
  );

  // Prevent clickjacking
  app.use(helmet.frameguard({ action: 'deny' }));

  // Prevent MIME type sniffing
  app.use(helmet.noSniff());

  // XSS protection (legacy browsers)
  app.use(helmet.xssFilter());

  // Referrer policy
  app.use(helmet.referrerPolicy({ policy: 'strict-origin-when-cross-origin' }));

  // Remove X-Powered-By header
  app.disable('x-powered-by');

  // Custom security headers
  app.use((_req: Request, res: Response, next: NextFunction) => {
    // Permissions Policy (replaces Feature-Policy)
    res.setHeader(
      'Permissions-Policy',
      'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()'
    );
    
    // Cross-Origin policies
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    
    next();
  });
}
```

**Update CORS configuration in `server/src/index.ts`:**
```typescript
import cors from 'cors';
import { configureSecurityHeaders } from './middleware/security';

const app = express();

// Configure CORS with specific origins
const allowedOrigins = [
  'http://localhost:3000',
  'http://localhost:8081', // Expo dev
  'https://nutri.app',
  'https://www.nutri.app',
  process.env.CORS_ORIGIN, // Allow override via env
].filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (mobile apps, curl, etc.)
      if (!origin) return callback(null, true);
      
      if (allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Correlation-ID'],
    exposedHeaders: ['X-Correlation-ID'],
    maxAge: 86400, // 24 hours
  })
);

// Configure security headers
configureSecurityHeaders(app);
```

**Add HTTPS redirect middleware (for when behind Traefik/reverse proxy):**
```typescript
// Trust proxy (Traefik)
app.set('trust proxy', 1);

// Redirect HTTP to HTTPS in production
if (process.env.NODE_ENV === 'production') {
  app.use((req, res, next) => {
    if (req.header('x-forwarded-proto') !== 'https') {
      res.redirect(301, `https://${req.header('host')}${req.url}`);
    } else {
      next();
    }
  });
}
```

**Test headers with curl:**
```bash
curl -I https://api.nutri.app/health
```

Expected headers:
```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'; ...
Referrer-Policy: strict-origin-when-cross-origin
```

**Test Strategy:**

1. Add security middleware and verify app still works
2. Test all API endpoints work with new CORS config
3. Use securityheaders.com to scan API and verify A+ rating
4. Test mobile app still connects with CORS
5. Verify X-Powered-By header is removed
6. Test HTTPS redirect works in production mode
7. Verify CSP doesn't break legitimate requests
8. Test preflight OPTIONS requests work correctly
