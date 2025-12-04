# Apple Sign In - Production Requirements

## Overview

This document outlines the requirements and steps needed to move Apple Sign In from development to production.

## Current Status

✅ **Implemented (Development)**
- Database schema with `appleId` field
- Backend API endpoint (`POST /auth/apple-signin`)
- Mobile app integration (iOS sign-in buttons)
- Basic token decoding (development only)
- User account creation and linking

⚠️ **Not Implemented (Production Required)**
- Apple identity token verification with Apple's servers
- Production-grade security measures
- App Store Connect configuration
- Error monitoring and logging
- Rate limiting for OAuth endpoints

---

## 1. Security: Token Verification

### Current Implementation (Development Only)

**Location:** `server/src/services/authService.ts:221-323`

The current implementation decodes the identity token WITHOUT verification:

```typescript
// ⚠️ DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION
const payload = JSON.parse(Buffer.from(tokenParts[1], 'base64').toString('utf8'));
```

### Required for Production

**Install dependencies:**

```bash
npm install jsonwebtoken jwks-rsa
npm install --save-dev @types/jsonwebtoken @types/jwks-rsa
```

**Implementation:**

```typescript
import jwt from 'jsonwebtoken';
import jwksClient from 'jwks-rsa';

// Create JWKS client for Apple's public keys
const appleJwksClient = jwksClient({
  jwksUri: 'https://appleid.apple.com/auth/keys',
  cache: true,
  cacheMaxAge: 86400000, // 24 hours
  rateLimit: true,
  jwksRequestsPerMinute: 10,
});

// Helper to get signing key
function getAppleSigningKey(header: jwt.JwtHeader, callback: jwt.SigningKeyCallback) {
  appleJwksClient.getSigningKey(header.kid!, (err, key) => {
    if (err) {
      callback(err);
      return;
    }
    const signingKey = key?.getPublicKey();
    callback(null, signingKey);
  });
}

// Verify token
async function verifyAppleToken(identityToken: string): Promise<AppleTokenPayload> {
  return new Promise((resolve, reject) => {
    jwt.verify(
      identityToken,
      getAppleSigningKey,
      {
        algorithms: ['RS256'],
        audience: process.env.APPLE_APP_ID!, // Your bundle identifier
        issuer: 'https://appleid.apple.com',
      },
      (err, decoded) => {
        if (err) {
          reject(new Error('Invalid Apple identity token'));
          return;
        }
        resolve(decoded as AppleTokenPayload);
      }
    );
  });
}

// Update appleSignIn method
async appleSignIn(data: { identityToken: string; ... }) {
  // Verify token with Apple's servers
  const payload = await verifyAppleToken(data.identityToken);

  const appleId = payload.sub;
  const email = payload.email || data.user?.email;

  // ... rest of implementation
}
```

**Type definitions:**

```typescript
interface AppleTokenPayload {
  iss: string; // https://appleid.apple.com
  aud: string; // Your app's bundle identifier
  exp: number; // Expiration time
  iat: number; // Issued at time
  sub: string; // Apple user ID
  email?: string; // User's email (only on first sign-in)
  email_verified?: boolean;
  is_private_email?: boolean;
  nonce_supported?: boolean;
}
```

**Environment variables:**

Add to `server/.env`:

```env
APPLE_APP_ID=your.bundle.identifier
APPLE_TEAM_ID=YOUR_TEAM_ID
```

---

## 2. Apple Developer Account Configuration

### App ID Setup

1. **Go to:** https://developer.apple.com/account/resources/identifiers/list
2. **Select your App ID**
3. **Enable:** "Sign in with Apple"
4. **Configure:**
   - Primary App ID (if using grouped App IDs)
   - Edit configuration if needed

### Services ID (Optional - for Web/Android)

If you plan to support Apple Sign In on web or Android:

1. **Create Services ID**
2. **Enable:** "Sign in with Apple"
3. **Configure:**
   - Domains: `yourdomain.com`
   - Return URLs: `https://yourdomain.com/auth/apple/callback`

### Keys (for Server-to-Server Token Validation)

1. **Create Key** with "Sign in with Apple" enabled
2. **Download** the key file (only available once!)
3. **Note:** Key ID and Team ID
4. **Store securely** in production environment

---

## 3. Mobile App Configuration

### Expo Configuration

**Update `app.json`:**

```json
{
  "expo": {
    "name": "Nutri",
    "ios": {
      "bundleIdentifier": "com.yourcompany.nutri",
      "usesAppleSignIn": true,
      "infoPlist": {
        "NSAppTransportSecurity": {
          "NSAllowsArbitraryLoads": false
        }
      }
    },
    "plugins": [
      "expo-apple-authentication"
    ]
  }
}
```

### Build Requirements

Apple Sign In requires native iOS code and will NOT work with Expo Go:

**Development builds:**

```bash
npx expo run:ios
# or
eas build --profile development --platform ios
```

**Production builds:**

```bash
eas build --profile production --platform ios
```

### App Store Connect

1. **Enable capability** in Xcode:
   - Signing & Capabilities → + Capability → Sign in with Apple
2. **Or use EAS:**
   - Capabilities are automatically configured from `app.json`

---

## 4. Database Migration

### Apply Schema Changes

**Run migration:**

```bash
cd server
npm run db:migrate -- --name add_apple_sign_in
```

**Migration includes:**
- `appleId` String? @unique
- `password` String? (changed from required to optional)
- `@@index([appleId])`

### Backup Strategy

Before deploying to production:

1. **Backup database:**
   ```bash
   pg_dump -U username -d nutri_db > backup_before_apple_signin.sql
   ```

2. **Test migration on staging** environment first

3. **Verify:**
   - Existing users can still log in
   - New Apple users can sign up
   - Email linking works correctly

---

## 5. Security Enhancements

### Rate Limiting

**Add to `server/src/middleware/rateLimiter.ts`:**

```typescript
export const appleAuthLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // 10 requests per window per IP
  message: 'Too many Apple sign-in attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});
```

**Apply to route (`server/src/routes/authRoutes.ts`):**

```typescript
import { appleAuthLimiter } from '../middleware/rateLimiter';

router.post('/apple-signin', appleAuthLimiter, (req, res) =>
  authController.appleSignIn(req, res)
);
```

### Input Validation

Already implemented via Zod schema (`appleSignInSchema`), but verify:

- Identity token format
- Authorization code format
- Email format (if provided)
- Name validation (if provided)

### Logging and Monitoring

**Add to `authService.appleSignIn()`:**

```typescript
// Log authentication attempts (without sensitive data)
logger.info('Apple sign-in attempt', {
  hasEmail: !!email,
  isNewUser: !user,
  timestamp: new Date().toISOString(),
});

// Log failures
logger.error('Apple sign-in failed', {
  error: error.message,
  hasEmail: !!email,
});
```

**Monitoring alerts:**
- Failed token verification rate
- New user creation failures
- Account linking issues
- Unusual sign-in patterns

---

## 6. Error Handling

### Client-Side Error Messages

**Update mobile app error handling:**

```typescript
// app/auth/signin.tsx & signup.tsx
catch (error: unknown) {
  // User canceled
  if (error.code === 'ERR_REQUEST_CANCELED') {
    return;
  }

  // Apple service unavailable
  if (error.code === 'ERR_REQUEST_UNKNOWN') {
    Alert.alert(
      'Service Unavailable',
      'Apple Sign In is temporarily unavailable. Please try again or use email sign-in.'
    );
    return;
  }

  // Network errors
  if (isAxiosError(error) && !error.response) {
    Alert.alert(
      'Connection Error',
      'Please check your internet connection and try again.'
    );
    return;
  }

  // Backend errors
  if (isAxiosError(error) && error.response?.status === 401) {
    Alert.alert(
      'Authentication Failed',
      'Could not verify your Apple account. Please try again.'
    );
    return;
  }

  // Generic error
  Alert.alert(
    'Sign In Failed',
    getErrorMessage(error, 'Could not sign in with Apple')
  );
}
```

### Server-Side Error Codes

**Define error types (`server/src/types/errors.ts`):**

```typescript
export class AppleAuthError extends Error {
  constructor(
    message: string,
    public code: 'INVALID_TOKEN' | 'EXPIRED_TOKEN' | 'EMAIL_REQUIRED' | 'SERVICE_ERROR'
  ) {
    super(message);
    this.name = 'AppleAuthError';
  }
}
```

**Use in authService:**

```typescript
if (!appleId) {
  throw new AppleAuthError('Invalid identity token', 'INVALID_TOKEN');
}

if (!email && !user) {
  throw new AppleAuthError('Email is required for new users', 'EMAIL_REQUIRED');
}
```

---

## 7. Testing Requirements

### Unit Tests

**Add to `server/src/__tests__/auth.test.ts`:**

```typescript
describe('Apple Sign In', () => {
  describe('POST /auth/apple-signin', () => {
    it('should create new user with Apple ID', async () => {
      // Test implementation
    });

    it('should link Apple ID to existing user by email', async () => {
      // Test implementation
    });

    it('should sign in existing Apple user', async () => {
      // Test implementation
    });

    it('should reject invalid identity token', async () => {
      // Test implementation
    });

    it('should handle missing email for new users', async () => {
      // Test implementation
    });
  });
});
```

### Integration Tests

**Test flows:**
1. ✅ New user signs up with Apple (with email)
2. ✅ New user signs up with Apple (email hidden)
3. ✅ Existing user links Apple ID to account
4. ✅ User signs in with linked Apple ID
5. ✅ Invalid token is rejected
6. ✅ Expired token is rejected

### Manual Testing Checklist

- [ ] Sign in with Apple on physical iOS device
- [ ] Sign up with Apple (new user)
- [ ] Link Apple ID to existing email account
- [ ] Sign in after linking
- [ ] Test with private relay email
- [ ] Test with real email
- [ ] Verify user can access protected endpoints
- [ ] Test account deletion/unlinking
- [ ] Test on different iOS versions (13.5+)
- [ ] Test in production-like environment (TestFlight)

---

## 8. Privacy and Compliance

### Data Handling

Apple requires specific privacy practices:

1. **Email Relay:** Support users who choose "Hide My Email"
   - Private relay emails look like: `random@privaterelay.appleid.com`
   - Treat as valid, primary email address
   - Store and respect user's privacy choice

2. **Data Minimization:**
   - Only request email and name scopes
   - Don't request more data than needed

3. **Account Deletion:**
   - Provide way to unlink Apple ID
   - Comply with "Sign in with Apple" account deletion requirements

**Implement unlink endpoint:**

```typescript
// server/src/controllers/authController.ts
async unlinkApple(req: AuthenticatedRequest, res: Response): Promise<void> {
  const userId = requireAuth(req, res);
  if (!userId) return;

  await prisma.user.update({
    where: { id: userId },
    data: { appleId: null },
  });

  res.status(200).json({ message: 'Apple ID unlinked successfully' });
}
```

### Privacy Policy

Update your app's privacy policy to include:

- Apple Sign In usage
- Data collected (email, name)
- How Apple user IDs are stored
- Email relay support
- Account deletion process

### App Store Review

Apple requires:
- Privacy policy URL in App Store Connect
- Account deletion method (if accounts are created)
- Proper handling of Sign in with Apple button (design guidelines)

---

## 9. Deployment Checklist

### Pre-Deployment

- [ ] Token verification implemented and tested
- [ ] Environment variables configured (`APPLE_APP_ID`, etc.)
- [ ] Database migration tested on staging
- [ ] Rate limiting configured
- [ ] Error handling and logging added
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Manual testing completed on TestFlight

### Deployment Steps

1. **Backend Deployment:**
   ```bash
   # Set environment variables
   export APPLE_APP_ID="com.yourcompany.nutri"
   export APPLE_TEAM_ID="YOUR_TEAM_ID"

   # Run migration
   npm run db:migrate

   # Deploy to production
   npm run build
   npm start
   ```

2. **Mobile Deployment:**
   ```bash
   # Build production app
   eas build --profile production --platform ios

   # Submit to App Store
   eas submit --platform ios
   ```

3. **Monitoring:**
   - Watch error logs for first 24 hours
   - Monitor authentication success rate
   - Track new user signups via Apple
   - Monitor account linking rate

### Post-Deployment

- [ ] Verify Apple Sign In works in production
- [ ] Monitor error rates
- [ ] Check authentication logs
- [ ] Verify database records are created correctly
- [ ] Test account linking in production
- [ ] Verify rate limiting is working
- [ ] Check performance metrics

---

## 10. Troubleshooting

### Common Issues

**"Invalid client" error:**
- Verify `APPLE_APP_ID` matches bundle identifier
- Check Apple Developer Console configuration
- Ensure App ID has Sign in with Apple enabled

**Token verification fails:**
- Check Apple's JWKS endpoint is accessible
- Verify token hasn't expired (exp claim)
- Ensure audience (aud) matches your App ID
- Check issuer (iss) is https://appleid.apple.com

**Email not provided:**
- Email is only provided on first authentication
- Store email from first sign-in
- Handle missing email gracefully for returning users
- Consider asking user to provide email manually if needed

**Button doesn't appear:**
- Check `Platform.OS === 'ios'` condition
- Verify running on physical device or simulator (not Expo Go)
- Check `usesAppleSignIn: true` in app.json
- Ensure iOS version is 13.5+

**App Store rejection:**
- Follow Apple's Human Interface Guidelines
- Don't customize button appearance excessively
- Provide prominent Apple Sign In option if other social logins exist
- Include proper privacy policy
- Implement account deletion

---

## 11. Performance Optimization

### Token Verification Caching

Cache Apple's public keys to reduce latency:

```typescript
const appleJwksClient = jwksClient({
  jwksUri: 'https://appleid.apple.com/auth/keys',
  cache: true,
  cacheMaxAge: 86400000, // 24 hours
});
```

### Database Queries

Optimize user lookup:

```typescript
// Use OR query with proper indexes
const user = await prisma.user.findFirst({
  where: {
    OR: [
      { appleId },
      { email: email || undefined },
    ],
  },
});
```

Verify indexes exist:
- `@@index([appleId])`
- `@@index([email])`

---

## 12. Support and Maintenance

### Monitoring Dashboards

Track these metrics:

- Apple sign-in attempts per day
- Success rate vs. failure rate
- New users created via Apple
- Account linking rate
- Token verification errors
- Average response time

### User Support

Prepare support documentation:

- How to sign in with Apple
- How to link Apple ID to existing account
- How to unlink Apple ID
- What to do if sign-in fails
- Privacy relay email explanation

### Maintenance Schedule

- Monthly: Review error logs and success rates
- Quarterly: Update dependencies (jsonwebtoken, jwks-rsa)
- Annually: Review Apple's Sign in with Apple guidelines for changes

---

## 13. Resources

### Documentation

- [Apple Sign In REST API](https://developer.apple.com/documentation/sign_in_with_apple/sign_in_with_apple_rest_api)
- [Expo Apple Authentication](https://docs.expo.dev/versions/latest/sdk/apple-authentication/)
- [Token Verification Guide](https://developer.apple.com/documentation/sign_in_with_apple/sign_in_with_apple_rest_api/verifying_a_user)
- [Apple HIG: Sign in with Apple](https://developer.apple.com/design/human-interface-guidelines/sign-in-with-apple)

### Code References

- Backend: `server/src/services/authService.ts:221-323`
- Controller: `server/src/controllers/authController.ts:195-214`
- Routes: `server/src/routes/authRoutes.ts:10`
- Mobile: `app/auth/signin.tsx`, `app/auth/signup.tsx`
- Schema: `server/prisma/schema.prisma:13-52`

---

## 14. Rollback Plan

If critical issues arise in production:

### Immediate Actions

1. **Disable endpoint** (server/src/routes/authRoutes.ts):
   ```typescript
   // Comment out Apple sign-in route temporarily
   // router.post('/apple-signin', (req, res) => authController.appleSignIn(req, res));
   ```

2. **Deploy hotfix** to hide button (mobile):
   ```typescript
   // Temporarily disable Apple button
   const APPLE_SIGNIN_ENABLED = false;

   {Platform.OS === 'ios' && APPLE_SIGNIN_ENABLED && (
     // Button code
   )}
   ```

3. **Monitor existing Apple users:**
   - Ensure they can still access accounts
   - Don't remove `appleId` from database

### Recovery

1. Fix underlying issue
2. Test thoroughly on staging
3. Re-enable feature
4. Monitor closely

### Data Integrity

- Never delete `appleId` field from database
- Preserve user accounts created via Apple
- Allow alternative sign-in method (email/password recovery)

---

**Last Updated:** 2025-01-20
**Status:** Development Complete, Production Pending
**Priority:** High - Required for App Store compliance if offering social sign-in
