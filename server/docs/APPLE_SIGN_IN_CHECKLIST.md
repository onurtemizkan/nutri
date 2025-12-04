# Apple Sign In - Production Deployment Checklist

Quick reference checklist for deploying Apple Sign In to production.

**For detailed information, see:** [APPLE_SIGN_IN_PRODUCTION.md](./APPLE_SIGN_IN_PRODUCTION.md)

---

## Pre-Deployment

### 1. Security Implementation

- [ ] Install dependencies: `jsonwebtoken`, `jwks-rsa`
- [ ] Implement token verification (replace development decoder)
- [ ] Add Apple JWKS client configuration
- [ ] Add `getAppleSigningKey()` helper function
- [ ] Update `appleSignIn()` to use `verifyAppleToken()`
- [ ] Add `AppleTokenPayload` type definition

**Location:** `server/src/services/authService.ts:232-256`

### 2. Environment Configuration

- [ ] Add `APPLE_APP_ID` to production `.env`
- [ ] Add `APPLE_TEAM_ID` to production `.env`
- [ ] Verify bundle identifier matches Apple Developer Console

**File:** `server/.env`

```env
APPLE_APP_ID=com.yourcompany.nutri
APPLE_TEAM_ID=YOUR_TEAM_ID
```

### 3. Apple Developer Setup

- [ ] Enable "Sign in with Apple" for App ID
- [ ] Create/configure Services ID (if needed for web)
- [ ] Create signing key (download and store securely)
- [ ] Note Key ID and Team ID

**Portal:** https://developer.apple.com/account/resources/identifiers/list

### 4. Mobile App Configuration

- [ ] Update `app.json` with correct `bundleIdentifier`
- [ ] Set `usesAppleSignIn: true` in `app.json`
- [ ] Add `expo-apple-authentication` to plugins
- [ ] Test build with `npx expo run:ios` or `eas build`

**File:** `app.json`

### 5. Rate Limiting

- [ ] Add `appleAuthLimiter` to `rateLimiter.ts`
- [ ] Apply to `/auth/apple-signin` route
- [ ] Configure appropriate limits (e.g., 10 per 15 min)

**Files:** `server/src/middleware/rateLimiter.ts`, `server/src/routes/authRoutes.ts`

### 6. Error Handling

- [ ] Add `AppleAuthError` class to types
- [ ] Update error messages in authService
- [ ] Enhance client-side error handling
- [ ] Add comprehensive error codes

**Files:** `server/src/types/errors.ts`, `app/auth/signin.tsx`, `app/auth/signup.tsx`

### 7. Logging and Monitoring

- [ ] Add authentication attempt logging
- [ ] Add failure logging (without sensitive data)
- [ ] Set up monitoring alerts for:
  - Failed token verification rate
  - New user creation failures
  - Account linking issues

**Location:** `server/src/services/authService.ts`

### 8. Testing

- [ ] Write unit tests for `appleSignIn()` method
- [ ] Test new user signup flow
- [ ] Test account linking flow
- [ ] Test existing user sign-in flow
- [ ] Test invalid token rejection
- [ ] Manual testing on physical iOS device
- [ ] Test with private relay email
- [ ] Test on TestFlight build

**Files:** `server/src/__tests__/auth.test.ts`

### 9. Database

- [ ] Backup production database
- [ ] Test migration on staging environment
- [ ] Verify migration script: `add_apple_sign_in`
- [ ] Plan rollback strategy

**Command:** `npm run db:migrate -- --name add_apple_sign_in`

### 10. Privacy and Compliance

- [ ] Update privacy policy to include Apple Sign In
- [ ] Implement account unlinking endpoint
- [ ] Add unlink option in user profile (mobile)
- [ ] Verify email relay support
- [ ] Prepare account deletion flow

---

## Deployment Day

### Backend Deployment

```bash
# 1. Set environment variables
export APPLE_APP_ID="com.yourcompany.nutri"
export APPLE_TEAM_ID="YOUR_TEAM_ID"

# 2. Backup database
pg_dump -U username -d nutri_db > backup_$(date +%Y%m%d).sql

# 3. Run migration
npm run db:migrate

# 4. Build and deploy
npm run build
npm start

# 5. Verify health
curl https://api.yourdomain.com/health
```

### Mobile Deployment

```bash
# 1. Build production iOS app
eas build --profile production --platform ios

# 2. Test on TestFlight
eas submit --platform ios

# 3. Monitor initial downloads
```

---

## Post-Deployment Verification

### Immediate Checks (First Hour)

- [ ] Test Apple Sign In on production app
- [ ] Create test account via Apple
- [ ] Link Apple ID to existing account
- [ ] Sign in with linked account
- [ ] Check database for new user records
- [ ] Verify JWT tokens are issued correctly
- [ ] Check error logs for any failures

### 24-Hour Monitoring

- [ ] Monitor authentication success rate (target: >95%)
- [ ] Check error logs every 4 hours
- [ ] Track new user signups via Apple
- [ ] Monitor account linking rate
- [ ] Verify no database issues
- [ ] Check API response times (<500ms)

### Week-One Monitoring

- [ ] Daily review of error logs
- [ ] Track user adoption rate
- [ ] Monitor support requests
- [ ] Review token verification errors
- [ ] Check for unusual patterns
- [ ] Verify rate limiting is effective

---

## Rollback Plan (If Issues Arise)

### Emergency Rollback

1. **Disable endpoint immediately:**
   ```typescript
   // server/src/routes/authRoutes.ts
   // router.post('/apple-signin', (req, res) => authController.appleSignIn(req, res));
   ```

2. **Hide button in mobile:**
   ```typescript
   // Set to false in signin.tsx and signup.tsx
   const APPLE_SIGNIN_ENABLED = false;
   ```

3. **Deploy hotfix**

4. **Don't delete database fields** - preserve existing Apple users

---

## Metrics to Track

### Daily
- Total Apple sign-in attempts
- Success vs. failure rate
- New users created via Apple
- Account linking attempts/successes

### Weekly
- User retention (Apple vs. email signups)
- Support tickets related to Apple Sign In
- Token verification error rate
- Average authentication time

### Monthly
- Review Apple Sign In adoption rate
- Analyze most common errors
- Update dependencies if needed
- Review logs for security issues

---

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| "Invalid client" error | Verify `APPLE_APP_ID` matches bundle ID |
| Token verification fails | Check JWKS endpoint is accessible |
| Button doesn't appear | Ensure iOS 13.5+, not using Expo Go |
| Email not provided | Expected on repeat sign-ins (store on first) |
| Rate limiting too strict | Adjust `appleAuthLimiter` config |

---

## Support Resources

- **Full Documentation:** [APPLE_SIGN_IN_PRODUCTION.md](./APPLE_SIGN_IN_PRODUCTION.md)
- **Apple Docs:** https://developer.apple.com/sign-in-with-apple/
- **Expo Docs:** https://docs.expo.dev/versions/latest/sdk/apple-authentication/
- **Code Locations:**
  - Backend: `server/src/services/authService.ts:221-323`
  - Mobile: `app/auth/signin.tsx`, `app/auth/signup.tsx`
  - Schema: `server/prisma/schema.prisma:13-52`

---

## Status Tracking

**Current Status:** ✅ Development Complete

**Production Status:** ⏳ Pending

**Priority:** 🔴 High (Required for App Store if offering social sign-in)

---

**Last Updated:** 2025-01-20
