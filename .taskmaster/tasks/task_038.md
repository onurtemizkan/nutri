# Task ID: 38

**Title:** Implement In-App Purchases with Subscription Management

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement a complete in-app purchase system for selling subscription plans (monthly/yearly) using StoreKit 2, including trial periods, promotional offers, discount codes, purchase restoration, server-side receipt validation, and proper entitlement management.

**Details:**

## Overview
Implement a production-ready in-app purchase (IAP) system for the Nutri app using Apple's StoreKit 2 framework. The system will support auto-renewable subscriptions with multiple tiers, trial periods, promotional offers, and proper server-side validation.

## Subscription Tiers (Suggested)
1. **Nutri Free** - Basic tracking, limited history
2. **Nutri Pro (Monthly)** - Full features, ML insights, unlimited history
3. **Nutri Pro (Yearly)** - Same as monthly with discount (~2 months free)

## Technical Requirements

### 1. StoreKit 2 Implementation (React Native / Expo)
- Use `expo-in-app-purchases` or `react-native-iap` (prefer react-native-iap v12+ for StoreKit 2 support)
- Implement async/await patterns for all StoreKit operations
- Handle Transaction.updates listener for real-time purchase updates
- Implement proper transaction finishing to prevent duplicate charges
- Support for App Store sandbox testing environment

### 2. Product Configuration
- Configure products in App Store Connect:
  - `com.nutri.pro.monthly` - Monthly subscription
  - `com.nutri.pro.yearly` - Yearly subscription
- Set up subscription groups for upgrade/downgrade paths
- Configure introductory offers (free trial, pay-up-front, pay-as-you-go)
- Set up promotional offers for win-back campaigns
- Configure offer codes for marketing campaigns

### 3. Trial Periods & Offers
- Free trial: 7-day trial for new subscribers
- Introductory pricing: First month at 50% discount
- Promotional offers: Configurable discounts for lapsed subscribers
- Offer code redemption: Support App Store offer codes
- Family Sharing: Proper handling if enabled

### 4. Purchase Flow
- Display subscription options with localized pricing (use Product.displayPrice)
- Show trial eligibility status (Product.subscription.isEligibleForIntroOffer)
- Handle purchase confirmation with biometric/password authentication
- Process successful purchases and grant entitlements immediately
- Handle purchase failures gracefully with user-friendly messages
- Support deferred purchases (Ask to Buy for Family Sharing)

### 5. Restore Purchases
- Implement "Restore Purchases" button in settings/paywall
- Use Transaction.currentEntitlements for efficient restoration
- Handle cases where no purchases exist to restore
- Sync restored purchases with backend
- Required by App Store Review Guidelines

### 6. Server-Side Validation (Backend)
- Implement App Store Server API integration (not deprecated verifyReceipt)
- Use App Store Server Notifications V2 for real-time status updates:
  - SUBSCRIBED, DID_RENEW, DID_CHANGE_RENEWAL_STATUS
  - DID_FAIL_TO_RENEW, EXPIRED, REFUND, REVOKE
  - OFFER_REDEEMED, GRACE_PERIOD_EXPIRED
- Store subscription status in database with proper schema
- Handle billing retry state and grace periods
- Implement JWT-based authentication for App Store Server API

### 7. Entitlement Management
- Create EntitlementService for checking subscription status
- Cache entitlements locally with secure storage
- Sync entitlements on app launch and purchase events
- Handle offline entitlement checking gracefully
- Implement feature flags based on subscription tier

### 8. Database Schema (Prisma)
```prisma
model Subscription {
  id                    String   @id @default(cuid())
  userId                String   @unique
  user                  User     @relation(fields: [userId], references: [id])
  productId             String   // e.g., "com.nutri.pro.monthly"
  originalTransactionId String   @unique
  status                SubscriptionStatus
  expiresAt             DateTime
  isTrialPeriod         Boolean  @default(false)
  isIntroOfferPeriod    Boolean  @default(false)
  autoRenewEnabled      Boolean  @default(true)
  gracePeriodExpiresAt  DateTime?
  billingRetryPeriod    Boolean  @default(false)
  priceLocale           String?
  priceCurrency         String?
  priceAmount           Decimal?
  environment           String   // "sandbox" or "production"
  createdAt             DateTime @default(now())
  updatedAt             DateTime @updatedAt
  
  @@index([userId])
  @@index([originalTransactionId])
  @@index([status, expiresAt])
}

enum SubscriptionStatus {
  ACTIVE
  EXPIRED
  IN_GRACE_PERIOD
  IN_BILLING_RETRY
  REVOKED
  REFUNDED
}

model SubscriptionEvent {
  id                    String   @id @default(cuid())
  subscriptionId        String
  subscription          Subscription @relation(fields: [subscriptionId], references: [id])
  notificationType      String   // App Store notification type
  subtype               String?
  transactionId         String
  originalTransactionId String
  eventData             Json
  processedAt           DateTime @default(now())
  
  @@index([subscriptionId])
  @@index([originalTransactionId])
}
```

### 9. UI Components
- **PaywallScreen**: Full-screen subscription offering
  - Feature comparison between tiers
  - Localized pricing with trial info
  - Terms of Service and Privacy Policy links (required)
  - Restore Purchases button
- **SubscriptionBadge**: Show Pro status in UI
- **UpgradePrompt**: Contextual upgrade nudges
- **ManageSubscriptionScreen**: View current plan, link to App Store management

### 10. Security Considerations
- Never trust client-side purchase verification alone
- Always validate with App Store Server API
- Use App Store Server Notifications for authoritative status
- Protect webhook endpoint with signature verification
- Store sensitive data (transaction IDs) securely
- Implement proper error handling to prevent purchase fraud

### 11. Testing Strategy
- Use StoreKit Testing in Xcode for local testing
- Test in App Store Sandbox environment
- Test all subscription states:
  - New subscription, renewal, expiration
  - Trial to paid conversion
  - Upgrade/downgrade between tiers
  - Cancellation and re-subscription
  - Billing issues and grace period
  - Refunds and revocation
- Test Family Sharing scenarios if enabled
- Test Ask to Buy (deferred transactions)
- Test restore purchases flow

### 12. Analytics & Monitoring
- Track subscription events for analytics
- Monitor conversion rates (trial to paid)
- Track churn and retention metrics
- Alert on unusual refund patterns
- Log all webhook processing for debugging

### 13. App Store Review Compliance
- Include "Restore Purchases" functionality
- Display subscription terms clearly
- Link to Terms of Service and Privacy Policy
- Handle all edge cases gracefully
- Proper error messages for users

## Dependencies
- Requires App Store Connect configuration
- Requires Apple Developer Program membership
- Backend webhook endpoint for notifications
- Expo development build (not Expo Go) for IAP testing

## Files to Create/Modify

### Mobile (React Native/Expo)
- `lib/services/purchases/` - Purchase service directory
  - `index.ts` - Main purchase service
  - `types.ts` - TypeScript interfaces
  - `products.ts` - Product ID constants
  - `entitlements.ts` - Entitlement checking
- `lib/context/SubscriptionContext.tsx` - React context for subscription state
- `lib/hooks/useSubscription.ts` - Hook for subscription status
- `lib/hooks/usePurchases.ts` - Hook for purchase operations
- `app/paywall.tsx` - Paywall screen
- `app/subscription.tsx` - Manage subscription screen
- `lib/components/PaywallCard.tsx` - Subscription option card
- `lib/components/SubscriptionBadge.tsx` - Pro badge component
- `lib/components/UpgradePrompt.tsx` - Upgrade nudge component
- `lib/api/subscriptions.ts` - API client for subscription endpoints

### Backend (Express/Node.js)
- `server/src/services/subscriptionService.ts` - Subscription business logic
- `server/src/services/appStoreService.ts` - App Store Server API integration
- `server/src/controllers/subscriptionController.ts` - Subscription endpoints
- `server/src/controllers/webhookController.ts` - App Store webhook handler
- `server/src/routes/subscription.ts` - Subscription routes
- `server/src/routes/webhook.ts` - Webhook routes
- `server/src/validation/subscriptionSchemas.ts` - Zod schemas
- `server/src/middleware/webhookAuth.ts` - Webhook signature verification
- `server/prisma/schema.prisma` - Add Subscription models

### Configuration
- `app.json` - Add In-App Purchase capability
- App Store Connect - Configure products and subscriptions

## Acceptance Criteria
1. Users can view and purchase subscription plans
2. Trial periods work correctly with proper eligibility checking
3. Subscriptions renew automatically and status syncs to backend
4. Users can restore purchases on new devices
5. Server validates all purchases before granting access
6. Webhook processes all App Store notifications correctly
7. Entitlements are checked efficiently with proper caching
8. All edge cases (expiration, billing issues, refunds) handled
9. Analytics track key subscription metrics
10. Passes App Store review guidelines

**Test Strategy:**

## Testing Strategy

### Unit Tests
1. **Purchase Service Tests**
   - Product fetching and caching
   - Purchase flow state management
   - Entitlement calculation logic
   - Error handling for various failure modes

2. **Backend Service Tests**
   - App Store Server API mock responses
   - Subscription status transitions
   - Webhook payload parsing and validation
   - JWT token generation and validation

3. **Entitlement Logic Tests**
   - Active subscription detection
   - Grace period handling
   - Trial eligibility checking
   - Feature flag resolution

### Integration Tests
1. **StoreKit Sandbox Testing**
   - Full purchase flow with sandbox accounts
   - Subscription renewal simulation
   - Trial to paid conversion
   - Restore purchases flow

2. **Backend Webhook Tests**
   - Webhook signature verification
   - Event processing for all notification types
   - Database state updates
   - Error recovery scenarios

3. **End-to-End Tests**
   - Complete purchase journey
   - Entitlement sync after purchase
   - Cross-device restore
   - Subscription management

### Manual Testing Checklist
- [ ] Purchase monthly subscription
- [ ] Purchase yearly subscription
- [ ] Start free trial
- [ ] Trial expires and converts to paid
- [ ] Cancel subscription
- [ ] Re-subscribe after cancellation
- [ ] Restore purchases on new device
- [ ] Upgrade from monthly to yearly
- [ ] Downgrade from yearly to monthly
- [ ] Handle billing issue/grace period
- [ ] Refund processing
- [ ] Family Sharing (if enabled)
- [ ] Ask to Buy deferred purchase
- [ ] Offer code redemption
- [ ] Promotional offer application

## Subtasks

### 38.1. Set up react-native-iap library and StoreKit 2 integration

**Status:** pending  
**Dependencies:** None  

Install react-native-iap v12+ with StoreKit 2 support, configure Expo development build for IAP capabilities, implement transaction listener with async/await patterns, and set up proper transaction finishing to prevent duplicate charges.

**Details:**

Install react-native-iap (v12+) which supports StoreKit 2. Configure app.json with In-App Purchase capability and create Expo development build (IAP doesn't work in Expo Go). Create lib/services/purchases/index.ts with purchase service initialization, implement Transaction.updates listener for real-time updates, set up proper transaction finishing logic, configure sandbox vs production environment detection, and implement error handling for common StoreKit errors (user canceled, network issues, etc.).

### 38.2. Configure App Store Connect products and subscription groups

**Status:** pending  
**Dependencies:** 38.1  

Set up subscription products in App Store Connect (com.nutri.pro.monthly, com.nutri.pro.yearly), configure subscription groups for upgrade/downgrade paths, and set up introductory offers and promotional offers.

**Details:**

In App Store Connect, create subscription group 'Nutri Pro Subscriptions'. Add two products: com.nutri.pro.monthly ($9.99/month) and com.nutri.pro.yearly ($99.99/year, ~17% discount). Configure subscription group settings to allow upgrades/downgrades. Set up introductory offer: 7-day free trial for both tiers. Create promotional offers for win-back campaigns (e.g., 50% off for 3 months). Generate offer codes for marketing. Document all product IDs in lib/services/purchases/products.ts as constants.

### 38.3. Implement trial period and promotional offer eligibility checking

**Status:** pending  
**Dependencies:** 38.2  

Implement logic to check user eligibility for free trials and promotional offers using StoreKit 2's Product.subscription.isEligibleForIntroOffer API, and display trial information in UI.

**Details:**

Create lib/services/purchases/entitlements.ts with functions to check trial eligibility using Product.subscription.isEligibleForIntroOffer (StoreKit 2 API). Implement promotional offer eligibility checking based on subscription history. Create TrialEligibilityChecker class that caches eligibility status to avoid repeated API calls. Add logic to display trial information ('7-day free trial, then $9.99/month') when eligible, or regular pricing when not eligible. Handle edge cases: never subscribed (eligible), currently subscribed (not eligible), lapsed subscriber (promotional offer eligible).

### 38.4. Build PaywallScreen with subscription options and purchase flow

**Status:** pending  
**Dependencies:** 38.3  

Create full-screen paywall UI (app/paywall.tsx) displaying subscription tiers with localized pricing, trial eligibility, feature comparison, and purchase buttons with biometric authentication.

**Details:**

Create app/paywall.tsx as modal screen with: (1) Feature comparison table (Free vs Pro), (2) Subscription option cards showing Product.displayPrice (localized), trial information if eligible, and prominent CTA button, (3) Legal footer with Terms of Service and Privacy Policy links (required by App Store), (4) Restore Purchases button, (5) Loading states during purchase, (6) Success/error handling with user-friendly messages. Implement purchase flow: user taps Subscribe → StoreKit shows confirmation with biometric/password → process transaction → grant entitlements → dismiss paywall. Handle deferred transactions (Ask to Buy). Create lib/components/PaywallCard.tsx for reusable subscription option cards.

### 38.5. Build ManageSubscriptionScreen and SubscriptionBadge component

**Status:** pending  
**Dependencies:** 38.4  

Create subscription management screen (app/subscription.tsx) showing current plan details and App Store management link, plus a Pro badge component for displaying subscription status throughout the app.

**Details:**

Create app/subscription.tsx displaying: current subscription tier, renewal date, pricing, trial/intro offer status, auto-renew toggle status (read-only, managed via App Store), 'Manage Subscription' button linking to App Store subscription management (using Linking.openURL with App Store subscription URL), and Restore Purchases button. Create lib/components/SubscriptionBadge.tsx showing 'Pro' badge with styling when user has active subscription. Create lib/components/UpgradePrompt.tsx for contextual upgrade nudges when free users try premium features. Add subscription status to profile screen.

### 38.6. Implement Restore Purchases functionality

**Status:** pending  
**Dependencies:** 38.5  

Implement restore purchases flow using Transaction.currentEntitlements to sync past purchases and validate them with backend, required by App Store Review Guidelines.

**Details:**

In lib/services/purchases/index.ts, implement restorePurchases() function using Transaction.currentEntitlements (StoreKit 2 efficient API, not deprecated restoreCompletedTransactions). Flow: (1) User taps 'Restore Purchases', (2) Show loading indicator, (3) Fetch currentEntitlements from StoreKit, (4) For each entitlement, extract originalTransactionId and send to backend for validation, (5) Backend validates with App Store Server API, (6) Sync subscription status to local state, (7) Show success message ('Subscription restored') or info message ('No purchases to restore'). Handle errors gracefully. Add 'Restore Purchases' button to both PaywallScreen and ManageSubscriptionScreen.

### 38.7. Implement backend App Store Server API integration for receipt validation

**Status:** pending  
**Dependencies:** 38.6  

Create backend service to validate purchases using the modern App Store Server API (NOT deprecated verifyReceipt), implement JWT-based authentication, and query transaction/subscription status.

**Details:**

Create server/src/services/appStoreService.ts implementing: (1) App Store Server API client using JWT authentication (generate JWT with private key from App Store Connect), (2) getTransactionInfo(transactionId) endpoint to fetch transaction details, (3) getSubscriptionStatus(originalTransactionId) endpoint to check current subscription state, (4) validateTransaction() function that checks signature and decodes JWSTransaction, (5) Environment detection (sandbox vs production, use different API endpoints). Store App Store Connect API credentials in environment variables (KEY_ID, ISSUER_ID, PRIVATE_KEY). Never use deprecated verifyReceipt endpoint. Create server/src/validation/subscriptionSchemas.ts with Zod schemas for validation.

### 38.8. Implement App Store Server Notifications V2 webhook with signature verification

**Status:** pending  
**Dependencies:** 38.7  

Create webhook endpoint to receive real-time subscription status updates from Apple (renewals, cancellations, refunds), implement cryptographic signature verification, and process all notification types.

**Details:**

Create server/src/controllers/webhookController.ts with POST /api/webhooks/app-store endpoint. Implement signature verification using Apple's public key (fetch from Apple's JWKS endpoint, verify JWS signature). Parse notification payload (signedPayload is JWS, decode and verify). Handle all App Store Server Notifications V2 types: SUBSCRIBED, DID_RENEW, DID_CHANGE_RENEWAL_STATUS, DID_FAIL_TO_RENEW, EXPIRED, REFUND, REVOKE, OFFER_REDEEMED, GRACE_PERIOD_EXPIRED. For each notification, update Subscription record in database and create SubscriptionEvent audit log. Implement idempotency using notification UUID. Create server/src/middleware/webhookAuth.ts for signature verification middleware. Configure webhook URL in App Store Connect.

### 38.9. Create Prisma database schema for Subscription and SubscriptionEvent models

**Status:** pending  
**Dependencies:** 38.8  

Add Subscription and SubscriptionEvent models to Prisma schema with all required fields for subscription lifecycle tracking, run migrations, and generate Prisma client.

**Details:**

Add to server/prisma/schema.prisma: (1) Subscription model with fields: id, userId (unique), productId, originalTransactionId (unique), status (enum: ACTIVE, EXPIRED, IN_GRACE_PERIOD, IN_BILLING_RETRY, REVOKED, REFUNDED), expiresAt, isTrialPeriod, isIntroOfferPeriod, autoRenewEnabled, gracePeriodExpiresAt, billingRetryPeriod, priceLocale, priceCurrency, priceAmount (Decimal), environment (sandbox/production), timestamps. (2) SubscriptionEvent model with fields: id, subscriptionId, notificationType, subtype, transactionId, originalTransactionId, eventData (Json), processedAt. Add indexes for efficient queries: userId, originalTransactionId, status+expiresAt composite. Add relation to User model (user.subscription). Run 'npm run db:generate' and 'npm run db:migrate' to apply schema changes.

### 38.10. Implement entitlement service with secure caching and offline support

**Status:** pending  
**Dependencies:** 38.9  

Create EntitlementService to check subscription status efficiently with secure local caching, sync entitlements on app launch and purchase events, handle offline scenarios, and implement feature flags based on subscription tier.

**Details:**

Create lib/services/purchases/entitlements.ts with EntitlementService class: (1) checkEntitlement(feature) returns boolean based on subscription status, (2) syncEntitlements() fetches current status from backend and caches in Expo SecureStore, (3) getCachedEntitlements() reads from SecureStore for offline support (with staleness check), (4) Feature flags: UNLIMITED_HISTORY, ML_INSIGHTS, ADVANCED_ANALYTICS mapped to subscription tiers. Create lib/context/SubscriptionContext.tsx providing subscription status to entire app. Create lib/hooks/useSubscription.ts hook returning { isPro, isTrial, expiresAt, syncEntitlements }. Call syncEntitlements() on app launch, after successful purchase, and on restore. Implement cache expiration (refresh every 24 hours). Handle offline mode: use cached entitlements if < 7 days old.

### 38.11. Set up StoreKit Testing configuration and Sandbox testing flows

**Status:** pending  
**Dependencies:** 38.10  

Configure StoreKit Testing in Xcode for local testing without App Store, create Sandbox test accounts, and test all subscription lifecycle scenarios (new subscription, renewal, trial conversion, upgrade/downgrade, cancellation, billing issues, refunds).

**Details:**

Configure StoreKit Configuration file (.storekit) in Xcode with test products matching App Store Connect (com.nutri.pro.monthly, com.nutri.pro.yearly). Set up subscription durations (monthly = 5 minutes, yearly = 1 hour for faster testing). Create Sandbox test accounts in App Store Connect with different regions (US, UK, EU for currency testing). Test scenarios: (1) New subscription with trial → conversion to paid, (2) Successful renewal, (3) Expiration after cancellation, (4) Upgrade from monthly to yearly, (5) Downgrade from yearly to monthly, (6) Billing failure → grace period → billing retry, (7) Refund processing, (8) Family Sharing if enabled, (9) Ask to Buy (deferred transactions), (10) Restore purchases on new device. Document testing checklist in ml-service or server README.

### 38.12. Integrate subscription analytics and ensure App Store Review compliance

**Status:** pending  
**Dependencies:** 38.11  

Add analytics tracking for subscription events (impressions, conversions, churn), implement monitoring for refund patterns, and ensure full compliance with App Store Review Guidelines (restore purchases, terms display, error handling).

**Details:**

Analytics integration: Track events (paywall_viewed, subscription_started, trial_started, trial_converted, subscription_renewed, subscription_canceled, subscription_expired, purchase_restored, upgrade_completed, downgrade_completed) using existing analytics service or add new one (e.g., Mixpanel, Amplitude). Create server/src/services/subscriptionAnalyticsService.ts to calculate metrics: conversion rate (trial → paid), churn rate, MRR (Monthly Recurring Revenue), LTV (Lifetime Value). Set up alerts for unusual refund patterns (>5% refund rate). App Store compliance checklist: (1) Restore Purchases button visible and functional, (2) Terms of Service and Privacy Policy links on paywall, (3) Clear subscription terms display (price, duration, auto-renewal), (4) Graceful error handling with user-friendly messages, (5) No misleading marketing claims. Add subscription_tier field to User model for feature flag checks.
