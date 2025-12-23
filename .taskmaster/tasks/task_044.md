# Task ID: 44

**Title:** In-App Purchases & Subscription Management (StoreKit 2)

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement comprehensive in-app purchase system using StoreKit 2 for iOS with subscription management, including premium feature gating, purchase restoration, receipt validation, and admin panel subscription analytics.

**Details:**

No details provided.

**Test Strategy:**

No test strategy provided.

## Subtasks

### 44.1. Implement Backend Subscription Service with StoreKit 2 API Integration

**Status:** pending  
**Dependencies:** None  

Create comprehensive backend subscription service with App Store Server API v2 integration, JWS token validation, receipt verification, and subscription status management using production-ready security practices.

**Details:**

Create `server/src/services/subscriptionService.ts` with:

1. **App Store Server API v2 Client:**
   - Implement JWT signing for App Store Server API requests using ES256 algorithm
   - Configure API endpoints for production (https://api.storekit.itunes.apple.com) and sandbox (https://api.storekit-sandbox.itunes.apple.com)
   - Add methods: getTransactionInfo(), getSubscriptionStatuses(), getTransactionHistory()
   - Store App Store Connect API credentials securely in environment variables (APPLE_KEY_ID, APPLE_ISSUER_ID, APPLE_PRIVATE_KEY)

2. **JWS Token Validation:**
   - Implement signedPayload verification using Apple's public keys
   - Decode and validate JWS tokens from purchase receipts
   - Extract transaction data: originalTransactionId, productId, purchaseDate, expirationDate
   - Verify signature chain and certificate validity

3. **Subscription Management:**
   - Create purchaseSubscription(userId, transactionId, receipt): Validate receipt, update User model (subscriptionTier, subscriptionStartDate, subscriptionEndDate, subscriptionBillingCycle, subscriptionPrice)
   - Create restorePurchases(userId): Query App Store for active subscriptions, restore entitlements
   - Create getSubscriptionStatus(userId): Return current subscription state with isActive, daysRemaining, tier, billingCycle
   - Create syncSubscriptionStatus(transactionId): Fetch latest status from App Store Server API and sync to database

4. **Integration with Existing Infrastructure:**
   - Use existing User model fields (lines 39-44 in schema.prisma)
   - Follow patterns from `adminSubscriptionService.ts` for database operations
   - Add Zod validation schemas in `server/src/validation/schemas.ts`
   - Create controller in `server/src/controllers/subscriptionController.ts`
   - Add routes in `server/src/routes/subscription.ts`

5. **Error Handling:**
   - Handle App Store API errors (21000-21010 status codes)
   - Implement retry logic with exponential backoff
   - Log all subscription events for audit trail

6. **Environment Variables (.env):**
   - APPLE_KEY_ID, APPLE_ISSUER_ID, APPLE_PRIVATE_KEY (for App Store Server API)
   - APPLE_BUNDLE_ID (com.nutri.app)
   - APP_STORE_ENVIRONMENT (production/sandbox)

### 44.2. Implement App Store Server Notifications V2 Webhook Handler

**Status:** pending  
**Dependencies:** 44.1  

Build secure webhook endpoint to receive and process App Store Server Notifications V2 with signature verification, event processing, and retry logic for subscription lifecycle events.

**Details:**

Create webhook handler in `server/src/controllers/webhookController.ts` and `server/src/services/webhookService.ts`:

1. **Webhook Endpoint Setup:**
   - Create POST /api/webhooks/appstore endpoint
   - Disable rate limiting for Apple's servers (add IP allowlist: 17.0.0.0/8)
   - Add raw body parser for signature verification
   - Configure HTTPS-only access (App Store requirement)

2. **Signature Verification:**
   - Extract x-signature header from Apple's webhook requests
   - Verify signature using Apple's root certificates
   - Decode signedPayload JWS token
   - Validate certificate chain and timestamp
   - Reject requests with invalid signatures (return 401)

3. **Event Processing (Using Existing AppStoreWebhookEvent Model):**
   - Parse notificationType from payload (SUBSCRIBED, DID_RENEW, DID_CHANGE_RENEWAL_STATUS, DID_FAIL_TO_RENEW, EXPIRED, GRACE_PERIOD_EXPIRED, REFUND, etc.)
   - Store event in `AppStoreWebhookEvent` table (lines 663-697 in schema.prisma) with PENDING status
   - Process asynchronously to avoid blocking webhook response
   - Update status to SUCCESS/FAILED after processing

4. **Subscription Status Updates:**
   - SUBSCRIBED: Create/update subscription, set tier to PRO_TRIAL or PRO
   - DID_RENEW: Extend subscriptionEndDate based on renewal transaction
   - DID_CHANGE_RENEWAL_STATUS: Handle auto-renew enabled/disabled
   - EXPIRED/DID_FAIL_TO_RENEW: Set tier to FREE, mark subscription as ended
   - REFUND: Revoke subscription immediately, log for support team
   - GRACE_PERIOD_EXPIRED: Final subscription termination

5. **User Lookup:**
   - Use originalTransactionId to find user (may need to add transactionId field to User model or store in metadata JSON)
   - Update userId field in AppStoreWebhookEvent after successful lookup
   - Handle cases where user cannot be found (log warning, retry later)

6. **Retry Logic:**
   - Implement exponential backoff for failed event processing
   - Store errorMessage and retryCount in AppStoreWebhookEvent
   - Max retries: 5 attempts
   - Use existing adminWebhookService patterns for retry implementation

7. **Integration with Admin Panel:**
   - Existing webhook table/detail views already implemented (admin-panel/components/webhooks/)
   - Ensure webhook events are queryable via adminWebhookService

8. **Configure in App Store Connect:**
   - Set webhook URL: https://api.nutri.app/api/webhooks/appstore
   - Enable notification types: Subscriptions (version 2)
   - Test with sandbox environment first

### 44.3. Implement Mobile IAP with react-native-iap (StoreKit 2 Support)

**Status:** pending  
**Dependencies:** 44.1  

Integrate react-native-iap v13+ for iOS in-app purchases with StoreKit 2 support, including product fetching, purchase flow, purchase restoration, and transaction handling with proper error management.

**Details:**

Create mobile IAP implementation in `lib/services/iap/`:

1. **Install and Configure react-native-iap:**
   - Install: `npm install react-native-iap@^13.0.0`
   - Configure expo-build-properties in app.json for iOS deployment target 13.0+
   - Add StoreKit configuration file for local testing (Configuration.storekit)
   - Configure product IDs in constants: 'nutri_pro_monthly', 'nutri_pro_annual'

2. **Create IAP Service (`lib/services/iap/IAPService.ts`):**
   - Initialize IAP connection: initConnection(), endConnection()
   - Fetch products: getProducts(['nutri_pro_monthly', 'nutri_pro_annual'])
   - Purchase flow: requestPurchase(productId), finishTransaction(purchase)
   - Restore purchases: getAvailablePurchases()
   - Get purchase history: getPurchaseHistory()

3. **Purchase Flow:**
   - Call requestPurchase(productId) -> returns Purchase object with transactionReceipt
   - Send receipt to backend: POST /api/subscriptions/verify with { transactionId, receipt }
   - Backend validates with App Store Server API
   - On success: Update local user state with new subscription tier
   - Call finishTransaction() to complete purchase (CRITICAL: prevents refunds)
   - Handle errors: User canceled, network errors, already purchased, etc.

4. **Purchase Restoration:**
   - Implement restorePurchases(): Call getAvailablePurchases()
   - Send all valid receipts to backend for verification
   - Update UI with restored subscription status
   - Show success/error messages to user

5. **Transaction Listener:**
   - Set up purchaseUpdatedListener for pending transactions
   - Handle transactions on app startup (clear pending transactions)
   - Process queued transactions automatically
   - Ensure all transactions are finished to avoid stuck purchases

6. **Error Handling:**
   - Handle StoreKit errors: E_USER_CANCELLED, E_NETWORK_ERROR, E_SERVICE_ERROR
   - Display user-friendly error messages
   - Retry logic for network failures
   - Fallback to manual restoration if automatic fails

7. **Context/Hook (`lib/context/SubscriptionContext.tsx` and `lib/hooks/useSubscription.ts`):**
   - Provide subscription state: tier, isActive, expiresAt, billingCycle
   - Methods: purchase(productId), restore(), checkStatus()
   - Sync with backend subscription status on app launch
   - Cache subscription status locally with SecureStore

8. **Testing Configuration:**
   - Configure Sandbox tester accounts in App Store Connect
   - Create test products with various pricing
   - Test interrupted purchases (background app, network loss)

9. **Expo Configuration (app.json):**
   - Add expo-build-properties for iOS 13.0+ target
   - Configure bundle ID to match App Store Connect
   - Add in-app purchase capability to iOS entitlements

### 44.4. Build Paywall and Subscription Management UI

**Status:** pending  
**Dependencies:** 44.3  

Create beautiful, conversion-optimized paywall UI with feature comparison, pricing plans, trial information, and subscription management screens following iOS design guidelines and App Store requirements.

**Details:**

Create UI components for subscription management:

1. **Paywall Screen (`app/paywall.tsx`):**
   - Hero section with app value proposition
   - Feature comparison table: Free vs Pro
   - Pricing cards with monthly/annual toggle
   - Highlight savings for annual plan (e.g., "Save 2 months")
   - Show trial period prominently (if applicable)
   - Clear CTA buttons: "Start Free Trial" or "Subscribe Now"
   - Terms of service link (required by App Store)
   - Privacy policy link (required by App Store)
   - Restore purchases button (required by App Store)

2. **Feature Comparison:**
   Free tier:
   - Basic meal tracking (last 30 days)
   - Manual entry only
   - Basic calorie/macro tracking
   - Limited health metric history (7 days)
   
   Pro tier:
   - Unlimited meal history
   - AI-powered food recognition (barcode + image scanning)
   - Advanced micronutrient tracking
   - ML health predictions (RHR, HRV)
   - Unlimited health metric history
   - Nutrition-health correlations
   - Supplement tracking
   - Priority support

3. **Pricing Plans Component:**
   - Fetch products from IAP service: useProducts()
   - Display localized prices from StoreKit
   - Show billing cycle (per month / per year)
   - Highlight "Best Value" for annual plan
   - Show per-month cost for annual plan (e.g., "$4.99/month, billed annually")
   - Loading states while fetching products
   - Error states if products fail to load

4. **Purchase Flow UI:**
   - Show loading indicator during purchase
   - Success modal with confetti/celebration animation
   - Error handling with retry option
   - Handle edge cases: already subscribed, restore needed, etc.

5. **Subscription Management Screen (`app/(tabs)/profile.tsx` -> Manage Subscription):**
   - Display current subscription tier and status
   - Show expiration date and auto-renew status
   - Show billing cycle (monthly/annual)
   - Show price paid
   - Button to manage subscription in App Store (deep link)
   - Button to restore purchases
   - Button to cancel subscription (deep link to App Store)
   - Show subscription benefits/features

6. **Subscription Status Badge (Profile Tab):**
   - Show "Free", "Pro Trial", or "Pro" badge
   - Show days remaining for trial/subscription
   - Visual indicator for expiring soon (<7 days)

7. **Restore Purchases UI:**
   - Modal or screen for restoration flow
   - Loading state during restoration
   - Success message: "Subscription restored!"
   - Error message: "No purchases found" or specific error

8. **App Store Requirements:**
   - Include terms link: https://nutri.app/terms
   - Include privacy link: https://nutri.app/privacy
   - Show auto-renewable subscription terms
   - Include restore purchases option
   - No alternative payment methods shown

9. **Accessibility:**
   - All interactive elements have accessibilityLabel
   - Support Dynamic Type for text scaling
   - VoiceOver friendly navigation
   - Sufficient color contrast

10. **Responsive Design:**
    - Support iPhone (all sizes) and iPad
    - Landscape and portrait orientations
    - Use existing responsive utilities from lib/responsive/

11. **Analytics Events (Future):**
    - Track paywall views
    - Track purchase attempts
    - Track successful purchases
    - Track restoration attempts

### 44.5. Implement Premium Feature Gating System

**Status:** pending  
**Dependencies:** 44.3, 44.4  

Create comprehensive feature gating system to restrict premium features to Pro subscribers, including middleware, UI components, and graceful upgrade prompts throughout the app.

**Details:**

Implement feature gating across the app:

1. **Subscription Context Enhancement (`lib/context/SubscriptionContext.tsx`):**
   - Add subscription state to context: tier, isActive, expiresAt, billingCycle
   - Add methods: isPro(), isFree(), isTrialing(), daysRemaining()
   - Sync subscription status from backend on app launch
   - Refresh subscription status periodically (every 24 hours)
   - Store locally with SecureStore for offline access

2. **useSubscription Hook (`lib/hooks/useSubscription.ts`):**
   - Access subscription state: `const { isPro, tier, isActive } = useSubscription()`
   - Feature-specific checks: canAccessFeature('ml_predictions'), canAccessFeature('barcode_scan')
   - Return upgrade prompt function: showUpgradePrompt(feature)

3. **Feature Gate Component (`lib/components/FeatureGate.tsx`):**
   ```tsx
   <FeatureGate feature="ml_predictions" fallback={<UpgradePrompt />}>
     <MLPredictionsView />
   </FeatureGate>
   ```
   - Show upgrade prompt if user doesn't have access
   - Optionally show locked state with blur/overlay
   - Track feature gate impressions for analytics

4. **Upgrade Prompt Component (`lib/components/UpgradePrompt.tsx`):**
   - Modal or inline prompt explaining premium feature
   - Show specific benefits for the gated feature
   - CTA button: "Upgrade to Pro"
   - Navigate to paywall on CTA press
   - Option to dismiss and continue as Free user

5. **Feature Definitions (`lib/constants/features.ts`):**
   ```ts
   export const FEATURES = {
     ML_PREDICTIONS: { tier: 'PRO', name: 'Health Predictions' },
     BARCODE_SCAN: { tier: 'PRO', name: 'Barcode Scanning' },
     IMAGE_RECOGNITION: { tier: 'PRO', name: 'AI Food Recognition' },
     UNLIMITED_HISTORY: { tier: 'PRO', name: 'Unlimited History' },
     ADVANCED_ANALYTICS: { tier: 'PRO', name: 'Advanced Analytics' },
     SUPPLEMENT_TRACKING: { tier: 'PRO', name: 'Supplement Tracking' },
   }
   ```

6. **Implement Feature Gates in Screens:**
   - **Barcode Scanner (`app/scan-barcode.tsx`):** Gate entire screen, show upgrade prompt
   - **Food Image Recognition (`app/scan-food.tsx`):** Gate entire screen
   - **ML Predictions (Dashboard):** Show locked state with upgrade CTA
   - **Meal History:** Limit to 30 days for Free, show upgrade banner
   - **Health Metrics History:** Limit to 7 days for Free
   - **Supplement Tracking (`app/supplements.tsx`):** Gate entire feature
   - **Advanced Analytics:** Gate correlation charts and insights

7. **Backend Enforcement (Critical!):**
   - Add middleware: `server/src/middleware/subscriptionGuard.ts`
   - Check subscription tier before processing premium API requests
   - Return 402 Payment Required for gated features
   - Add tier check to relevant endpoints:
     - POST /api/food-analysis (image/barcode scanning)
     - GET /api/ml/predictions
     - GET /api/meals (limit results for Free tier)
     - GET /api/health-metrics (limit results for Free tier)
     - POST /api/supplements

8. **Data Retention for Free Users:**
   - Implement automatic data truncation for Free tier
   - Keep last 30 days of meals, 7 days of health metrics
   - Run cleanup job daily (cron job or scheduled task)
   - Preserve data if user upgrades (don't delete old data)

9. **Graceful Degradation:**
   - Show preview of premium features with blur/lock icon
   - Allow users to see what they're missing
   - Provide context-specific upgrade messaging
   - Never crash or show errors for gated features

10. **Trial Period Handling:**
    - PRO_TRIAL tier has full Pro access
    - Show trial expiration countdown in UI
    - Prompt to subscribe before trial ends
    - Grace period after trial expiration (optional)

11. **Subscription Expiration:**
    - Check subscriptionEndDate on every request
    - Downgrade to FREE tier if expired
    - Show renewal prompt in UI
    - Allow grace period for failed renewals (optional)

12. **Offline Handling:**
    - Cache subscription status locally
    - Allow offline access based on cached status
    - Sync on next online session
    - Handle edge case: subscription expired while offline
