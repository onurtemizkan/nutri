# StoreKit Testing Guide

## Overview

This guide covers testing in-app purchases for Nutri Pro subscriptions using both StoreKit Testing (Xcode) and App Store Sandbox.

## StoreKit Testing Configuration

### Setup

1. Open the project in Xcode
2. Select Product → Scheme → Edit Scheme
3. Under Run → Options, set "StoreKit Configuration" to `NutriProducts.storekit`
4. Build and run

### Products Configured

| Product ID          | Name    | Price       | Trial            |
| ------------------- | ------- | ----------- | ---------------- |
| `nutri.pro.monthly` | Monthly | $9.99/month | 7-day free trial |
| `nutri.pro.yearly`  | Yearly  | $79.99/year | 7-day free trial |

### Testing Time Scaling

In StoreKit Testing, subscription periods are accelerated:

| Real Period | Test Duration |
| ----------- | ------------- |
| 1 week      | 3 minutes     |
| 1 month     | 5 minutes     |
| 3 months    | 15 minutes    |
| 6 months    | 30 minutes    |
| 1 year      | 1 hour        |

## Sandbox Testing Setup

### Creating Sandbox Test Accounts

1. Go to [App Store Connect](https://appstoreconnect.apple.com)
2. Navigate to Users and Access → Sandbox → Testers
3. Create test accounts for different scenarios:

| Account Purpose   | Email                           | Region         |
| ----------------- | ------------------------------- | -------------- |
| New user (US)     | `nutri.test.us@example.com`     | United States  |
| New user (UK)     | `nutri.test.uk@example.com`     | United Kingdom |
| New user (EU)     | `nutri.test.eu@example.com`     | Germany        |
| Lapsed subscriber | `nutri.test.lapsed@example.com` | United States  |
| Family sharing    | `nutri.test.family@example.com` | United States  |

### Using Sandbox on Device

1. On iPhone, go to Settings → App Store → Sandbox Account
2. Sign in with sandbox test account
3. Sandbox transactions will use accelerated time:
   - Monthly = 5 minutes
   - Yearly = 1 hour

## Testing Scenarios Checklist

### 1. New Subscription with Trial

- [ ] Launch app as new user
- [ ] Navigate to paywall
- [ ] Verify 7-day free trial is displayed
- [ ] Start trial (monthly)
- [ ] Verify Pro features unlocked immediately
- [ ] Verify trial badge shown in subscription screen
- [ ] Wait 3 minutes (trial expiry in test mode)
- [ ] Verify conversion to paid subscription
- [ ] Verify no interruption in service

### 2. Successful Renewal

- [ ] With active subscription
- [ ] Wait for renewal period (5 minutes for monthly)
- [ ] Verify renewal notification received (if webhooks configured)
- [ ] Verify expiresAt updated
- [ ] Verify no service interruption

### 3. Expiration After Cancellation

- [ ] With active subscription
- [ ] Cancel via Apple Settings
- [ ] Verify auto-renew disabled in app
- [ ] Verify subscription remains active until expiresAt
- [ ] Wait for expiration
- [ ] Verify features locked after expiration
- [ ] Verify free tier shown

### 4. Upgrade (Monthly → Yearly)

- [ ] With active monthly subscription
- [ ] Navigate to subscription management
- [ ] Tap upgrade to yearly
- [ ] Verify proration message
- [ ] Complete upgrade
- [ ] Verify immediate access to yearly benefits
- [ ] Verify new expiration date (1 year from upgrade)

### 5. Downgrade (Yearly → Monthly)

- [ ] With active yearly subscription
- [ ] Navigate to subscription management
- [ ] Tap downgrade to monthly
- [ ] Verify downgrade takes effect at renewal
- [ ] Verify current access maintained until expiry
- [ ] After expiry, verify monthly subscription active

### 6. Billing Failure → Grace Period → Billing Retry

- [ ] In StoreKit Testing, enable billing failure
- [ ] Wait for renewal attempt
- [ ] Verify grace period state
- [ ] Verify access maintained during grace period
- [ ] Verify webhook notification received
- [ ] Simulate successful billing retry
- [ ] Verify restoration to active state

### 7. Refund Processing

- [ ] With active subscription
- [ ] Request refund via Report a Problem
- [ ] Or simulate refund in StoreKit Testing
- [ ] Verify revoked state
- [ ] Verify Pro features locked
- [ ] Verify webhook notification received

### 8. Family Sharing (if enabled)

- [ ] Set up Family Sharing in sandbox
- [ ] Subscribe with organizer account
- [ ] On family member device, verify shared access
- [ ] Verify ownership type shows FAMILY_SHARED

### 9. Ask to Buy (Deferred Transactions)

- [ ] Set up Family Sharing with child account
- [ ] Attempt purchase with Ask to Buy enabled
- [ ] Verify deferred state handled
- [ ] Approve purchase from parent
- [ ] Verify subscription activated

### 10. Restore Purchases

- [ ] With existing subscription
- [ ] Sign out and sign in on new device
- [ ] Tap "Restore Purchases"
- [ ] Verify subscription restored
- [ ] Verify Pro features unlocked

## Webhook Testing

### Local Testing with ngrok

1. Install ngrok: `brew install ngrok`
2. Start tunnel: `ngrok http 3000`
3. Configure webhook URL in App Store Connect:
   - Use ngrok URL + `/api/webhooks/app-store`
4. Test with `requestTestNotification()` API

### Verifying Webhook Events

Check backend logs for:

- [ ] SUBSCRIBED notification on new subscription
- [ ] DID_RENEW on renewal
- [ ] DID_CHANGE_RENEWAL_STATUS on cancel/resubscribe
- [ ] EXPIRED after expiration
- [ ] REFUND on refund
- [ ] DID_FAIL_TO_RENEW on billing failure

## Currency Testing

Test with sandbox accounts from different regions:

- [ ] US account shows USD pricing
- [ ] UK account shows GBP pricing
- [ ] EU account shows EUR pricing
- [ ] Verify correct formatting (€9.99 vs $9.99 vs £8.99)

## Error Handling Testing

### StoreKit Testing Error Simulation

In Xcode, use Debug → StoreKit → Manage Transactions to:

- Fail purchase
- Fail verification
- Interrupt transaction

Verify app handles:

- [ ] Purchase failure gracefully
- [ ] Network errors with retry
- [ ] Verification failures
- [ ] User cancellation

## Debugging Tools

### StoreKit Transaction Manager

- Open: Debug → StoreKit → Manage Transactions
- View all transactions
- Delete transactions to reset state
- Simulate renewal/expiration

### Console Logs

Look for these log prefixes:

- `[Purchases]` - Purchase flow
- `[Entitlements]` - Entitlement checks
- `[SubscriptionContext]` - Context updates

## Pre-Release Checklist

Before submitting to App Store:

- [ ] Remove StoreKit Configuration from scheme
- [ ] Verify real product IDs in App Store Connect
- [ ] Verify webhook URL is production
- [ ] Test with production server
- [ ] Verify subscription pricing is correct
- [ ] Review App Store Review Guidelines for IAP

## Common Issues

### Transaction Stuck in "Purchasing"

- Reset StoreKit Testing environment
- Delete transaction in Transaction Manager
- Restart app

### Webhook Not Received

- Check ngrok is running
- Verify webhook URL in App Store Connect
- Check backend logs for errors
- Verify signature verification

### Trial Not Showing

- Check eligibility in StoreKit Testing
- Verify `isEligibleForIntroOfferIOS()` returns true
- Check product configuration

---

_Last updated: December 2025_
