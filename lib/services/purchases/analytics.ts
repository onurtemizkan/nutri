/**
 * Purchase Analytics
 *
 * Tracks subscription-related events for analytics.
 * Integrates with your analytics provider (Mixpanel, Amplitude, etc.)
 */

/**
 * Subscription analytics event types
 */
export type SubscriptionAnalyticsEvent =
  | 'paywall_viewed'
  | 'paywall_dismissed'
  | 'subscription_started'
  | 'trial_started'
  | 'trial_converted'
  | 'subscription_renewed'
  | 'subscription_canceled'
  | 'subscription_expired'
  | 'purchase_restored'
  | 'upgrade_completed'
  | 'downgrade_completed'
  | 'purchase_failed'
  | 'purchase_canceled_by_user';

/**
 * Event properties for analytics
 */
interface EventProperties {
  productId?: string;
  price?: number;
  currency?: string;
  source?: string;
  error?: string;
  trialDays?: number;
  previousProductId?: string;
  [key: string]: unknown;
}

/**
 * Track a subscription analytics event
 *
 * @param event - The event type
 * @param properties - Additional properties
 */
export function trackSubscriptionEvent(
  event: SubscriptionAnalyticsEvent,
  properties?: EventProperties
): void {
  console.log('[Analytics] Subscription event:', event, properties);

  // TODO: Integrate with your analytics provider
  // Examples:

  // Mixpanel:
  // mixpanel.track(event, properties);

  // Amplitude:
  // amplitude.logEvent(event, properties);

  // Segment:
  // analytics.track(event, properties);

  // Firebase Analytics:
  // analytics().logEvent(event, properties);

  // For now, we just log to console in development
  if (__DEV__) {
    console.log(`[Analytics] ${event}`, properties);
  }
}

/**
 * Track paywall viewed
 */
export function trackPaywallViewed(source?: string): void {
  trackSubscriptionEvent('paywall_viewed', { source });
}

/**
 * Track paywall dismissed without purchase
 */
export function trackPaywallDismissed(source?: string): void {
  trackSubscriptionEvent('paywall_dismissed', { source });
}

/**
 * Track subscription started (including trial)
 */
export function trackSubscriptionStarted(
  productId: string,
  price: number,
  currency: string,
  isTrial: boolean,
  trialDays?: number
): void {
  if (isTrial) {
    trackSubscriptionEvent('trial_started', {
      productId,
      price,
      currency,
      trialDays,
    });
  } else {
    trackSubscriptionEvent('subscription_started', {
      productId,
      price,
      currency,
    });
  }
}

/**
 * Track trial conversion to paid
 */
export function trackTrialConverted(productId: string, price: number, currency: string): void {
  trackSubscriptionEvent('trial_converted', {
    productId,
    price,
    currency,
  });
}

/**
 * Track purchase failure
 */
export function trackPurchaseFailed(productId: string, error: string, userCanceled: boolean): void {
  if (userCanceled) {
    trackSubscriptionEvent('purchase_canceled_by_user', {
      productId,
    });
  } else {
    trackSubscriptionEvent('purchase_failed', {
      productId,
      error,
    });
  }
}

/**
 * Track restore purchases
 */
export function trackPurchaseRestored(productId?: string): void {
  trackSubscriptionEvent('purchase_restored', {
    productId,
  });
}

/**
 * Track upgrade (e.g., monthly → yearly)
 */
export function trackUpgrade(
  previousProductId: string,
  newProductId: string,
  newPrice: number,
  currency: string
): void {
  trackSubscriptionEvent('upgrade_completed', {
    previousProductId,
    productId: newProductId,
    price: newPrice,
    currency,
  });
}

/**
 * Track downgrade (e.g., yearly → monthly)
 */
export function trackDowngrade(previousProductId: string, newProductId: string): void {
  trackSubscriptionEvent('downgrade_completed', {
    previousProductId,
    productId: newProductId,
  });
}

/**
 * Set user property for subscription tier
 * Call this when subscription status changes
 */
export function setSubscriptionUserProperty(tier: 'free' | 'pro', productId?: string): void {
  console.log('[Analytics] Setting user property:', { tier, productId });

  // TODO: Set user property in analytics
  // Mixpanel: mixpanel.people.set({ subscription_tier: tier, product_id: productId });
  // Amplitude: amplitude.setUserProperties({ subscription_tier: tier });
  // Segment: analytics.identify({ traits: { subscription_tier: tier } });
}

export default {
  trackSubscriptionEvent,
  trackPaywallViewed,
  trackPaywallDismissed,
  trackSubscriptionStarted,
  trackTrialConverted,
  trackPurchaseFailed,
  trackPurchaseRestored,
  trackUpgrade,
  trackDowngrade,
  setSubscriptionUserProperty,
};
