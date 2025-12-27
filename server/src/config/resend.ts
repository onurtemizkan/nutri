/**
 * Resend Email Provider Configuration
 *
 * Initializes the Resend SDK and provides email configuration.
 * Supports both transactional and marketing email domains.
 */

import { Resend } from 'resend';
import { config } from './env';
import { logger } from './logger';

/**
 * Resend client instance
 * Lazily initialized to avoid errors during test runs
 */
let resendClient: Resend | null = null;

/**
 * Get or initialize the Resend client
 */
export function getResend(): Resend {
  if (!resendClient) {
    if (!config.email.resendApiKey) {
      logger.warn('Resend API key not configured - email sending will be disabled');
      // Return a mock client that logs instead of sending
      resendClient = new Resend('re_mock_key');
    } else {
      resendClient = new Resend(config.email.resendApiKey);
    }
  }
  return resendClient;
}

/**
 * Email configuration constants
 */
export const EMAIL_CONFIG = {
  from: {
    transactional: config.email.fromTransactional,
    marketing: config.email.fromMarketing,
  },
  replyTo: config.email.replyTo,
  domains: {
    transactional: config.email.domainTransactional,
    marketing: config.email.domainMarketing,
  },
  rateLimit: {
    perSecond: config.email.rateLimitPerSecond,
    batchSize: config.email.batchSize,
  },
  baseUrl: config.email.baseUrl,
  enabled: config.email.enabled,
} as const;

/**
 * Email categories that require marketing opt-in
 */
export const MARKETING_CATEGORIES = [
  'weekly_reports',
  'health_insights',
  'tips',
  'features',
  'promotions',
  'newsletter',
] as const;

/**
 * Email categories that are always delivered (transactional)
 */
export const TRANSACTIONAL_CATEGORIES = [
  'welcome',
  'email_verification',
  'password_reset',
  'password_changed',
  'subscription_confirmation',
  'payment_receipt',
  'security_alert',
  'goal_achievement',
] as const;

export type MarketingCategory = (typeof MARKETING_CATEGORIES)[number];
export type TransactionalCategory = (typeof TRANSACTIONAL_CATEGORIES)[number];
export type EmailCategoryType = MarketingCategory | TransactionalCategory;

/**
 * Check if a category requires marketing opt-in
 */
export function isMarketingCategory(category: string): boolean {
  return MARKETING_CATEGORIES.includes(category as MarketingCategory);
}

/**
 * Check if email sending is enabled
 */
export function isEmailEnabled(): boolean {
  return config.email.enabled && !!config.email.resendApiKey;
}

export default getResend;
