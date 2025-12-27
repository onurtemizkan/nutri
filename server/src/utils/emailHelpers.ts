/**
 * Email Helper Utilities
 *
 * Provides utility functions for email operations including:
 * - Unsubscribe token generation and verification
 * - Email validation
 * - List-Unsubscribe header generation (RFC 8058)
 * - Webhook signature verification
 */

import crypto from 'crypto';
import { config } from '../config/env';
import { EMAIL_CONFIG } from '../config/resend';

/**
 * Token expiration time in days
 */
const TOKEN_EXPIRY_DAYS = 30;

/**
 * Generate a secure unsubscribe token for a user
 * The token is URL-safe and includes an expiration timestamp
 *
 * @param userId - User ID to encode in token
 * @param campaignId - Optional campaign ID for tracking
 * @returns Secure token string
 */
export function generateUnsubscribeToken(userId: string, campaignId?: string): string {
  const expiresAt = Date.now() + TOKEN_EXPIRY_DAYS * 24 * 60 * 60 * 1000;
  const payload = JSON.stringify({
    uid: userId,
    cid: campaignId,
    exp: expiresAt,
    nonce: crypto.randomBytes(8).toString('hex'),
  });

  // Create HMAC signature
  const hmac = crypto.createHmac('sha256', config.jwt.secret);
  hmac.update(payload);
  const signature = hmac.digest('base64url');

  // Encode payload
  const encodedPayload = Buffer.from(payload).toString('base64url');

  return `${encodedPayload}.${signature}`;
}

/**
 * Verify and decode an unsubscribe token
 *
 * @param token - Token to verify
 * @returns Decoded payload or null if invalid/expired
 */
export function verifyUnsubscribeToken(
  token: string
): { userId: string; campaignId?: string } | null {
  try {
    const [encodedPayload, signature] = token.split('.');
    if (!encodedPayload || !signature) {
      return null;
    }

    // Decode payload
    const payload = Buffer.from(encodedPayload, 'base64url').toString('utf-8');
    const data = JSON.parse(payload) as {
      uid: string;
      cid?: string;
      exp: number;
      nonce: string;
    };

    // Verify signature
    const hmac = crypto.createHmac('sha256', config.jwt.secret);
    hmac.update(payload);
    const expectedSignature = hmac.digest('base64url');

    if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSignature))) {
      return null;
    }

    // Check expiration
    if (Date.now() > data.exp) {
      return null;
    }

    return {
      userId: data.uid,
      campaignId: data.cid,
    };
  } catch {
    return null;
  }
}

/**
 * Basic email validation
 * Uses a simple regex for format validation
 *
 * @param email - Email address to validate
 * @returns Whether the email appears valid
 */
export function validateEmailFormat(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Check if an email domain has valid MX records
 * Note: This is a basic check - for production, consider using a service like ZeroBounce
 *
 * @param email - Email address to check
 * @returns Whether the domain appears deliverable
 */
export async function validateEmailDeliverability(email: string): Promise<{
  valid: boolean;
  reason?: string;
}> {
  if (!validateEmailFormat(email)) {
    return { valid: false, reason: 'Invalid email format' };
  }

  // In production, you might want to:
  // 1. Check MX records with dns.resolveMx()
  // 2. Use a validation service like ZeroBounce, Kickbox, etc.
  // For now, we just do format validation

  // Check for common disposable email domains
  const disposableDomains = [
    'mailinator.com',
    'guerrillamail.com',
    '10minutemail.com',
    'tempmail.com',
    'throwaway.email',
  ];

  const domain = email.split('@')[1]?.toLowerCase();
  if (domain && disposableDomains.includes(domain)) {
    return { valid: false, reason: 'Disposable email addresses are not allowed' };
  }

  return { valid: true };
}

/**
 * Generate List-Unsubscribe headers for RFC 8058 compliance
 * This enables one-click unsubscribe in email clients
 *
 * @param userId - User ID
 * @param campaignId - Optional campaign ID
 * @returns Headers object with List-Unsubscribe and List-Unsubscribe-Post
 */
export function generateListUnsubscribeHeaders(
  userId: string,
  campaignId?: string
): {
  'List-Unsubscribe': string;
  'List-Unsubscribe-Post': string;
} {
  const token = generateUnsubscribeToken(userId, campaignId);
  const unsubscribeUrl = `${EMAIL_CONFIG.baseUrl}/api/email/unsubscribe?token=${encodeURIComponent(token)}`;

  return {
    'List-Unsubscribe': `<${unsubscribeUrl}>`,
    'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
  };
}

/**
 * Generate unsubscribe link URL for email footer
 *
 * @param userId - User ID
 * @param campaignId - Optional campaign ID
 * @returns Full unsubscribe URL
 */
export function generateUnsubscribeUrl(userId: string, campaignId?: string): string {
  const token = generateUnsubscribeToken(userId, campaignId);
  return `${EMAIL_CONFIG.baseUrl}/email/unsubscribe?token=${encodeURIComponent(token)}`;
}

/**
 * Generate preference center link URL
 *
 * @param userId - User ID
 * @returns Full preference center URL
 */
export function generatePreferenceCenterUrl(userId: string): string {
  const token = generateUnsubscribeToken(userId);
  return `${EMAIL_CONFIG.baseUrl}/email/preferences?token=${encodeURIComponent(token)}`;
}

/**
 * Verify Resend webhook signature
 * Resend uses HMAC SHA256 for webhook signatures
 *
 * @param payload - Raw request body (string)
 * @param signature - Signature from Resend-Signature header
 * @returns Whether the signature is valid
 */
export function verifyResendWebhookSignature(payload: string, signature: string): boolean {
  if (!config.email.webhookSecret) {
    console.warn('Resend webhook secret not configured - skipping signature verification');
    return true; // Allow in development
  }

  try {
    // Resend signature format: t=timestamp,v1=signature
    const parts = signature.split(',');
    const timestampPart = parts.find((p) => p.startsWith('t='));
    const signaturePart = parts.find((p) => p.startsWith('v1='));

    if (!timestampPart || !signaturePart) {
      return false;
    }

    const timestamp = timestampPart.slice(2);
    const providedSignature = signaturePart.slice(3);

    // Verify timestamp is within 5 minutes
    const timestampMs = parseInt(timestamp, 10) * 1000;
    const now = Date.now();
    if (Math.abs(now - timestampMs) > 5 * 60 * 1000) {
      return false;
    }

    // Calculate expected signature
    const signedPayload = `${timestamp}.${payload}`;
    const hmac = crypto.createHmac('sha256', config.email.webhookSecret);
    hmac.update(signedPayload);
    const expectedSignature = hmac.digest('hex');

    // Timing-safe comparison
    return crypto.timingSafeEqual(Buffer.from(providedSignature), Buffer.from(expectedSignature));
  } catch {
    return false;
  }
}

/**
 * Sanitize email content to prevent XSS
 * Removes script tags and event handlers from HTML
 *
 * @param html - HTML content to sanitize
 * @returns Sanitized HTML
 */
export function sanitizeEmailHtml(html: string): string {
  // Remove script tags
  let sanitized = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

  // Remove event handlers
  sanitized = sanitized.replace(/\s*on\w+\s*=\s*["'][^"']*["']/gi, '');

  // Remove javascript: URLs
  sanitized = sanitized.replace(/javascript:[^"']*/gi, '');

  return sanitized;
}

/**
 * Generate a correlation ID for email tracking
 *
 * @returns Unique correlation ID
 */
export function generateEmailCorrelationId(): string {
  return `email_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
}
