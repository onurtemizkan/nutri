import { z } from 'zod';

// ============================================================================
// ADMIN AUTHENTICATION SCHEMAS
// ============================================================================

export const adminLoginSchema = z.object({
  email: z.string().email('Invalid email format'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});

export const adminMFAVerifySchema = z.object({
  pendingToken: z.string().min(1, 'Pending token is required'),
  code: z
    .string()
    .length(6, 'MFA code must be exactly 6 digits')
    .regex(/^\d{6}$/, 'MFA code must be numeric'),
});

export const adminMFASetupSchema = z.object({
  pendingToken: z.string().min(1, 'Pending token is required'),
});

// ============================================================================
// ADMIN USER MANAGEMENT SCHEMAS
// ============================================================================

export const listUsersQuerySchema = z.object({
  search: z.string().optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sortBy: z.enum(['createdAt', 'email', 'name']).default('createdAt'),
  sortOrder: z.enum(['asc', 'desc']).default('desc'),
  status: z.enum(['active', 'disabled']).optional(),
  subscriptionStatus: z.enum(['active', 'trial', 'expired', 'none']).optional(),
});

export const userIdParamSchema = z.object({
  id: z.string().cuid('Invalid user ID format'),
});

export const deleteUserSchema = z.object({
  reason: z
    .string()
    .min(10, 'Reason must be at least 10 characters')
    .max(500, 'Reason must be at most 500 characters'),
});

// ============================================================================
// ADMIN SUBSCRIPTION MANAGEMENT SCHEMAS
// ============================================================================

export const listSubscriptionsQuerySchema = z.object({
  status: z
    .enum([
      'ACTIVE',
      'EXPIRED',
      'IN_GRACE_PERIOD',
      'IN_BILLING_RETRY',
      'REVOKED',
      'REFUNDED',
      'CANCELLED',
    ])
    .optional(),
  productId: z.string().optional(),
  userId: z.string().optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

export const subscriptionIdParamSchema = z.object({
  id: z.string().cuid('Invalid subscription ID format'),
});

export const lookupSubscriptionQuerySchema = z.object({
  txn: z.string().min(1, 'Transaction ID is required'),
});

export const grantSubscriptionSchema = z.object({
  duration: z.enum(['7_days', '30_days', '90_days', '1_year']),
  reason: z
    .string()
    .min(10, 'Reason must be at least 10 characters')
    .max(500, 'Reason must be at most 500 characters'),
});

export const extendSubscriptionSchema = z.object({
  days: z.number().int().min(1).max(365),
  reason: z
    .string()
    .min(10, 'Reason must be at least 10 characters')
    .max(500, 'Reason must be at most 500 characters'),
});

export const revokeSubscriptionSchema = z.object({
  reason: z
    .string()
    .min(10, 'Reason must be at least 10 characters')
    .max(500, 'Reason must be at most 500 characters'),
});

// ============================================================================
// ADMIN WEBHOOK SCHEMAS
// ============================================================================

export const listWebhooksQuerySchema = z.object({
  notificationType: z
    .enum([
      'SUBSCRIBED',
      'DID_RENEW',
      'DID_CHANGE_RENEWAL_STATUS',
      'DID_FAIL_TO_RENEW',
      'EXPIRED',
      'REFUND',
      'REVOKE',
      'OFFER_REDEEMED',
      'GRACE_PERIOD_EXPIRED',
    ])
    .optional(),
  status: z.enum(['success', 'failed', 'pending']).optional(),
  startDate: z.string().datetime().optional(),
  endDate: z.string().datetime().optional(),
  subscriptionId: z.string().optional(),
  originalTransactionId: z.string().optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

export const webhookIdParamSchema = z.object({
  id: z.string().cuid('Invalid webhook ID format'),
});

// ============================================================================
// ADMIN AUDIT LOG SCHEMAS
// ============================================================================

export const listAuditLogsQuerySchema = z.object({
  adminUserId: z.string().cuid().optional(),
  action: z.string().optional(),
  startDate: z.string().datetime().optional(),
  endDate: z.string().datetime().optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type AdminLoginInput = z.infer<typeof adminLoginSchema>;
export type AdminMFAVerifyInput = z.infer<typeof adminMFAVerifySchema>;
export type ListUsersQuery = z.infer<typeof listUsersQuerySchema>;
export type DeleteUserInput = z.infer<typeof deleteUserSchema>;
export type ListSubscriptionsQuery = z.infer<typeof listSubscriptionsQuerySchema>;
export type GrantSubscriptionInput = z.infer<typeof grantSubscriptionSchema>;
export type ExtendSubscriptionInput = z.infer<typeof extendSubscriptionSchema>;
export type RevokeSubscriptionInput = z.infer<typeof revokeSubscriptionSchema>;
export type ListWebhooksQuery = z.infer<typeof listWebhooksQuerySchema>;
export type ListAuditLogsQuery = z.infer<typeof listAuditLogsQuerySchema>;
