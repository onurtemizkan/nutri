/**
 * Admin Panel Type Definitions
 * Strict TypeScript types - no 'any' allowed
 */

// Admin user roles
export type AdminRole = 'SUPER_ADMIN' | 'SUPPORT' | 'ANALYST' | 'VIEWER';

// Admin user type
export interface AdminUser {
  id: string;
  email: string;
  name: string;
  role: AdminRole;
  mfaEnabled: boolean;
  isActive: boolean;
  lastLoginAt: string | null;
  createdAt: string;
  updatedAt: string;
}

// Session user (subset for client)
export interface SessionUser {
  id: string;
  email: string;
  name: string;
  role: AdminRole;
}

// Pagination metadata
export interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

// Paginated response
export interface PaginatedResponse<T> {
  data: T[];
  pagination: PaginationMeta;
}

// Subscription status
export type SubscriptionStatus =
  | 'ACTIVE'
  | 'EXPIRED'
  | 'IN_GRACE_PERIOD'
  | 'IN_BILLING_RETRY'
  | 'REVOKED'
  | 'REFUNDED'
  | 'CANCELLED';

// Subscription type
export interface Subscription {
  id: string;
  userId: string;
  productId: string;
  originalTransactionId: string;
  status: SubscriptionStatus;
  expiresAt: string;
  isTrialPeriod: boolean;
  isIntroOfferPeriod: boolean;
  autoRenewEnabled: boolean;
  gracePeriodExpiresAt: string | null;
  billingRetryPeriod: boolean;
  priceLocale: string | null;
  priceCurrency: string | null;
  priceAmount: number | null;
  environment: 'sandbox' | 'production';
  createdAt: string;
  updatedAt: string;
  user?: User;
}

// Subscription tier (from backend)
export type SubscriptionTier = 'FREE' | 'PRO_TRIAL' | 'PRO';
export type BillingCycle = 'MONTHLY' | 'ANNUAL' | null;

// App user type (from main app)
export interface User {
  id: string;
  email: string;
  name: string;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight: number | null;
  goalWeight: number | null;
  height: number | null;
  activityLevel: string;
  createdAt: string;
  updatedAt: string;
  // Subscription fields on User model
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle: BillingCycle;
  subscriptionStartDate: string | null;
  subscriptionEndDate: string | null;
  subscriptionPrice: number | null;
}

// User list item (subset for list views)
export interface UserListItem {
  id: string;
  email: string;
  name: string;
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle: BillingCycle;
  subscriptionStartDate: string | null;
  subscriptionEndDate: string | null;
  createdAt: string;
}

// User detail response with activity stats
export interface UserDetail extends User {
  mealsCount: number;
  healthMetricsCount: number;
  activitiesCount: number;
  recentMeals: number;
  recentHealthMetrics: number;
}

// Users list response
export interface UsersListResponse {
  users: UserListItem[];
  pagination: PaginationMeta;
}

// Webhook event status
export type WebhookStatus = 'success' | 'failed' | 'pending';

// Webhook notification types (App Store)
export type WebhookNotificationType =
  | 'SUBSCRIBED'
  | 'DID_RENEW'
  | 'DID_CHANGE_RENEWAL_STATUS'
  | 'DID_FAIL_TO_RENEW'
  | 'EXPIRED'
  | 'REFUND'
  | 'REVOKE'
  | 'OFFER_REDEEMED'
  | 'GRACE_PERIOD_EXPIRED';

// Webhook event type
export interface WebhookEvent {
  id: string;
  subscriptionId: string | null;
  notificationType: WebhookNotificationType;
  subtype: string | null;
  transactionId: string;
  originalTransactionId: string;
  payload: Record<string, unknown>;
  status: WebhookStatus;
  errorMessage: string | null;
  processedAt: string;
  createdAt: string;
  subscription?: Subscription | null;
}

// Audit log action types
export type AuditAction =
  | 'USER_LIST'
  | 'USER_VIEW'
  | 'USER_EXPORT'
  | 'USER_DELETE'
  | 'SUBSCRIPTION_LIST'
  | 'SUBSCRIPTION_VIEW'
  | 'SUBSCRIPTION_GRANT'
  | 'SUBSCRIPTION_EXTEND'
  | 'SUBSCRIPTION_REVOKE'
  | 'WEBHOOK_LIST'
  | 'WEBHOOK_VIEW'
  | 'WEBHOOK_RETRY'
  | 'IP_BLOCKED';

// Audit log type
export interface AuditLog {
  id: string;
  adminUserId: string;
  adminUser?: AdminUser;
  action: AuditAction;
  targetType: string | null;
  targetId: string | null;
  details: Record<string, unknown> | null;
  ipAddress: string;
  userAgent: string | null;
  createdAt: string;
}

// Analytics types
export interface SubscriptionMetrics {
  mrr: number;
  activeSubscribers: {
    total: number;
    proMonthly: number;
    proYearly: number;
    trial: number;
  };
  newSubscriptions: {
    today: number;
    week: number;
    month: number;
  };
  churn: {
    rate: number;
    count: number;
  };
  trials: {
    active: number;
    conversionRate: number;
  };
}

export interface TimeSeriesDataPoint {
  date: string;
  value: number;
}

// Feature flag types
export type FeatureFlagType = 'BOOLEAN' | 'STRING' | 'NUMBER' | 'JSON';

export interface FeatureFlag {
  id: string;
  key: string;
  name: string;
  description: string | null;
  type: FeatureFlagType;
  value: unknown;
  isEnabled: boolean;
  targeting: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

// API error response
export interface ApiError {
  error: string;
  message?: string;
  statusCode?: number;
}

// Table column definition for TanStack Table
export interface TableColumn<T> {
  id: string;
  header: string;
  accessorKey?: keyof T;
  accessorFn?: (row: T) => unknown;
  cell?: (info: { getValue: () => unknown; row: { original: T } }) => React.ReactNode;
  enableSorting?: boolean;
  enableFiltering?: boolean;
}
