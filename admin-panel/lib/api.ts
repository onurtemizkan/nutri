import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

/**
 * API client for communicating with the Nutri backend
 * Configured to work with NextAuth session tokens
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_URL}/api`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token cache to avoid repeated session calls
let cachedToken: string | null = null;
let tokenPromise: Promise<string | null> | null = null;

async function getAuthToken(): Promise<string | null> {
  // Return cached token if available
  if (cachedToken) return cachedToken;

  // Dedupe concurrent requests
  if (tokenPromise) return tokenPromise;

  tokenPromise = (async () => {
    try {
      const { getSession } = await import('next-auth/react');
      const session = await getSession();
      cachedToken = session?.accessToken || null;
      return cachedToken;
    } catch {
      return null;
    } finally {
      tokenPromise = null;
    }
  })();

  return tokenPromise;
}

// Clear token cache on logout
export function clearTokenCache() {
  cachedToken = null;
}

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  async (config) => {
    if (typeof window !== 'undefined') {
      const token = await getAuthToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<{ error?: string; message?: string }>) => {
    // Handle 401 - clear cache and let NextAuth/middleware handle redirect
    if (error.response?.status === 401) {
      clearTokenCache();
      // Use NextAuth's signOut instead of manual redirect to avoid loops
      if (typeof window !== 'undefined') {
        const { signOut } = await import('next-auth/react');
        signOut({ callbackUrl: '/login' });
      }
    }

    // Extract error message
    const message =
      error.response?.data?.error ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';

    return Promise.reject(new Error(message));
  }
);

/**
 * API wrapper with typed responses
 */
export const api = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    apiClient.get<T>(url, config).then((res) => res.data),

  post: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.post<T>(url, data, config).then((res) => res.data),

  put: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.put<T>(url, data, config).then((res) => res.data),

  patch: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiClient.patch<T>(url, data, config).then((res) => res.data),

  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    apiClient.delete<T>(url, config).then((res) => res.data),
};

/**
 * Admin-specific API endpoints
 */
export const adminApi = {
  // Auth
  login: (email: string, password: string) =>
    api.post<{ token: string; requiresMFA: boolean; qrCode?: string }>(
      '/admin/auth/login',
      { email, password }
    ),

  verifyMFA: (token: string, code: string) =>
    api.post<{ token: string }>('/admin/auth/mfa/verify', { token, code }),

  logout: () => api.post('/admin/auth/logout'),

  getMe: () =>
    api.get<{
      id: string;
      email: string;
      name: string;
      role: string;
    }>('/admin/auth/me'),

  // Users
  getUsers: (params?: {
    search?: string;
    page?: number;
    limit?: number;
    status?: string;
    subscriptionStatus?: string;
  }) => api.get<{ users: unknown[]; pagination: unknown }>('/admin/users', { params }),

  getUser: (id: string) => api.get<unknown>(`/admin/users/${id}`),

  exportUserData: (id: string) =>
    api.post<Blob>(`/admin/users/${id}/export`, undefined, {
      responseType: 'blob',
    }),

  deleteUser: (id: string, reason: string) =>
    api.delete(`/admin/users/${id}`, { data: { reason } }),

  // Subscriptions
  getSubscriptions: (params?: {
    status?: string;
    productId?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ subscriptions: unknown[]; pagination: unknown }>(
      '/admin/subscriptions',
      { params }
    ),

  getSubscription: (id: string) => api.get<unknown>(`/admin/subscriptions/${id}`),

  lookupSubscription: (transactionId: string) =>
    api.get<unknown>('/admin/subscriptions/lookup', {
      params: { txn: transactionId },
    }),

  grantSubscription: (userId: string, duration: string, reason: string) =>
    api.post(`/admin/subscriptions/${userId}/grant`, { duration, reason }),

  extendSubscription: (id: string, days: number, reason: string) =>
    api.post(`/admin/subscriptions/${id}/extend`, { days, reason }),

  revokeSubscription: (id: string, reason: string) =>
    api.post(`/admin/subscriptions/${id}/revoke`, { reason }),

  // Webhooks
  getWebhooks: (params?: {
    notificationType?: string;
    status?: string;
    startDate?: string;
    endDate?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ webhooks: unknown[]; pagination: unknown }>('/admin/webhooks', {
      params,
    }),

  getWebhook: (id: string) => api.get<unknown>(`/admin/webhooks/${id}`),

  retryWebhook: (id: string) => api.post(`/admin/webhooks/${id}/retry`),

  // Analytics
  getAnalyticsOverview: () =>
    api.get<{
      mrr: number;
      activeSubscribers: { total: number; proMonthly: number; proYearly: number; trial: number };
      newSubscriptions: { today: number; week: number; month: number };
      churn: { rate: number; count: number };
      trials: { active: number; conversionRate: number };
    }>('/admin/analytics/overview'),

  getSubscribersOverTime: (days?: number) =>
    api.get<{ date: string; count: number }[]>(
      '/admin/analytics/subscribers-over-time',
      { params: { days } }
    ),

  getRevenueOverTime: (months?: number) =>
    api.get<{ date: string; value: number }[]>(
      '/admin/analytics/revenue-over-time',
      { params: { months } }
    ),

  // Audit logs
  getAuditLogs: (params?: {
    adminUserId?: string;
    action?: string;
    startDate?: string;
    endDate?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ logs: unknown[]; pagination: unknown }>('/admin/audit-logs', {
      params,
    }),

  // Email Templates
  getEmailTemplates: (params?: {
    category?: 'TRANSACTIONAL' | 'MARKETING';
    isActive?: boolean;
    search?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ templates: EmailTemplate[]; pagination: Pagination }>(
      '/admin/email/templates',
      { params }
    ),

  getEmailTemplate: (id: string) =>
    api.get<EmailTemplate>(`/admin/email/templates/${id}`),

  createEmailTemplate: (data: CreateEmailTemplateData) =>
    api.post<EmailTemplate>('/admin/email/templates', data),

  updateEmailTemplate: (id: string, data: Partial<CreateEmailTemplateData>) =>
    api.put<EmailTemplate>(`/admin/email/templates/${id}`, data),

  deleteEmailTemplate: (id: string) =>
    api.delete(`/admin/email/templates/${id}`),

  previewEmailTemplate: (id: string, testData?: Record<string, unknown>) =>
    api.post<{ html: string; subject: string }>(
      `/admin/email/templates/${id}/preview`,
      { testData }
    ),

  sendTestEmail: (id: string, email: string, testData?: Record<string, unknown>) =>
    api.post(`/admin/email/templates/${id}/test`, { email, testData }),

  // Template Versions
  getEmailTemplateVersions: (
    id: string,
    params?: { page?: number; limit?: number }
  ) =>
    api.get<{
      versions: EmailTemplateVersion[];
      currentVersion: number;
      pagination: Pagination;
    }>(`/admin/email/templates/${id}/versions`, { params }),

  getEmailTemplateVersion: (id: string, version: number) =>
    api.get<{ version: EmailTemplateVersion }>(
      `/admin/email/templates/${id}/versions/${version}`
    ),

  restoreEmailTemplateVersion: (id: string, version: number) =>
    api.post<{ template: EmailTemplate; message: string }>(
      `/admin/email/templates/${id}/versions/${version}/restore`
    ),

  // Email Campaigns
  getEmailCampaigns: (params?: {
    status?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ campaigns: EmailCampaign[]; pagination: Pagination }>(
      '/admin/email/campaigns',
      { params }
    ),

  getEmailCampaign: (id: string) =>
    api.get<EmailCampaign>(`/admin/email/campaigns/${id}`),

  createEmailCampaign: (data: CreateEmailCampaignData) =>
    api.post<EmailCampaign>('/admin/email/campaigns', data),

  updateEmailCampaign: (id: string, data: Partial<CreateEmailCampaignData>) =>
    api.put<EmailCampaign>(`/admin/email/campaigns/${id}`, data),

  deleteEmailCampaign: (id: string) =>
    api.delete(`/admin/email/campaigns/${id}`),

  sendEmailCampaign: (id: string) =>
    api.post(`/admin/email/campaigns/${id}/send`),

  cancelEmailCampaign: (id: string) =>
    api.post(`/admin/email/campaigns/${id}/cancel`),

  // Email Sequences
  getEmailSequences: (params?: {
    status?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ sequences: EmailSequence[]; pagination: Pagination }>(
      '/admin/email/sequences',
      { params }
    ),

  getEmailSequence: (id: string) =>
    api.get<EmailSequence>(`/admin/email/sequences/${id}`),

  createEmailSequence: (data: CreateEmailSequenceData) =>
    api.post<EmailSequence>('/admin/email/sequences', data),

  updateEmailSequence: (id: string, data: Partial<CreateEmailSequenceData>) =>
    api.put<EmailSequence>(`/admin/email/sequences/${id}`, data),

  deleteEmailSequence: (id: string) =>
    api.delete(`/admin/email/sequences/${id}`),

  // Email Analytics
  getEmailAnalytics: (params?: { days?: number }) =>
    api.get<EmailAnalytics>('/admin/email/analytics', { params }),

  getEmailSubscriberStats: () =>
    api.get<SubscriberStats>('/admin/email/subscribers'),
};

// Email Types
export interface EmailTemplate {
  id: string;
  name: string;
  slug: string;
  category: 'TRANSACTIONAL' | 'MARKETING';
  subject: string;
  mjmlContent: string;
  htmlContent: string | null;
  variables: Record<string, unknown>;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface CreateEmailTemplateData {
  name: string;
  slug: string;
  category: 'TRANSACTIONAL' | 'MARKETING';
  subject: string;
  mjmlContent: string;
  variables?: Record<string, unknown>;
}

export interface EmailTemplateVersion {
  id: string;
  templateId: string;
  version: number;
  subject: string;
  mjmlContent: string;
  htmlContent: string | null;
  plainTextContent: string | null;
  variables: Record<string, unknown> | null;
  changeNotes: string | null;
  createdByAdminId: string | null;
  createdAt: string;
}

export interface EmailCampaign {
  id: string;
  name: string;
  description: string | null;
  templateId: string;
  template?: EmailTemplate;
  status: 'DRAFT' | 'SCHEDULED' | 'SENDING' | 'SENT' | 'CANCELLED';
  scheduledAt: string | null;
  sentAt: string | null;
  completedAt: string | null;
  segmentCriteria: Record<string, unknown>;
  estimatedAudience: number;
  actualSent: number;
  createdAt: string;
  updatedAt: string;
}

export interface CreateEmailCampaignData {
  name: string;
  description?: string;
  templateId: string;
  scheduledAt?: string;
  segmentCriteria?: Record<string, unknown>;
}

export interface EmailSequence {
  id: string;
  name: string;
  description: string | null;
  triggerEvent: string;
  isActive: boolean;
  steps: unknown[];
  enrollmentCriteria: Record<string, unknown>;
  exitCriteria: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
  _count?: {
    enrollments: number;
  };
}

export interface CreateEmailSequenceData {
  name: string;
  description?: string;
  triggerEvent: string;
  isActive?: boolean;
  steps?: unknown[];
  enrollmentCriteria?: Record<string, unknown>;
  exitCriteria?: Record<string, unknown>;
}

export interface EmailAnalytics {
  period: {
    days: number;
    startDate: string;
  };
  overview: {
    totalEmails: number;
    delivered: number;
    opened: number;
    clicked: number;
    bounced: number;
    complained: number;
    openRate: string;
    clickRate: string;
    bounceRate: string;
    complaintRate: string;
  };
  dailyStats: {
    date: string;
    sent: number;
    opened: number;
    clicked: number;
  }[];
  topCampaigns: {
    id: string;
    name: string;
    sentAt: string;
    actualSent: number;
    openedCount: number;
    clickedCount: number;
  }[];
  bouncesByType: {
    type: string;
    count: number;
  }[];
}

export interface SubscriberStats {
  totalSubscribers: number;
  unsubscribed: number;
  subscribed: number;
  marketingOptIn: number;
  doubleOptInConfirmed: number;
  subscriptionRate: string;
}

export interface Pagination {
  page: number;
  limit: number;
  totalPages: number;
  totalItems: number;
}

export default apiClient;
