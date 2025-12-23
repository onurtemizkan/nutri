import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import type { UserListItem, PaginationMeta } from '@/lib/types';

interface UseSubscriptionsParams {
  status?: 'active' | 'trial' | 'expired' | 'none';
  page?: number;
  limit?: number;
}

// Subscription list item extends UserListItem since subscriptions are on User model
export interface SubscriptionListItem extends UserListItem {
  subscriptionPrice: number | null;
}

export interface SubscriptionsListResponse {
  subscriptions: SubscriptionListItem[];
  pagination: PaginationMeta;
}

export interface SubscriptionDetail extends SubscriptionListItem {
  isActive: boolean;
  daysRemaining: number | null;
  appleId: string | null;
  updatedAt: string;
}

/**
 * Hook for fetching paginated subscriptions list
 */
export function useSubscriptions(params: UseSubscriptionsParams = {}) {
  const { status, page = 1, limit = 20 } = params;

  return useQuery<SubscriptionsListResponse>({
    queryKey: ['subscriptions', { status, page, limit }],
    queryFn: async () => {
      const response = await adminApi.getSubscriptions({
        status: status || undefined,
        page,
        limit,
      });
      return response as SubscriptionsListResponse;
    },
  });
}

/**
 * Hook for fetching a single subscription detail
 */
export function useSubscription(userId: string | null) {
  return useQuery<SubscriptionDetail>({
    queryKey: ['subscription', userId],
    queryFn: async () => {
      if (!userId) throw new Error('User ID is required');
      const response = await adminApi.getSubscription(userId);
      return response as SubscriptionDetail;
    },
    enabled: !!userId,
  });
}

/**
 * Hook for looking up subscription by transaction ID
 */
export function useLookupSubscription(transactionId: string | null) {
  return useQuery<SubscriptionDetail>({
    queryKey: ['subscription-lookup', transactionId],
    queryFn: async () => {
      if (!transactionId) throw new Error('Transaction ID is required');
      const response = await adminApi.lookupSubscription(transactionId);
      return response as SubscriptionDetail;
    },
    enabled: !!transactionId && transactionId.length > 5,
  });
}

/**
 * Hook for granting subscription
 */
export function useGrantSubscription() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      userId,
      duration,
      reason,
    }: {
      userId: string;
      duration: string;
      reason: string;
    }) => {
      return adminApi.grantSubscription(userId, duration, reason);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['subscriptions'] });
      queryClient.invalidateQueries({ queryKey: ['subscription'] });
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

/**
 * Hook for extending subscription
 */
export function useExtendSubscription() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      userId,
      days,
      reason,
    }: {
      userId: string;
      days: number;
      reason: string;
    }) => {
      return adminApi.extendSubscription(userId, days, reason);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['subscriptions'] });
      queryClient.invalidateQueries({ queryKey: ['subscription'] });
    },
  });
}

/**
 * Hook for revoking subscription
 */
export function useRevokeSubscription() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ userId, reason }: { userId: string; reason: string }) => {
      return adminApi.revokeSubscription(userId, reason);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['subscriptions'] });
      queryClient.invalidateQueries({ queryKey: ['subscription'] });
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}
