import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';

// Analytics overview data
export interface AnalyticsOverview {
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

// Subscriber over time data point
export interface SubscriberDataPoint {
  date: string;
  count: number;
}

// Revenue over time data point
export interface RevenueDataPoint {
  date: string;
  value: number;
}

/**
 * Hook for fetching analytics overview
 */
export function useAnalyticsOverview() {
  return useQuery<AnalyticsOverview>({
    queryKey: ['analytics', 'overview'],
    queryFn: async () => {
      const response = await adminApi.getAnalyticsOverview();
      return response as AnalyticsOverview;
    },
    // Refetch every 5 minutes since analytics data doesn't change frequently
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook for fetching subscriber count over time
 */
export function useSubscribersOverTime(days: number = 30) {
  return useQuery<SubscriberDataPoint[]>({
    queryKey: ['analytics', 'subscribers-over-time', days],
    queryFn: async () => {
      const response = await adminApi.getSubscribersOverTime(days);
      return response as SubscriberDataPoint[];
    },
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook for fetching revenue over time
 */
export function useRevenueOverTime(months: number = 12) {
  return useQuery<RevenueDataPoint[]>({
    queryKey: ['analytics', 'revenue-over-time', months],
    queryFn: async () => {
      const response = await adminApi.getRevenueOverTime(months);
      return response as RevenueDataPoint[];
    },
    staleTime: 5 * 60 * 1000,
  });
}
