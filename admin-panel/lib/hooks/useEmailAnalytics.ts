'use client';

import { useQuery } from '@tanstack/react-query';
import { adminApi } from '../api';

export function useEmailAnalytics(params?: { days?: number }) {
  return useQuery({
    queryKey: ['emailAnalytics', params],
    queryFn: () => adminApi.getEmailAnalytics(params),
  });
}

export function useEmailSubscriberStats() {
  return useQuery({
    queryKey: ['emailSubscriberStats'],
    queryFn: () => adminApi.getEmailSubscriberStats(),
  });
}
