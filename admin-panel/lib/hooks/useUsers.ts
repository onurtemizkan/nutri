import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import type { UsersListResponse, UserDetail, PaginationMeta } from '@/lib/types';

interface UseUsersParams {
  search?: string;
  page?: number;
  limit?: number;
  subscriptionStatus?: 'active' | 'trial' | 'expired' | 'none';
}

/**
 * Hook for fetching paginated users list
 */
export function useUsers(params: UseUsersParams = {}) {
  const { search, page = 1, limit = 20, subscriptionStatus } = params;

  return useQuery<UsersListResponse>({
    queryKey: ['users', { search, page, limit, subscriptionStatus }],
    queryFn: async () => {
      const response = await adminApi.getUsers({
        search: search || undefined,
        page,
        limit,
        subscriptionStatus: subscriptionStatus || undefined,
      });
      return response as UsersListResponse;
    },
  });
}

/**
 * Hook for fetching a single user detail
 */
export function useUser(userId: string | null) {
  return useQuery<UserDetail>({
    queryKey: ['user', userId],
    queryFn: async () => {
      if (!userId) throw new Error('User ID is required');
      const response = await adminApi.getUser(userId);
      return response as UserDetail;
    },
    enabled: !!userId,
  });
}

/**
 * Type for the users query response
 */
export interface UsersQueryResult {
  users: UsersListResponse['users'];
  pagination: PaginationMeta;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}
