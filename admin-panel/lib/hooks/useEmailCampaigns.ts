'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi, CreateEmailCampaignData } from '../api';

export function useEmailCampaigns(params?: {
  status?: string;
  page?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['emailCampaigns', params],
    queryFn: () => adminApi.getEmailCampaigns(params),
  });
}

export function useEmailCampaign(id: string) {
  return useQuery({
    queryKey: ['emailCampaign', id],
    queryFn: () => adminApi.getEmailCampaign(id),
    enabled: !!id,
  });
}

export function useCreateEmailCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateEmailCampaignData) =>
      adminApi.createEmailCampaign(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emailCampaigns'] });
    },
  });
}

export function useUpdateEmailCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string;
      data: Partial<CreateEmailCampaignData>;
    }) => adminApi.updateEmailCampaign(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['emailCampaigns'] });
      queryClient.invalidateQueries({ queryKey: ['emailCampaign', id] });
    },
  });
}

export function useDeleteEmailCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => adminApi.deleteEmailCampaign(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emailCampaigns'] });
    },
  });
}

export function useSendEmailCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => adminApi.sendEmailCampaign(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['emailCampaigns'] });
      queryClient.invalidateQueries({ queryKey: ['emailCampaign', id] });
    },
  });
}

export function useCancelEmailCampaign() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => adminApi.cancelEmailCampaign(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['emailCampaigns'] });
      queryClient.invalidateQueries({ queryKey: ['emailCampaign', id] });
    },
  });
}
