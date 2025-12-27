'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi, CreateEmailTemplateData } from '../api';

export function useEmailTemplates(params?: {
  category?: 'TRANSACTIONAL' | 'MARKETING';
  isActive?: boolean;
  search?: string;
  page?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['emailTemplates', params],
    queryFn: () => adminApi.getEmailTemplates(params),
  });
}

export function useEmailTemplate(id: string) {
  return useQuery({
    queryKey: ['emailTemplate', id],
    queryFn: () => adminApi.getEmailTemplate(id),
    enabled: !!id,
  });
}

export function useCreateEmailTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateEmailTemplateData) =>
      adminApi.createEmailTemplate(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emailTemplates'] });
    },
  });
}

export function useUpdateEmailTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string;
      data: Partial<CreateEmailTemplateData>;
    }) => adminApi.updateEmailTemplate(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['emailTemplates'] });
      queryClient.invalidateQueries({ queryKey: ['emailTemplate', id] });
    },
  });
}

export function useDeleteEmailTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => adminApi.deleteEmailTemplate(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emailTemplates'] });
    },
  });
}

export function usePreviewEmailTemplate() {
  return useMutation({
    mutationFn: ({
      id,
      testData,
    }: {
      id: string;
      testData?: Record<string, unknown>;
    }) => adminApi.previewEmailTemplate(id, testData),
  });
}

export function useSendTestEmail() {
  return useMutation({
    mutationFn: ({
      id,
      email,
      testData,
    }: {
      id: string;
      email: string;
      testData?: Record<string, unknown>;
    }) => adminApi.sendTestEmail(id, email, testData),
  });
}

// Template Version Hooks

export function useEmailTemplateVersions(
  id: string,
  params?: { page?: number; limit?: number }
) {
  return useQuery({
    queryKey: ['emailTemplateVersions', id, params],
    queryFn: () => adminApi.getEmailTemplateVersions(id, params),
    enabled: !!id,
  });
}

export function useEmailTemplateVersion(id: string, version: number) {
  return useQuery({
    queryKey: ['emailTemplateVersion', id, version],
    queryFn: () => adminApi.getEmailTemplateVersion(id, version),
    enabled: !!id && version > 0,
  });
}

export function useRestoreEmailTemplateVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, version }: { id: string; version: number }) =>
      adminApi.restoreEmailTemplateVersion(id, version),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['emailTemplate', id] });
      queryClient.invalidateQueries({ queryKey: ['emailTemplateVersions', id] });
      queryClient.invalidateQueries({ queryKey: ['emailTemplates'] });
    },
  });
}
