'use client';

import { useState } from 'react';
import { Search, X, Loader2, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { useSession } from 'next-auth/react';
import {
  useWebhooks,
  useWebhook,
  useRetryWebhook,
  type WebhookEventStatus,
} from '@/lib/hooks/useWebhooks';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { WebhookTable } from '@/components/webhooks/webhook-table';
import { WebhookDetailModal } from '@/components/webhooks/webhook-detail-modal';

type NotificationTypeFilter =
  | ''
  | 'SUBSCRIBED'
  | 'DID_RENEW'
  | 'DID_FAIL_TO_RENEW'
  | 'EXPIRED'
  | 'REFUND'
  | 'REVOKE';

export default function WebhooksPage() {
  const session = useSession();
  const [notificationType, setNotificationType] = useState<NotificationTypeFilter>('');
  const [status, setStatus] = useState<WebhookEventStatus | ''>('');
  const [transactionSearch, setTransactionSearch] = useState('');
  const [page, setPage] = useState(1);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  // Debounce transaction search
  const debouncedTransactionSearch = useDebounce(transactionSearch, 500);

  const { data, isLoading, isError, error } = useWebhooks({
    notificationType: notificationType || undefined,
    status: status || undefined,
    originalTransactionId: debouncedTransactionSearch || undefined,
    page,
    limit: 20,
  });

  const {
    data: selectedWebhook,
    isLoading: isLoadingDetail,
  } = useWebhook(selectedEventId);

  const retryMutation = useRetryWebhook();

  const isSuperAdmin = session?.data?.user?.role === 'SUPER_ADMIN';

  const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setNotificationType(e.target.value as NotificationTypeFilter);
    setPage(1);
  };

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatus(e.target.value as WebhookEventStatus | '');
    setPage(1);
  };

  const handleTransactionSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTransactionSearch(e.target.value);
    setPage(1);
  };

  const clearTransactionSearch = () => {
    setTransactionSearch('');
    setPage(1);
  };

  const handleViewDetail = (id: string) => {
    setSelectedEventId(id);
  };

  const handleCloseDetail = () => {
    setSelectedEventId(null);
  };

  const handleRetry = async (id: string) => {
    try {
      await retryMutation.mutateAsync(id);
    } catch {
      // Error handled by mutation
    }
  };

  // Calculate stats from data
  const stats = data?.events ? {
    total: data.pagination.total,
    pending: data.events.filter(e => e.status === 'PENDING').length,
    success: data.events.filter(e => e.status === 'SUCCESS').length,
    failed: data.events.filter(e => e.status === 'FAILED').length,
  } : null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">Webhook Events</h2>
          <p className="mt-1 text-text-tertiary">
            View and manage App Store Server Notifications
          </p>
        </div>
        {data?.pagination && (
          <div className="text-sm text-text-tertiary">
            {data.pagination.total} total events
          </div>
        )}
      </div>

      {/* Stats Cards */}
      {stats && stats.total > 0 && (
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-500/10">
                <Clock className="h-5 w-5 text-yellow-500" />
              </div>
              <div>
                <p className="text-2xl font-bold text-text-primary">
                  {stats.pending}
                </p>
                <p className="text-sm text-text-tertiary">Pending</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-500/10">
                <CheckCircle className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-2xl font-bold text-text-primary">
                  {stats.success}
                </p>
                <p className="text-sm text-text-tertiary">Successful</p>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10">
                <AlertCircle className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <p className="text-2xl font-bold text-text-primary">
                  {stats.failed}
                </p>
                <p className="text-sm text-text-tertiary">Failed</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {/* Transaction ID Search */}
        <div className="relative flex-1 min-w-[250px] max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-disabled" />
          <input
            type="text"
            value={transactionSearch}
            onChange={handleTransactionSearchChange}
            placeholder="Search by transaction ID..."
            className="w-full rounded-md border border-border bg-background-elevated py-2 pl-10 pr-10 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
          {transactionSearch && (
            <button
              onClick={clearTransactionSearch}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-disabled hover:text-text-secondary"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Notification Type Filter */}
        <select
          value={notificationType}
          onChange={handleTypeChange}
          className="rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
        >
          <option value="">All Types</option>
          <option value="SUBSCRIBED">Subscribed</option>
          <option value="DID_RENEW">Renewed</option>
          <option value="DID_FAIL_TO_RENEW">Failed to Renew</option>
          <option value="EXPIRED">Expired</option>
          <option value="REFUND">Refund</option>
          <option value="REVOKE">Revoke</option>
        </select>

        {/* Status Filter */}
        <select
          value={status}
          onChange={handleStatusChange}
          className="rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
        >
          <option value="">All Statuses</option>
          <option value="SUCCESS">Success</option>
          <option value="FAILED">Failed</option>
          <option value="PENDING">Pending</option>
        </select>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}

      {/* Error State */}
      {isError && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-red-500">
          Failed to load webhook events: {error?.message || 'Unknown error'}
        </div>
      )}

      {/* Webhooks Table */}
      {!isLoading && !isError && (
        <WebhookTable
          webhooks={data?.events || []}
          pagination={
            data?.pagination || { page: 1, limit: 20, total: 0, totalPages: 0 }
          }
          isLoading={isLoading}
          onPageChange={setPage}
          onViewDetail={handleViewDetail}
          onRetry={handleRetry}
          isRetrying={retryMutation.isPending}
          isSuperAdmin={isSuperAdmin}
        />
      )}

      {/* Webhook Detail Modal */}
      <WebhookDetailModal
        isOpen={selectedEventId !== null}
        onClose={handleCloseDetail}
        webhook={selectedWebhook || null}
        isLoading={isLoadingDetail}
        onRetry={handleRetry}
        isRetrying={retryMutation.isPending}
        isSuperAdmin={isSuperAdmin}
      />
    </div>
  );
}
