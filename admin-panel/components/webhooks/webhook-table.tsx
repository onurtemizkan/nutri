'use client';

import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
} from '@tanstack/react-table';
import { ChevronLeft, ChevronRight, Eye, RotateCw, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { formatDate } from '@/lib/utils';
import type { PaginationMeta } from '@/lib/types';
import type { WebhookEventListItem, WebhookEventStatus } from '@/lib/hooks/useWebhooks';

interface WebhookTableProps {
  webhooks: WebhookEventListItem[];
  pagination: PaginationMeta;
  isLoading: boolean;
  onPageChange: (page: number) => void;
  onViewDetail: (id: string) => void;
  onRetry: (id: string) => void;
  isRetrying: boolean;
  isSuperAdmin: boolean;
}

const columnHelper = createColumnHelper<WebhookEventListItem>();

function getStatusBadgeVariant(status: WebhookEventStatus): 'success' | 'warning' | 'danger' | 'muted' {
  switch (status) {
    case 'SUCCESS':
      return 'success';
    case 'PENDING':
      return 'warning';
    case 'FAILED':
      return 'danger';
    default:
      return 'muted';
  }
}

function getNotificationTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    'SUBSCRIBED': 'Subscribed',
    'DID_RENEW': 'Renewed',
    'DID_FAIL_TO_RENEW': 'Failed to Renew',
    'EXPIRED': 'Expired',
    'REFUND': 'Refund',
    'REVOKE': 'Revoked',
    'DID_CHANGE_RENEWAL_STATUS': 'Renewal Status Changed',
    'DID_CHANGE_RENEWAL_PREF': 'Renewal Preference Changed',
    'OFFER_REDEEMED': 'Offer Redeemed',
    'GRACE_PERIOD_EXPIRED': 'Grace Period Expired',
    'PRICE_INCREASE': 'Price Increase',
    'CONSUMPTION_REQUEST': 'Consumption Request',
  };
  return labels[type] || type;
}

export function WebhookTable({
  webhooks,
  pagination,
  isLoading,
  onPageChange,
  onViewDetail,
  onRetry,
  isRetrying,
  isSuperAdmin,
}: WebhookTableProps) {
  const columns: ColumnDef<WebhookEventListItem, unknown>[] = [
    columnHelper.accessor('notificationType', {
      header: 'Type',
      cell: (info) => {
        const type = info.getValue();
        const subtype = info.row.original.subtype;
        return (
          <div>
            <span className="font-medium text-text-primary">
              {getNotificationTypeLabel(type)}
            </span>
            {subtype && (
              <span className="ml-1.5 text-xs text-text-tertiary">
                ({subtype})
              </span>
            )}
          </div>
        );
      },
    }),
    columnHelper.accessor('originalTransactionId', {
      header: 'Transaction ID',
      cell: (info) => {
        const txnId = info.getValue();
        if (!txnId) return <span className="text-text-disabled">-</span>;
        return (
          <span className="font-mono text-xs text-text-secondary">
            {txnId}
          </span>
        );
      },
    }),
    columnHelper.accessor('status', {
      header: 'Status',
      cell: (info) => {
        const status = info.getValue();
        return (
          <Badge variant={getStatusBadgeVariant(status)}>
            {status}
          </Badge>
        );
      },
    }),
    columnHelper.accessor('receivedAt', {
      header: 'Received',
      cell: (info) => {
        const date = info.getValue();
        return (
          <span className="text-text-secondary text-sm">
            {formatDate(date, { dateStyle: 'medium', timeStyle: 'short' })}
          </span>
        );
      },
    }),
    columnHelper.accessor('processedAt', {
      header: 'Processed',
      cell: (info) => {
        const date = info.getValue();
        if (!date) return <span className="text-text-disabled">-</span>;
        return (
          <span className="text-text-secondary text-sm">
            {formatDate(date, { dateStyle: 'medium', timeStyle: 'short' })}
          </span>
        );
      },
    }),
    columnHelper.display({
      id: 'actions',
      header: 'Actions',
      cell: (info) => {
        const row = info.row.original;
        const canRetry = isSuperAdmin && row.status === 'FAILED';

        return (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewDetail(row.id)}
              className="gap-1.5"
            >
              <Eye className="h-4 w-4" />
              View
            </Button>
            {canRetry && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onRetry(row.id)}
                disabled={isRetrying}
                className="gap-1.5"
              >
                {isRetrying ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RotateCw className="h-4 w-4" />
                )}
                Retry
              </Button>
            )}
          </div>
        );
      },
    }),
  ] as ColumnDef<WebhookEventListItem, unknown>[];

  const table = useReactTable({
    data: webhooks,
    columns,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    pageCount: pagination.totalPages,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (webhooks.length === 0) {
    return (
      <div className="py-12 text-center text-text-disabled">
        No webhook events found matching your criteria.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="overflow-hidden rounded-lg border border-border bg-card">
        <table className="w-full">
          <thead className="bg-background-secondary">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-text-tertiary"
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-border">
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="hover:bg-background-secondary/50 transition-colors"
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="whitespace-nowrap px-6 py-4">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-text-tertiary">
          Showing {(pagination.page - 1) * pagination.limit + 1} to{' '}
          {Math.min(pagination.page * pagination.limit, pagination.total)} of{' '}
          {pagination.total} events
        </p>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(pagination.page - 1)}
            disabled={pagination.page <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>

          <span className="text-sm text-text-secondary">
            Page {pagination.page} of {pagination.totalPages}
          </span>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(pagination.page + 1)}
            disabled={pagination.page >= pagination.totalPages}
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
