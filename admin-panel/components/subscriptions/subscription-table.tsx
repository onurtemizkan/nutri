'use client';

import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
} from '@tanstack/react-table';
import { ChevronLeft, ChevronRight, Eye, Loader2 } from 'lucide-react';
import Link from 'next/link';
import { Badge, getSubscriptionBadgeVariant, getSubscriptionBadgeLabel } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { SubscriptionActions } from '@/components/subscriptions/subscription-modals';
import { formatDate } from '@/lib/utils';
import type { PaginationMeta } from '@/lib/types';
import type { SubscriptionListItem } from '@/lib/hooks/useSubscriptions';

interface SubscriptionTableProps {
  subscriptions: SubscriptionListItem[];
  pagination: PaginationMeta;
  isLoading: boolean;
  onPageChange: (page: number) => void;
}

const columnHelper = createColumnHelper<SubscriptionListItem>();

const columns: ColumnDef<SubscriptionListItem, unknown>[] = [
  columnHelper.accessor('email', {
    header: 'User',
    cell: (info) => (
      <div>
        <span className="font-medium text-text-primary">
          {info.row.original.name}
        </span>
        <br />
        <span className="text-sm text-text-tertiary">{info.getValue()}</span>
      </div>
    ),
  }),
  columnHelper.accessor('subscriptionTier', {
    header: 'Tier',
    cell: (info) => {
      const tier = info.getValue();
      const row = info.row.original;

      return (
        <Badge variant={getSubscriptionBadgeVariant(tier)}>
          {getSubscriptionBadgeLabel(tier)}
          {row.subscriptionBillingCycle && tier === 'PRO' && (
            <span className="ml-1 opacity-75">
              ({row.subscriptionBillingCycle === 'MONTHLY' ? 'Monthly' : 'Annual'})
            </span>
          )}
        </Badge>
      );
    },
  }),
  columnHelper.accessor('subscriptionStartDate', {
    header: 'Started',
    cell: (info) => {
      const date = info.getValue();
      if (!date) return <span className="text-text-disabled">-</span>;
      return (
        <span className="text-text-secondary text-sm">
          {formatDate(date, { dateStyle: 'medium' })}
        </span>
      );
    },
  }),
  columnHelper.accessor('subscriptionEndDate', {
    header: 'Expires',
    cell: (info) => {
      const date = info.getValue();
      if (!date) return <span className="text-text-disabled">-</span>;

      const expDate = new Date(date);
      const now = new Date();
      const isExpired = expDate < now;
      const daysLeft = Math.ceil(
        (expDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
      );

      return (
        <span
          className={
            isExpired
              ? 'text-red-500'
              : daysLeft <= 7
                ? 'text-yellow-500'
                : 'text-text-secondary'
          }
        >
          {formatDate(date, { dateStyle: 'medium' })}
          {!isExpired && <span className="text-xs ml-1">({daysLeft}d)</span>}
        </span>
      );
    },
  }),
  columnHelper.accessor('subscriptionPrice', {
    header: 'Price',
    cell: (info) => {
      const price = info.getValue();
      if (price === null) return <span className="text-text-disabled">-</span>;
      if (price === 0) return <span className="text-text-tertiary">Free</span>;
      return (
        <span className="text-text-secondary">${price.toFixed(2)}</span>
      );
    },
  }),
  columnHelper.display({
    id: 'actions',
    header: 'Actions',
    cell: (info) => (
      <div className="flex items-center gap-2">
        <Link href={`/dashboard/subscriptions/${info.row.original.id}`}>
          <Button variant="ghost" size="sm" className="gap-1.5">
            <Eye className="h-4 w-4" />
            View
          </Button>
        </Link>
        <SubscriptionActions
          userId={info.row.original.id}
          userEmail={info.row.original.email}
          currentTier={info.row.original.subscriptionTier}
        />
      </div>
    ),
  }),
] as ColumnDef<SubscriptionListItem, unknown>[];

export function SubscriptionTable({
  subscriptions,
  pagination,
  isLoading,
  onPageChange,
}: SubscriptionTableProps) {
  const table = useReactTable({
    data: subscriptions,
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

  if (subscriptions.length === 0) {
    return (
      <div className="py-12 text-center text-text-disabled">
        No subscriptions found matching your criteria.
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
          {pagination.total} subscriptions
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
