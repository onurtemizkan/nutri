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
import { formatDate } from '@/lib/utils';
import type { UserListItem, PaginationMeta } from '@/lib/types';

interface UserTableProps {
  users: UserListItem[];
  pagination: PaginationMeta;
  isLoading: boolean;
  onPageChange: (page: number) => void;
}

const columnHelper = createColumnHelper<UserListItem>();

const columns: ColumnDef<UserListItem, unknown>[] = [
  columnHelper.accessor('email', {
    header: 'Email',
    cell: (info) => (
      <span className="font-medium text-text-primary">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor('name', {
    header: 'Name',
    cell: (info) => (
      <span className="text-text-secondary">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor('subscriptionTier', {
    header: 'Subscription',
    cell: (info) => {
      const tier = info.getValue();
      const row = info.row.original;
      const isExpired =
        row.subscriptionEndDate &&
        new Date(row.subscriptionEndDate) < new Date();

      if (isExpired && tier !== 'FREE') {
        return (
          <Badge variant="warning">
            Expired
          </Badge>
        );
      }

      return (
        <Badge variant={getSubscriptionBadgeVariant(tier)}>
          {getSubscriptionBadgeLabel(tier)}
          {row.subscriptionBillingCycle && tier === 'PRO' && (
            <span className="ml-1 opacity-75">
              ({row.subscriptionBillingCycle === 'MONTHLY' ? 'M' : 'Y'})
            </span>
          )}
        </Badge>
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
          {isExpired ? 'Expired' : `${daysLeft}d left`}
        </span>
      );
    },
  }),
  columnHelper.accessor('createdAt', {
    header: 'Created',
    cell: (info) => (
      <span className="text-text-tertiary text-sm">
        {formatDate(info.getValue(), { dateStyle: 'medium' })}
      </span>
    ),
  }),
  columnHelper.display({
    id: 'actions',
    header: 'Actions',
    cell: (info) => (
      <Link href={`/dashboard/users/${info.row.original.id}`}>
        <Button variant="ghost" size="sm" className="gap-1.5">
          <Eye className="h-4 w-4" />
          View
        </Button>
      </Link>
    ),
  }),
] as ColumnDef<UserListItem, unknown>[];

export function UserTable({
  users,
  pagination,
  isLoading,
  onPageChange,
}: UserTableProps) {
  const table = useReactTable({
    data: users,
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

  if (users.length === 0) {
    return (
      <div className="py-12 text-center text-text-disabled">
        No users found matching your criteria.
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
          {pagination.total} users
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
