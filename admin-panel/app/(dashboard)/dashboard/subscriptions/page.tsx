'use client';

import { useState } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { useSubscriptions, useLookupSubscription } from '@/lib/hooks/useSubscriptions';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { SubscriptionTable } from '@/components/subscriptions/subscription-table';
import { Badge, getSubscriptionBadgeVariant, getSubscriptionBadgeLabel } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

type SubscriptionStatusFilter = 'active' | 'trial' | 'expired' | 'none' | '';

export default function SubscriptionsPage() {
  const [status, setStatus] = useState<SubscriptionStatusFilter>('');
  const [page, setPage] = useState(1);
  const [transactionSearch, setTransactionSearch] = useState('');

  // Debounce transaction search
  const debouncedTransactionSearch = useDebounce(transactionSearch, 500);

  const { data, isLoading, isError, error } = useSubscriptions({
    status: status || undefined,
    page,
    limit: 20,
  });

  const {
    data: lookupResult,
    isLoading: isLookupLoading,
    isError: isLookupError,
  } = useLookupSubscription(
    debouncedTransactionSearch.length > 5 ? debouncedTransactionSearch : null
  );

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatus(e.target.value as SubscriptionStatusFilter);
    setPage(1);
  };

  const clearTransactionSearch = () => {
    setTransactionSearch('');
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">Subscriptions</h2>
          <p className="mt-1 text-text-tertiary">
            Manage subscriptions and manual operations
          </p>
        </div>
        {data?.pagination && (
          <div className="text-sm text-text-tertiary">
            {data.pagination.total} total subscriptions
          </div>
        )}
      </div>

      {/* Transaction ID Lookup */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="mb-3 font-medium text-text-primary">
          Lookup by Transaction ID
        </h3>
        <div className="flex gap-4">
          <div className="relative flex-1 max-w-lg">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-disabled" />
            <input
              type="text"
              value={transactionSearch}
              onChange={(e) => setTransactionSearch(e.target.value)}
              placeholder="Enter Apple originalTransactionId..."
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
        </div>

        {/* Lookup Result */}
        {isLookupLoading && debouncedTransactionSearch.length > 5 && (
          <div className="mt-4 flex items-center gap-2 text-text-tertiary">
            <Loader2 className="h-4 w-4 animate-spin" />
            Searching...
          </div>
        )}

        {isLookupError && debouncedTransactionSearch.length > 5 && (
          <div className="mt-4 text-text-tertiary">
            No subscription found for this transaction ID
          </div>
        )}

        {lookupResult && (
          <div className="mt-4 rounded-lg border border-border bg-background-secondary p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-text-primary">
                  {lookupResult.name}
                </p>
                <p className="text-sm text-text-tertiary">{lookupResult.email}</p>
              </div>
              <div className="flex items-center gap-3">
                <Badge
                  variant={getSubscriptionBadgeVariant(
                    lookupResult.subscriptionTier
                  )}
                >
                  {getSubscriptionBadgeLabel(lookupResult.subscriptionTier)}
                </Badge>
                <Link href={`/dashboard/subscriptions/${lookupResult.id}`}>
                  <Button size="sm">View Details</Button>
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <select
          value={status}
          onChange={handleStatusChange}
          className="rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
        >
          <option value="">All Subscriptions</option>
          <option value="active">Active (Pro)</option>
          <option value="trial">Trial</option>
          <option value="expired">Expired</option>
          <option value="none">Free</option>
        </select>
      </div>

      {/* Error State */}
      {isError && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-red-500">
          Failed to load subscriptions: {error?.message || 'Unknown error'}
        </div>
      )}

      {/* Subscriptions Table */}
      <SubscriptionTable
        subscriptions={data?.subscriptions || []}
        pagination={
          data?.pagination || { page: 1, limit: 20, total: 0, totalPages: 0 }
        }
        isLoading={isLoading}
        onPageChange={setPage}
      />
    </div>
  );
}
