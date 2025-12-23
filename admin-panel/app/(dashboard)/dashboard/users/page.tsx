'use client';

import { useState } from 'react';
import { Search, X } from 'lucide-react';
import { useUsers } from '@/lib/hooks/useUsers';
import { useDebounce } from '@/lib/hooks/useDebounce';
import { UserTable } from '@/components/users/user-table';

type SubscriptionStatusFilter = 'active' | 'trial' | 'expired' | 'none' | '';

export default function UsersPage() {
  const [search, setSearch] = useState('');
  const [subscriptionStatus, setSubscriptionStatus] =
    useState<SubscriptionStatusFilter>('');
  const [page, setPage] = useState(1);

  // Debounce search to avoid too many API calls
  const debouncedSearch = useDebounce(search, 300);

  const { data, isLoading, isError, error } = useUsers({
    search: debouncedSearch,
    page,
    limit: 20,
    subscriptionStatus: subscriptionStatus || undefined,
  });

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value);
    setPage(1); // Reset to first page on search
  };

  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSubscriptionStatus(e.target.value as SubscriptionStatusFilter);
    setPage(1); // Reset to first page on filter change
  };

  const clearSearch = () => {
    setSearch('');
    setPage(1);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">Users</h2>
          <p className="mt-1 text-text-tertiary">
            Search and manage user accounts
          </p>
        </div>
        {data?.pagination && (
          <div className="text-sm text-text-tertiary">
            {data.pagination.total} total users
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-disabled" />
          <input
            type="text"
            value={search}
            onChange={handleSearchChange}
            placeholder="Search by email or name..."
            className="w-full rounded-md border border-border bg-background-elevated py-2 pl-10 pr-10 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
          {search && (
            <button
              onClick={clearSearch}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-disabled hover:text-text-secondary"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <select
          value={subscriptionStatus}
          onChange={handleStatusChange}
          className="rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
        >
          <option value="">All Statuses</option>
          <option value="active">Active (Pro)</option>
          <option value="trial">Trial</option>
          <option value="expired">Expired</option>
          <option value="none">Free</option>
        </select>
      </div>

      {/* Error State */}
      {isError && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-red-500">
          Failed to load users: {error?.message || 'Unknown error'}
        </div>
      )}

      {/* Users Table */}
      <UserTable
        users={data?.users || []}
        pagination={
          data?.pagination || { page: 1, limit: 20, total: 0, totalPages: 0 }
        }
        isLoading={isLoading}
        onPageChange={setPage}
      />
    </div>
  );
}
