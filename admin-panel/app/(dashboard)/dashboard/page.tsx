'use client';

import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';

export default function DashboardPage() {
  const { data: analytics, isLoading, error } = useQuery({
    queryKey: ['analytics', 'overview'],
    queryFn: () => adminApi.getAnalyticsOverview(),
  });

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const metrics = analytics
    ? [
        {
          name: 'Active Subscribers',
          value: formatNumber(analytics.activeSubscribers?.total || 0),
          detail: `${analytics.activeSubscribers?.proMonthly || 0} monthly, ${analytics.activeSubscribers?.proYearly || 0} yearly`,
        },
        {
          name: 'Monthly Revenue',
          value: formatCurrency(analytics.mrr || 0),
          detail: 'Monthly recurring revenue',
        },
        {
          name: 'New This Month',
          value: formatNumber(analytics.newSubscriptions?.month || 0),
          detail: `${analytics.newSubscriptions?.today || 0} today, ${analytics.newSubscriptions?.week || 0} this week`,
        },
        {
          name: 'Churn Rate',
          value: `${(analytics.churn?.rate || 0).toFixed(1)}%`,
          detail: `${analytics.churn?.count || 0} churned this month`,
        },
      ]
    : [];

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">Overview</h2>
          <p className="text-text-tertiary mt-1">Subscription metrics and analytics</p>
        </div>
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-red-500">
          Failed to load analytics: {(error as Error).message}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">Overview</h2>
        <p className="text-text-tertiary mt-1">
          Subscription metrics and analytics
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="bg-card rounded-lg border border-border p-6 animate-pulse"
              >
                <div className="h-4 w-24 bg-background-tertiary rounded" />
                <div className="mt-2 h-8 w-32 bg-background-tertiary rounded" />
                <div className="mt-2 h-3 w-20 bg-background-tertiary rounded" />
              </div>
            ))
          : metrics.map((metric) => (
              <div
                key={metric.name}
                className="bg-card rounded-lg border border-border p-6"
              >
                <p className="text-sm font-medium text-text-tertiary">
                  {metric.name}
                </p>
                <p className="mt-2 text-3xl font-bold text-text-primary">
                  {metric.value}
                </p>
                <p className="mt-2 text-sm text-text-tertiary">{metric.detail}</p>
              </div>
            ))}
      </div>

      {/* Charts placeholder */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg border border-border p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Subscribers Over Time
          </h3>
          <div className="h-64 flex items-center justify-center text-text-disabled">
            {isLoading ? 'Loading...' : 'No subscription data yet'}
          </div>
        </div>

        <div className="bg-card rounded-lg border border-border p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Revenue Over Time
          </h3>
          <div className="h-64 flex items-center justify-center text-text-disabled">
            {isLoading ? 'Loading...' : 'No revenue data yet'}
          </div>
        </div>
      </div>

      {/* Trial stats */}
      {analytics && (
        <div className="bg-card rounded-lg border border-border p-6">
          <h3 className="text-lg font-semibold text-text-primary mb-4">
            Trial Performance
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-text-tertiary">Active Trials</p>
              <p className="text-2xl font-bold text-text-primary">
                {formatNumber(analytics.trials?.active || 0)}
              </p>
            </div>
            <div>
              <p className="text-sm text-text-tertiary">Conversion Rate</p>
              <p className="text-2xl font-bold text-text-primary">
                {(analytics.trials?.conversionRate || 0).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
