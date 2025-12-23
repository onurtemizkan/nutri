'use client';

import { useState } from 'react';
import { DollarSign, Users, TrendingUp, Repeat, Loader2 } from 'lucide-react';
import {
  useAnalyticsOverview,
  useSubscribersOverTime,
  useRevenueOverTime,
} from '@/lib/hooks/useAnalytics';
import { StatCard } from '@/components/analytics/stat-card';
import { RevenueChart } from '@/components/analytics/revenue-chart';
import { SubscriberChart } from '@/components/analytics/subscriber-chart';
import { DistributionChart } from '@/components/analytics/distribution-chart';
import { ConversionFunnel } from '@/components/analytics/conversion-funnel';

type TimeRange = 7 | 30 | 90 | 365;

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>(30);

  const { data: overview, isLoading: isLoadingOverview } = useAnalyticsOverview();
  const { data: subscribersData, isLoading: isLoadingSubscribers } =
    useSubscribersOverTime(timeRange);
  const { data: revenueData, isLoading: isLoadingRevenue } =
    useRevenueOverTime(12);

  const handleTimeRangeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setTimeRange(Number(e.target.value) as TimeRange);
  };

  const isLoading = isLoadingOverview;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">Analytics</h2>
          <p className="mt-1 text-text-tertiary">
            Subscription metrics and revenue insights
          </p>
        </div>
        <select
          value={timeRange}
          onChange={handleTimeRangeChange}
          className="rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
        >
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
          <option value={365}>Last year</option>
        </select>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}

      {/* Stats Cards */}
      {overview && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Monthly Recurring Revenue"
            value={formatCurrency(overview.mrr)}
            icon={DollarSign}
          />
          <StatCard
            title="Active Subscribers"
            value={overview.activeSubscribers.total.toLocaleString()}
            subtitle={`${overview.activeSubscribers.proMonthly} monthly, ${overview.activeSubscribers.proYearly} yearly`}
            icon={Users}
          />
          <StatCard
            title="New This Month"
            value={overview.newSubscriptions.month.toLocaleString()}
            subtitle={`${overview.newSubscriptions.today} today, ${overview.newSubscriptions.week} this week`}
            icon={TrendingUp}
          />
          <StatCard
            title="Trial Conversion"
            value={`${overview.trials.conversionRate.toFixed(1)}%`}
            subtitle={`${overview.trials.active} active trials`}
            icon={Repeat}
          />
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Revenue Chart */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 text-lg font-semibold text-text-primary">
            Monthly Recurring Revenue (MRR)
          </h3>
          <RevenueChart data={revenueData} isLoading={isLoadingRevenue} />
        </div>

        {/* Subscriber Growth */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 text-lg font-semibold text-text-primary">
            Subscriber Growth
          </h3>
          <SubscriberChart
            data={subscribersData}
            isLoading={isLoadingSubscribers}
          />
        </div>

        {/* Subscription Distribution */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 text-lg font-semibold text-text-primary">
            Subscription Distribution
          </h3>
          <DistributionChart
            data={overview?.activeSubscribers}
            isLoading={isLoadingOverview}
          />
        </div>

        {/* Conversion Funnel */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 text-lg font-semibold text-text-primary">
            Trial Conversion & Metrics
          </h3>
          <ConversionFunnel data={overview} isLoading={isLoadingOverview} />
        </div>
      </div>

      {/* Key Metrics Summary */}
      {overview && (
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 text-lg font-semibold text-text-primary">
            Key Metrics Summary
          </h3>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div>
              <p className="text-sm text-text-tertiary">Churn Rate</p>
              <p className="text-2xl font-bold text-text-primary">
                {overview.churn.rate.toFixed(1)}%
              </p>
              <p className="text-xs text-text-tertiary">
                {overview.churn.count} churned
              </p>
            </div>
            <div>
              <p className="text-sm text-text-tertiary">Active Trials</p>
              <p className="text-2xl font-bold text-text-primary">
                {overview.trials.active}
              </p>
              <p className="text-xs text-text-tertiary">
                {overview.trials.conversionRate.toFixed(1)}% convert
              </p>
            </div>
            <div>
              <p className="text-sm text-text-tertiary">Pro Monthly</p>
              <p className="text-2xl font-bold text-text-primary">
                {overview.activeSubscribers.proMonthly}
              </p>
              <p className="text-xs text-text-tertiary">subscribers</p>
            </div>
            <div>
              <p className="text-sm text-text-tertiary">Pro Yearly</p>
              <p className="text-2xl font-bold text-text-primary">
                {overview.activeSubscribers.proYearly}
              </p>
              <p className="text-xs text-text-tertiary">subscribers</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
