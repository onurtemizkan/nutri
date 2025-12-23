'use client';

import { Loader2 } from 'lucide-react';
import type { AnalyticsOverview } from '@/lib/hooks/useAnalytics';

interface ConversionFunnelProps {
  data: AnalyticsOverview | undefined;
  isLoading: boolean;
}

export function ConversionFunnel({ data, isLoading }: ConversionFunnelProps) {
  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-64 items-center justify-center text-text-disabled">
        No conversion data available
      </div>
    );
  }

  const conversionRate = data.trials.conversionRate;
  const activeTrials = data.trials.active;
  // Estimate converted from conversion rate (approximate)
  const totalSubscribers = data.activeSubscribers.total;

  const stages = [
    {
      label: 'Active Trials',
      value: activeTrials,
      color: 'bg-yellow-500',
      width: '100%',
    },
    {
      label: 'Conversion Rate',
      value: `${conversionRate.toFixed(1)}%`,
      color: 'bg-blue-500',
      width: `${Math.max(conversionRate, 10)}%`,
    },
    {
      label: 'Paid Subscribers',
      value: totalSubscribers - activeTrials,
      color: 'bg-green-500',
      width: `${Math.max((1 - activeTrials / (totalSubscribers || 1)) * 100, 10)}%`,
    },
  ];

  return (
    <div className="space-y-6 py-4">
      {stages.map((stage) => (
        <div key={stage.label} className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-tertiary">{stage.label}</span>
            <span className="font-medium text-text-primary">{stage.value}</span>
          </div>
          <div className="h-8 w-full rounded-lg bg-background-secondary overflow-hidden">
            <div
              className={`h-full ${stage.color} rounded-lg transition-all duration-500`}
              style={{ width: stage.width }}
            />
          </div>
        </div>
      ))}

      {/* Additional metrics */}
      <div className="mt-6 grid grid-cols-2 gap-4 pt-4 border-t border-border">
        <div>
          <p className="text-sm text-text-tertiary">Churn Rate</p>
          <p className="text-xl font-bold text-text-primary">
            {data.churn.rate.toFixed(1)}%
          </p>
          <p className="text-xs text-text-tertiary">
            {data.churn.count} churned this period
          </p>
        </div>
        <div>
          <p className="text-sm text-text-tertiary">New This Month</p>
          <p className="text-xl font-bold text-text-primary">
            {data.newSubscriptions.month}
          </p>
          <p className="text-xs text-text-tertiary">
            {data.newSubscriptions.today} today
          </p>
        </div>
      </div>
    </div>
  );
}
