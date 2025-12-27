'use client';

import { useState } from 'react';
import { BarChart3, Mail, MousePointer, Users, TrendingUp, TrendingDown, RefreshCcw, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useEmailAnalytics, useEmailSubscriberStats } from '@/lib/hooks/useEmailAnalytics';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  suffix?: string;
}

function StatCard({ title, value, change, icon, suffix }: StatCardProps) {
  return (
    <div className="bg-background-secondary border border-border rounded-lg p-5">
      <div className="flex items-center justify-between">
        <div className="text-text-muted">{icon}</div>
        {change !== undefined && (
          <Badge variant={change >= 0 ? 'success' : 'danger'} className="text-xs gap-1">
            {change >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(change).toFixed(1)}%
          </Badge>
        )}
      </div>
      <p className="mt-3 text-2xl font-semibold text-text-primary">
        {typeof value === 'number' ? value.toLocaleString() : value}
        {suffix && <span className="text-lg text-text-muted">{suffix}</span>}
      </p>
      <p className="mt-1 text-sm text-text-muted">{title}</p>
    </div>
  );
}

export default function EmailAnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'7' | '30' | '90'>('30');

  const {
    data: analytics,
    isLoading: analyticsLoading,
    refetch: refetchAnalytics,
  } = useEmailAnalytics({ days: parseInt(timeRange) });

  const {
    data: subscriberStats,
    isLoading: statsLoading,
    refetch: refetchStats,
  } = useEmailSubscriberStats();

  const isLoading = analyticsLoading || statsLoading;

  const handleRefresh = () => {
    refetchAnalytics();
    refetchStats();
  };

  // Use pre-calculated rates from API
  const overview = analytics?.overview;
  const openRate = parseFloat(overview?.openRate || '0');
  const clickRate = parseFloat(overview?.clickRate || '0');
  const bounceRate = parseFloat(overview?.bounceRate || '0');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Email Analytics</h1>
          <p className="text-text-secondary mt-1">
            Track email performance and subscriber engagement
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as '7' | '30' | '90')}
            className="px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </select>
          <Button variant="outline" onClick={handleRefresh} disabled={isLoading}>
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCcw className="w-4 h-4" />
            )}
          </Button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      )}

      {/* Stats Grid */}
      {!isLoading && (
        <>
          <div className="grid grid-cols-4 gap-4">
            <StatCard
              title="Emails Sent"
              value={overview?.totalEmails || 0}
              icon={<Mail className="w-5 h-5" />}
            />
            <StatCard
              title="Open Rate"
              value={openRate.toFixed(1)}
              suffix="%"
              icon={<BarChart3 className="w-5 h-5" />}
            />
            <StatCard
              title="Click Rate"
              value={clickRate.toFixed(1)}
              suffix="%"
              icon={<MousePointer className="w-5 h-5" />}
            />
            <StatCard
              title="Bounce Rate"
              value={bounceRate.toFixed(1)}
              suffix="%"
              icon={<TrendingDown className="w-5 h-5" />}
            />
          </div>

          {/* Subscriber Stats */}
          <div className="bg-background-secondary border border-border rounded-lg p-6">
            <h2 className="text-lg font-medium text-text-primary mb-4 flex items-center gap-2">
              <Users className="w-5 h-5" />
              Subscriber Overview
            </h2>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <p className="text-3xl font-semibold text-text-primary">
                  {subscriberStats?.totalSubscribers?.toLocaleString() || 0}
                </p>
                <p className="text-sm text-text-muted mt-1">Total Subscribers</p>
              </div>
              <div>
                <p className="text-3xl font-semibold text-green-500">
                  {subscriberStats?.marketingOptIn?.toLocaleString() || 0}
                </p>
                <p className="text-sm text-text-muted mt-1">Marketing Opted-In</p>
              </div>
              <div>
                <p className="text-3xl font-semibold text-blue-500">
                  {subscriberStats?.doubleOptInConfirmed?.toLocaleString() || 0}
                </p>
                <p className="text-sm text-text-muted mt-1">Double Opt-In Confirmed</p>
              </div>
              <div>
                <p className="text-3xl font-semibold text-yellow-500">
                  {subscriberStats?.unsubscribed?.toLocaleString() || 0}
                </p>
                <p className="text-sm text-text-muted mt-1">Unsubscribed</p>
              </div>
            </div>
          </div>

          {/* Email Breakdown */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-background-secondary border border-border rounded-lg p-6">
              <h2 className="text-lg font-medium text-text-primary mb-4">
                Delivery Status
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Delivered</span>
                  <span className="text-text-primary font-medium">
                    {(overview?.delivered || 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Bounced</span>
                  <span className="text-red-400 font-medium">
                    {(overview?.bounced || 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Complaints</span>
                  <span className="text-yellow-400 font-medium">
                    {(overview?.complained || 0).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-background-secondary border border-border rounded-lg p-6">
              <h2 className="text-lg font-medium text-text-primary mb-4">
                Engagement
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Opened</span>
                  <span className="text-text-primary font-medium">
                    {(overview?.opened || 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Clicked</span>
                  <span className="text-primary font-medium">
                    {(overview?.clicked || 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-text-secondary">Click-to-Open Rate</span>
                  <span className="text-primary font-medium">
                    {overview?.opened && overview.opened > 0
                      ? ((overview.clicked / overview.opened) * 100).toFixed(1)
                      : '0.0'}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Help Text */}
          <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400 text-sm">
            <p>
              <strong>Tip:</strong> A healthy email program typically has an open rate above 20% and a click rate above 2.5%.
              Keep bounce rates under 2% to maintain good sender reputation.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
