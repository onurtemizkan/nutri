'use client';

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Loader2 } from 'lucide-react';
import type { AnalyticsOverview } from '@/lib/hooks/useAnalytics';

interface DistributionChartProps {
  data: AnalyticsOverview['activeSubscribers'] | undefined;
  isLoading: boolean;
}

const COLORS = {
  proMonthly: '#22c55e',
  proYearly: '#3b82f6',
  trial: '#f59e0b',
};

export function DistributionChart({ data, isLoading }: DistributionChartProps) {
  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!data || data.total === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-text-disabled">
        No subscription data available
      </div>
    );
  }

  const chartData = [
    { name: 'Pro Monthly', value: data.proMonthly, color: COLORS.proMonthly },
    { name: 'Pro Yearly', value: data.proYearly, color: COLORS.proYearly },
    { name: 'Trial', value: data.trial, color: COLORS.trial },
  ].filter((item) => item.value > 0);

  return (
    <ResponsiveContainer width="100%" height={256}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={80}
          paddingAngle={5}
          dataKey="value"
          label={({ name, percent }) =>
            `${name}: ${(percent * 100).toFixed(0)}%`
          }
          labelLine={false}
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip
          formatter={(value: number) => [value.toLocaleString(), 'Subscribers']}
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
          }}
          labelStyle={{ color: 'hsl(var(--text-primary))' }}
        />
        <Legend
          wrapperStyle={{ color: 'hsl(var(--text-secondary))' }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
