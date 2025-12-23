'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Loader2 } from 'lucide-react';
import type { SubscriberDataPoint } from '@/lib/hooks/useAnalytics';

interface SubscriberChartProps {
  data: SubscriberDataPoint[] | undefined;
  isLoading: boolean;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function SubscriberChart({ data, isLoading }: SubscriberChartProps) {
  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-text-disabled">
        No subscriber data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={256}>
      <LineChart
        data={data}
        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDate}
          stroke="hsl(var(--text-tertiary))"
          fontSize={12}
        />
        <YAxis
          stroke="hsl(var(--text-tertiary))"
          fontSize={12}
        />
        <Tooltip
          formatter={(value: number) => [value.toLocaleString(), 'Subscribers']}
          labelFormatter={(label) => formatDate(label as string)}
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
          }}
          labelStyle={{ color: 'hsl(var(--text-primary))' }}
        />
        <Line
          type="monotone"
          dataKey="count"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6, fill: '#3b82f6' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
