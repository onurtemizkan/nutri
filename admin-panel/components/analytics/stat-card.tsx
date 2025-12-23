'use client';

import { ArrowUpRight, ArrowDownRight, type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  className?: string;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  className,
}: StatCardProps) {
  return (
    <div
      className={cn(
        'rounded-lg border border-border bg-card p-6',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium text-text-tertiary">{title}</p>
          <p className="text-3xl font-bold text-text-primary">{value}</p>
          {subtitle && (
            <p className="text-sm text-text-tertiary">{subtitle}</p>
          )}
        </div>
        {Icon && (
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-6 w-6 text-primary" />
          </div>
        )}
      </div>
      {trend && (
        <div
          className={cn(
            'mt-4 flex items-center gap-1 text-sm font-medium',
            trend.isPositive ? 'text-green-500' : 'text-red-500'
          )}
        >
          {trend.isPositive ? (
            <ArrowUpRight className="h-4 w-4" />
          ) : (
            <ArrowDownRight className="h-4 w-4" />
          )}
          <span>{Math.abs(trend.value)}%</span>
          <span className="text-text-tertiary font-normal">vs last period</span>
        </div>
      )}
    </div>
  );
}
