import prisma from '../config/database';
import { logger } from '../config/logger';

// Constants for pricing (these should come from config in production)
const MONTHLY_PRICE = 9.99;
const YEARLY_PRICE = 79.99;

// Types
export interface SubscriptionMetrics {
  mrr: number;
  activeSubscribers: {
    total: number;
    proMonthly: number;
    proYearly: number;
    trial: number;
  };
  newSubscriptions: {
    today: number;
    week: number;
    month: number;
  };
  churn: {
    rate: number;
    count: number;
  };
  trials: {
    active: number;
    conversionRate: number;
  };
}

export interface TimeSeriesDataPoint {
  date: string;
  value: number;
}

/**
 * Get comprehensive subscription analytics
 */
export async function getSubscriptionMetrics(): Promise<SubscriptionMetrics> {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  // Execute all queries in parallel
  const [
    proMonthly,
    proYearly,
    trials,
    newToday,
    newWeek,
    newMonth,
    expiredThisMonth,
    activeAtMonthStart,
    trialsStarted30DaysAgo,
    trialsConvertedToPaid,
  ] = await Promise.all([
    // Active PRO monthly subscribers
    prisma.user.count({
      where: {
        subscriptionTier: 'PRO',
        subscriptionBillingCycle: 'MONTHLY',
        subscriptionEndDate: { gt: now },
      },
    }),

    // Active PRO yearly subscribers
    prisma.user.count({
      where: {
        subscriptionTier: 'PRO',
        subscriptionBillingCycle: 'ANNUAL',
        subscriptionEndDate: { gt: now },
      },
    }),

    // Active trial users
    prisma.user.count({
      where: {
        subscriptionTier: 'PRO_TRIAL',
        subscriptionEndDate: { gt: now },
      },
    }),

    // New subscriptions today
    prisma.user.count({
      where: {
        subscriptionTier: { in: ['PRO', 'PRO_TRIAL'] },
        subscriptionStartDate: { gte: today },
      },
    }),

    // New subscriptions this week
    prisma.user.count({
      where: {
        subscriptionTier: { in: ['PRO', 'PRO_TRIAL'] },
        subscriptionStartDate: { gte: weekAgo },
      },
    }),

    // New subscriptions this month
    prisma.user.count({
      where: {
        subscriptionTier: { in: ['PRO', 'PRO_TRIAL'] },
        subscriptionStartDate: { gte: monthAgo },
      },
    }),

    // Subscriptions expired this month (churn)
    prisma.user.count({
      where: {
        subscriptionTier: 'FREE',
        subscriptionEndDate: {
          gte: monthAgo,
          lt: now,
        },
      },
    }),

    // Active subscribers at start of month (for churn rate calculation)
    prisma.user.count({
      where: {
        subscriptionTier: { in: ['PRO', 'PRO_TRIAL'] },
        subscriptionStartDate: { lt: monthAgo },
        OR: [
          { subscriptionEndDate: { gt: monthAgo } },
          { subscriptionEndDate: null },
        ],
      },
    }),

    // Trials started 30+ days ago (for conversion calculation)
    prisma.user.count({
      where: {
        subscriptionStartDate: { lte: monthAgo },
        // Could track original tier if we had a field for it
      },
    }),

    // Users who were trials and converted to PRO
    // This is an approximation - in production you'd track tier transitions
    prisma.user.count({
      where: {
        subscriptionTier: 'PRO',
        subscriptionStartDate: { lte: monthAgo },
      },
    }),
  ]);

  // Calculate MRR
  // PRO monthly subscriptions + PRO yearly (divided by 12)
  // Add manual grants with price = 0
  const proMonthlyWithPrice = await prisma.user.count({
    where: {
      subscriptionTier: 'PRO',
      subscriptionBillingCycle: 'MONTHLY',
      subscriptionEndDate: { gt: now },
      subscriptionPrice: { gt: 0 },
    },
  });

  const proYearlyWithPrice = await prisma.user.count({
    where: {
      subscriptionTier: 'PRO',
      subscriptionBillingCycle: 'ANNUAL',
      subscriptionEndDate: { gt: now },
      subscriptionPrice: { gt: 0 },
    },
  });

  const mrr =
    proMonthlyWithPrice * MONTHLY_PRICE +
    proYearlyWithPrice * (YEARLY_PRICE / 12);

  // Calculate churn rate
  const churnRate =
    activeAtMonthStart > 0
      ? (expiredThisMonth / activeAtMonthStart) * 100
      : 0;

  // Calculate trial conversion rate
  const conversionRate =
    trialsStarted30DaysAgo > 0
      ? (trialsConvertedToPaid / trialsStarted30DaysAgo) * 100
      : 0;

  const total = proMonthly + proYearly + trials;

  logger.debug(
    {
      total,
      proMonthly,
      proYearly,
      trials,
      mrr,
      churnRate,
      conversionRate,
    },
    'Analytics metrics calculated'
  );

  return {
    mrr: Math.round(mrr * 100) / 100, // Round to 2 decimal places
    activeSubscribers: {
      total,
      proMonthly,
      proYearly,
      trial: trials,
    },
    newSubscriptions: {
      today: newToday,
      week: newWeek,
      month: newMonth,
    },
    churn: {
      rate: Math.round(churnRate * 100) / 100,
      count: expiredThisMonth,
    },
    trials: {
      active: trials,
      conversionRate: Math.round(conversionRate * 100) / 100,
    },
  };
}

/**
 * Get subscribers count over time (daily for last N days)
 */
export async function getSubscribersOverTime(
  days: number = 30
): Promise<TimeSeriesDataPoint[]> {
  const now = new Date();
  const result: TimeSeriesDataPoint[] = [];

  // Generate data points for each day
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    date.setHours(23, 59, 59, 999); // End of day

    const dateStr = date.toISOString().split('T')[0];

    // Count active subscribers as of that date
    const count = await prisma.user.count({
      where: {
        subscriptionTier: { in: ['PRO', 'PRO_TRIAL'] },
        subscriptionStartDate: { lte: date },
        OR: [
          { subscriptionEndDate: { gt: date } },
          { subscriptionEndDate: null },
        ],
      },
    });

    result.push({
      date: dateStr,
      value: count,
    });
  }

  return result;
}

/**
 * Get revenue over time (monthly for last N months)
 */
export async function getRevenueOverTime(
  months: number = 12
): Promise<TimeSeriesDataPoint[]> {
  const now = new Date();
  const result: TimeSeriesDataPoint[] = [];

  for (let i = months - 1; i >= 0; i--) {
    const date = new Date(now.getFullYear(), now.getMonth() - i, 1);
    const endOfMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0);
    const monthStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

    // Count paying subscribers in that month
    const [monthlyCount, yearlyCount] = await Promise.all([
      prisma.user.count({
        where: {
          subscriptionTier: 'PRO',
          subscriptionBillingCycle: 'MONTHLY',
          subscriptionStartDate: { lte: endOfMonth },
          subscriptionEndDate: { gt: date },
          subscriptionPrice: { gt: 0 },
        },
      }),
      prisma.user.count({
        where: {
          subscriptionTier: 'PRO',
          subscriptionBillingCycle: 'ANNUAL',
          subscriptionStartDate: { lte: endOfMonth },
          subscriptionEndDate: { gt: date },
          subscriptionPrice: { gt: 0 },
        },
      }),
    ]);

    const revenue =
      monthlyCount * MONTHLY_PRICE + yearlyCount * (YEARLY_PRICE / 12);

    result.push({
      date: monthStr,
      value: Math.round(revenue * 100) / 100,
    });
  }

  return result;
}
