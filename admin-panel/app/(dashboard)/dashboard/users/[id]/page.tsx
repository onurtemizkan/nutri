'use client';

import { useParams, useRouter } from 'next/navigation';
import { useUser } from '@/lib/hooks/useUsers';
import {
  ArrowLeft,
  Loader2,
  Mail,
  Apple,
  BarChart3,
  CreditCard,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Badge,
  getSubscriptionBadgeVariant,
  getSubscriptionBadgeLabel,
} from '@/components/ui/badge';
import { formatDate } from '@/lib/utils';
import { ExportDataButton } from '@/components/users/export-data-button';
import { DeleteUserModal } from '@/components/users/delete-user-modal';

export default function UserDetailPage() {
  const params = useParams();
  const router = useRouter();
  const userId = params.id as string;

  const { data: user, isLoading, isError, error } = useUser(userId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (isError || !user) {
    return (
      <div className="space-y-4">
        <Button
          variant="ghost"
          onClick={() => router.push('/dashboard/users')}
          className="gap-1.5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Users
        </Button>
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-6 text-red-500">
          {error?.message || 'User not found'}
        </div>
      </div>
    );
  }

  const isSubscriptionActive =
    user.subscriptionTier !== 'FREE' &&
    user.subscriptionEndDate &&
    new Date(user.subscriptionEndDate) > new Date();

  const daysRemaining = user.subscriptionEndDate
    ? Math.max(
        0,
        Math.ceil(
          (new Date(user.subscriptionEndDate).getTime() - Date.now()) /
            (1000 * 60 * 60 * 24)
        )
      )
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          onClick={() => router.push('/dashboard/users')}
          className="gap-1.5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <div className="flex-1">
          <h2 className="text-2xl font-bold text-text-primary">{user.name}</h2>
          <p className="mt-0.5 text-text-tertiary">{user.email}</p>
        </div>
        <Badge variant={getSubscriptionBadgeVariant(user.subscriptionTier)}>
          {getSubscriptionBadgeLabel(user.subscriptionTier)}
        </Badge>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* User Profile Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <Mail className="h-4 w-4" />
            Profile
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Email</dt>
              <dd className="text-text-primary">{user.email}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Name</dt>
              <dd className="text-text-primary">{user.name}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Joined</dt>
              <dd className="text-text-primary">
                {formatDate(user.createdAt, { dateStyle: 'medium' })}
              </dd>
            </div>
            {user.currentWeight && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Weight</dt>
                <dd className="text-text-primary">{user.currentWeight} kg</dd>
              </div>
            )}
            {user.goalWeight && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Goal Weight</dt>
                <dd className="text-text-primary">{user.goalWeight} kg</dd>
              </div>
            )}
            {user.height && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Height</dt>
                <dd className="text-text-primary">{user.height} cm</dd>
              </div>
            )}
          </dl>
        </div>

        {/* Subscription Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <CreditCard className="h-4 w-4" />
            Subscription
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Tier</dt>
              <dd>
                <Badge
                  variant={getSubscriptionBadgeVariant(user.subscriptionTier)}
                >
                  {getSubscriptionBadgeLabel(user.subscriptionTier)}
                </Badge>
              </dd>
            </div>
            {user.subscriptionBillingCycle && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Billing</dt>
                <dd className="text-text-primary">
                  {user.subscriptionBillingCycle === 'MONTHLY'
                    ? 'Monthly'
                    : 'Annual'}
                </dd>
              </div>
            )}
            {user.subscriptionStartDate && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Started</dt>
                <dd className="text-text-primary">
                  {formatDate(user.subscriptionStartDate, { dateStyle: 'medium' })}
                </dd>
              </div>
            )}
            {user.subscriptionEndDate && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">
                  {isSubscriptionActive ? 'Expires' : 'Expired'}
                </dt>
                <dd
                  className={
                    isSubscriptionActive
                      ? daysRemaining <= 7
                        ? 'text-yellow-500'
                        : 'text-text-primary'
                      : 'text-red-500'
                  }
                >
                  {formatDate(user.subscriptionEndDate, { dateStyle: 'medium' })}
                  {isSubscriptionActive && ` (${daysRemaining}d left)`}
                </dd>
              </div>
            )}
            {user.subscriptionPrice !== null && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Price</dt>
                <dd className="text-text-primary">
                  {user.subscriptionPrice === 0
                    ? 'Free (Manual Grant)'
                    : `$${user.subscriptionPrice.toFixed(2)}`}
                </dd>
              </div>
            )}
          </dl>
        </div>

        {/* Activity Stats Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <BarChart3 className="h-4 w-4" />
            Activity (Last 7 Days)
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Meals Logged</dt>
              <dd className="text-text-primary">{user.recentMeals || 0}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Health Metrics</dt>
              <dd className="text-text-primary">
                {user.recentHealthMetrics || 0}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Total Meals</dt>
              <dd className="text-text-primary">{user.mealsCount || 0}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Total Activities</dt>
              <dd className="text-text-primary">{user.activitiesCount || 0}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Total Health Metrics</dt>
              <dd className="text-text-primary">
                {user.healthMetricsCount || 0}
              </dd>
            </div>
          </dl>
        </div>

        {/* Nutrition Goals Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <Apple className="h-4 w-4" />
            Nutrition Goals
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Calories</dt>
              <dd className="text-text-primary">{user.goalCalories} kcal</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Protein</dt>
              <dd className="text-text-primary">{user.goalProtein}g</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Carbs</dt>
              <dd className="text-text-primary">{user.goalCarbs}g</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Fat</dt>
              <dd className="text-text-primary">{user.goalFat}g</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Activity Level</dt>
              <dd className="text-text-primary capitalize">
                {user.activityLevel}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3 border-t border-border pt-6">
        <ExportDataButton userId={userId} userName={user.name} />
        <DeleteUserModal
          userId={userId}
          userEmail={user.email}
          userName={user.name}
        />
      </div>
    </div>
  );
}
