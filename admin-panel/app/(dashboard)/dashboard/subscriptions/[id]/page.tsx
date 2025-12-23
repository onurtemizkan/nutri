'use client';

import { useParams, useRouter } from 'next/navigation';
import { useSubscription } from '@/lib/hooks/useSubscriptions';
import { useSession } from 'next-auth/react';
import { useState } from 'react';
import {
  ArrowLeft,
  Loader2,
  CreditCard,
  Calendar,
  User,
  Gift,
  Clock,
  Ban,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Badge,
  getSubscriptionBadgeVariant,
  getSubscriptionBadgeLabel,
} from '@/components/ui/badge';
import { formatDate } from '@/lib/utils';
import {
  GrantSubscriptionModal,
  ExtendSubscriptionModal,
  RevokeSubscriptionModal,
} from '@/components/subscriptions/subscription-modals';

export default function SubscriptionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { data: session } = useSession();
  const userId = params.id as string;

  const [grantOpen, setGrantOpen] = useState(false);
  const [extendOpen, setExtendOpen] = useState(false);
  const [revokeOpen, setRevokeOpen] = useState(false);

  const { data: subscription, isLoading, isError, error } = useSubscription(userId);

  const isSuperAdmin = session?.user?.role === 'SUPER_ADMIN';

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (isError || !subscription) {
    return (
      <div className="space-y-4">
        <Button
          variant="ghost"
          onClick={() => router.push('/dashboard/subscriptions')}
          className="gap-1.5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Subscriptions
        </Button>
        <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-6 text-red-500">
          {error?.message || 'Subscription not found'}
        </div>
      </div>
    );
  }

  const isPro =
    subscription.subscriptionTier === 'PRO' ||
    subscription.subscriptionTier === 'PRO_TRIAL';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          onClick={() => router.push('/dashboard/subscriptions')}
          className="gap-1.5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <div className="flex-1">
          <h2 className="text-2xl font-bold text-text-primary">
            Subscription Details
          </h2>
          <p className="mt-0.5 text-text-tertiary">{subscription.email}</p>
        </div>
        <Badge variant={getSubscriptionBadgeVariant(subscription.subscriptionTier)}>
          {getSubscriptionBadgeLabel(subscription.subscriptionTier)}
        </Badge>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* User Info Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <User className="h-4 w-4" />
            User Information
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Name</dt>
              <dd className="text-text-primary">{subscription.name}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Email</dt>
              <dd className="text-text-primary">{subscription.email}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">User ID</dt>
              <dd className="font-mono text-xs text-text-secondary">
                {subscription.id}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Account Created</dt>
              <dd className="text-text-primary">
                {formatDate(subscription.createdAt, { dateStyle: 'medium' })}
              </dd>
            </div>
            {subscription.appleId && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Apple ID</dt>
                <dd className="font-mono text-xs text-text-secondary">
                  {subscription.appleId}
                </dd>
              </div>
            )}
          </dl>
        </div>

        {/* Subscription Status Card */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <CreditCard className="h-4 w-4" />
            Subscription Status
          </h3>
          <dl className="space-y-3 text-sm">
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Current Tier</dt>
              <dd>
                <Badge
                  variant={getSubscriptionBadgeVariant(
                    subscription.subscriptionTier
                  )}
                >
                  {getSubscriptionBadgeLabel(subscription.subscriptionTier)}
                </Badge>
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-text-tertiary">Status</dt>
              <dd
                className={
                  subscription.isActive ? 'text-green-500' : 'text-red-500'
                }
              >
                {subscription.isActive ? 'Active' : 'Inactive'}
              </dd>
            </div>
            {subscription.subscriptionBillingCycle && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Billing Cycle</dt>
                <dd className="text-text-primary">
                  {subscription.subscriptionBillingCycle === 'MONTHLY'
                    ? 'Monthly'
                    : 'Annual'}
                </dd>
              </div>
            )}
            {subscription.subscriptionStartDate && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Started</dt>
                <dd className="text-text-primary">
                  {formatDate(subscription.subscriptionStartDate, {
                    dateStyle: 'medium',
                  })}
                </dd>
              </div>
            )}
            {subscription.subscriptionEndDate && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">
                  {subscription.isActive ? 'Expires' : 'Expired'}
                </dt>
                <dd
                  className={
                    subscription.isActive
                      ? subscription.daysRemaining !== null &&
                        subscription.daysRemaining <= 7
                        ? 'text-yellow-500'
                        : 'text-text-primary'
                      : 'text-red-500'
                  }
                >
                  {formatDate(subscription.subscriptionEndDate, {
                    dateStyle: 'medium',
                  })}
                </dd>
              </div>
            )}
            {subscription.daysRemaining !== null && subscription.isActive && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Days Remaining</dt>
                <dd
                  className={
                    subscription.daysRemaining <= 7
                      ? 'text-yellow-500 font-medium'
                      : 'text-text-primary'
                  }
                >
                  {subscription.daysRemaining} days
                </dd>
              </div>
            )}
            {subscription.subscriptionPrice !== null && (
              <div className="flex justify-between">
                <dt className="text-text-tertiary">Price</dt>
                <dd className="text-text-primary">
                  {subscription.subscriptionPrice === 0
                    ? 'Free (Manual Grant)'
                    : `$${subscription.subscriptionPrice.toFixed(2)}`}
                </dd>
              </div>
            )}
          </dl>
        </div>
      </div>

      {/* Admin Actions */}
      {isSuperAdmin && (
        <div className="rounded-lg border border-border bg-card p-6">
          <h3 className="mb-4 flex items-center gap-2 font-semibold text-text-primary">
            <Calendar className="h-4 w-4" />
            Admin Actions
          </h3>
          <p className="mb-4 text-sm text-text-secondary">
            These actions are logged in the audit trail. Requires SUPER_ADMIN
            role.
          </p>
          <div className="flex flex-wrap gap-3">
            {!isPro && (
              <Button
                variant="outline"
                onClick={() => setGrantOpen(true)}
                className="gap-2"
              >
                <Gift className="h-4 w-4" />
                Grant Pro Access
              </Button>
            )}

            {isPro && (
              <>
                <Button
                  variant="outline"
                  onClick={() => setExtendOpen(true)}
                  className="gap-2"
                >
                  <Clock className="h-4 w-4" />
                  Extend Subscription
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setRevokeOpen(true)}
                  className="gap-2 text-red-500 hover:text-red-600"
                >
                  <Ban className="h-4 w-4" />
                  Revoke Access
                </Button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Modals */}
      <GrantSubscriptionModal
        isOpen={grantOpen}
        onClose={() => setGrantOpen(false)}
        userId={subscription.id}
        userEmail={subscription.email}
      />
      <ExtendSubscriptionModal
        isOpen={extendOpen}
        onClose={() => setExtendOpen(false)}
        userId={subscription.id}
        userEmail={subscription.email}
      />
      <RevokeSubscriptionModal
        isOpen={revokeOpen}
        onClose={() => setRevokeOpen(false)}
        userId={subscription.id}
        userEmail={subscription.email}
      />
    </div>
  );
}
