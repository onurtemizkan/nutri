'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { X, Loader2, Gift, Clock, Ban } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  useGrantSubscription,
  useExtendSubscription,
  useRevokeSubscription,
} from '@/lib/hooks/useSubscriptions';

interface BaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
  userEmail: string;
}

// Grant Subscription Modal
export function GrantSubscriptionModal({
  isOpen,
  onClose,
  userId,
  userEmail,
}: BaseModalProps) {
  const [duration, setDuration] = useState<string>('30_days');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  const grantMutation = useGrantSubscription();

  const handleGrant = async () => {
    if (reason.length < 5) {
      setError('Please provide a reason');
      return;
    }

    try {
      await grantMutation.mutateAsync({ userId, duration, reason });
      onClose();
      setDuration('30_days');
      setReason('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to grant subscription');
    }
  };

  const handleClose = () => {
    setDuration('30_days');
    setReason('');
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />
      <div className="relative w-full max-w-md rounded-lg bg-card border border-border p-6 shadow-xl">
        <button
          onClick={handleClose}
          className="absolute right-4 top-4 text-text-tertiary hover:text-text-primary"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-500/10">
            <Gift className="h-5 w-5 text-green-500" />
          </div>
          <div>
            <h3 className="font-semibold text-text-primary">Grant Pro Access</h3>
            <p className="text-sm text-text-tertiary">{userEmail}</p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Duration
            </label>
            <select
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
            >
              <option value="7_days">7 Days</option>
              <option value="30_days">30 Days</option>
              <option value="90_days">90 Days</option>
              <option value="1_year">1 Year</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Reason
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why are you granting this subscription?"
              rows={3}
              className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none"
            />
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex gap-3 pt-2">
            <Button variant="outline" onClick={handleClose} className="flex-1">
              Cancel
            </Button>
            <Button
              onClick={handleGrant}
              disabled={grantMutation.isPending || reason.length < 5}
              className="flex-1"
            >
              {grantMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Granting...
                </>
              ) : (
                'Grant Access'
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Extend Subscription Modal
export function ExtendSubscriptionModal({
  isOpen,
  onClose,
  userId,
  userEmail,
}: BaseModalProps) {
  const [days, setDays] = useState<number>(30);
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  const extendMutation = useExtendSubscription();

  const handleExtend = async () => {
    if (reason.length < 5) {
      setError('Please provide a reason');
      return;
    }

    try {
      await extendMutation.mutateAsync({ userId, days, reason });
      onClose();
      setDays(30);
      setReason('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extend subscription');
    }
  };

  const handleClose = () => {
    setDays(30);
    setReason('');
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />
      <div className="relative w-full max-w-md rounded-lg bg-card border border-border p-6 shadow-xl">
        <button
          onClick={handleClose}
          className="absolute right-4 top-4 text-text-tertiary hover:text-text-primary"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-500/10">
            <Clock className="h-5 w-5 text-blue-500" />
          </div>
          <div>
            <h3 className="font-semibold text-text-primary">Extend Subscription</h3>
            <p className="text-sm text-text-tertiary">{userEmail}</p>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Days to Add
            </label>
            <input
              type="number"
              min={1}
              max={365}
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value) || 0)}
              className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary focus:border-primary focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Reason
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why are you extending this subscription?"
              rows={3}
              className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none"
            />
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex gap-3 pt-2">
            <Button variant="outline" onClick={handleClose} className="flex-1">
              Cancel
            </Button>
            <Button
              onClick={handleExtend}
              disabled={extendMutation.isPending || days < 1 || reason.length < 5}
              className="flex-1"
            >
              {extendMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Extending...
                </>
              ) : (
                `Extend by ${days} Days`
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Revoke Subscription Modal
export function RevokeSubscriptionModal({
  isOpen,
  onClose,
  userId,
  userEmail,
}: BaseModalProps) {
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  const revokeMutation = useRevokeSubscription();

  const handleRevoke = async () => {
    if (reason.length < 10) {
      setError('Reason must be at least 10 characters');
      return;
    }

    try {
      await revokeMutation.mutateAsync({ userId, reason });
      onClose();
      setReason('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke subscription');
    }
  };

  const handleClose = () => {
    setReason('');
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={handleClose} />
      <div className="relative w-full max-w-md rounded-lg bg-card border border-border p-6 shadow-xl">
        <button
          onClick={handleClose}
          className="absolute right-4 top-4 text-text-tertiary hover:text-text-primary"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10">
            <Ban className="h-5 w-5 text-red-500" />
          </div>
          <div>
            <h3 className="font-semibold text-text-primary">Revoke Subscription</h3>
            <p className="text-sm text-text-tertiary">{userEmail}</p>
          </div>
        </div>

        <div className="space-y-4">
          <p className="text-sm text-text-secondary">
            This will immediately revoke the user&apos;s Pro access. They will be
            downgraded to Free tier.
          </p>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-2">
              Reason (min. 10 characters)
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Why are you revoking this subscription?"
              rows={3}
              className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary placeholder:text-text-disabled focus:border-red-500 focus:outline-none"
            />
            <p className="mt-1 text-xs text-text-tertiary">
              {reason.length}/10 characters
            </p>
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex gap-3 pt-2">
            <Button variant="outline" onClick={handleClose} className="flex-1">
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleRevoke}
              disabled={revokeMutation.isPending || reason.length < 10}
              className="flex-1"
            >
              {revokeMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Revoking...
                </>
              ) : (
                'Revoke Access'
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Wrapper component for subscription actions
interface SubscriptionActionsProps {
  userId: string;
  userEmail: string;
  currentTier: string;
}

export function SubscriptionActions({
  userId,
  userEmail,
  currentTier,
}: SubscriptionActionsProps) {
  const { data: session } = useSession();
  const [grantOpen, setGrantOpen] = useState(false);
  const [extendOpen, setExtendOpen] = useState(false);
  const [revokeOpen, setRevokeOpen] = useState(false);

  const isSuperAdmin = session?.user?.role === 'SUPER_ADMIN';

  if (!isSuperAdmin) {
    return null;
  }

  const isPro = currentTier === 'PRO' || currentTier === 'PRO_TRIAL';

  return (
    <div className="flex gap-2">
      {!isPro && (
        <>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setGrantOpen(true)}
            className="gap-1.5"
          >
            <Gift className="h-4 w-4" />
            Grant
          </Button>
          <GrantSubscriptionModal
            isOpen={grantOpen}
            onClose={() => setGrantOpen(false)}
            userId={userId}
            userEmail={userEmail}
          />
        </>
      )}

      {isPro && (
        <>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExtendOpen(true)}
            className="gap-1.5"
          >
            <Clock className="h-4 w-4" />
            Extend
          </Button>
          <ExtendSubscriptionModal
            isOpen={extendOpen}
            onClose={() => setExtendOpen(false)}
            userId={userId}
            userEmail={userEmail}
          />

          <Button
            variant="outline"
            size="sm"
            onClick={() => setRevokeOpen(true)}
            className="gap-1.5 text-red-500 hover:text-red-600"
          >
            <Ban className="h-4 w-4" />
            Revoke
          </Button>
          <RevokeSubscriptionModal
            isOpen={revokeOpen}
            onClose={() => setRevokeOpen(false)}
            userId={userId}
            userEmail={userEmail}
          />
        </>
      )}
    </div>
  );
}
