'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import { Trash2, Loader2, AlertTriangle, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { adminApi } from '@/lib/api';

interface DeleteUserModalProps {
  userId: string;
  userEmail: string;
  userName: string;
}

type Step = 'confirm' | 'verify' | 'reason' | 'deleting';

export function DeleteUserModal({
  userId,
  userEmail,
  userName,
}: DeleteUserModalProps) {
  const router = useRouter();
  const { data: session } = useSession();
  const [isOpen, setIsOpen] = useState(false);
  const [step, setStep] = useState<Step>('confirm');
  const [emailConfirmation, setEmailConfirmation] = useState('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Only SUPER_ADMIN can delete users
  const isSuperAdmin = session?.user?.role === 'SUPER_ADMIN';

  const handleOpen = () => {
    setIsOpen(true);
    setStep('confirm');
    setEmailConfirmation('');
    setReason('');
    setError(null);
  };

  const handleClose = () => {
    setIsOpen(false);
    setStep('confirm');
    setEmailConfirmation('');
    setReason('');
    setError(null);
  };

  const handleConfirm = () => {
    setStep('verify');
  };

  const handleVerify = () => {
    if (emailConfirmation !== userEmail) {
      setError('Email does not match');
      return;
    }
    setError(null);
    setStep('reason');
  };

  const handleDelete = async () => {
    if (reason.length < 10) {
      setError('Reason must be at least 10 characters');
      return;
    }

    setStep('deleting');
    setError(null);

    try {
      await adminApi.deleteUser(userId, reason);
      handleClose();
      router.push('/dashboard/users?deleted=true');
    } catch (err) {
      setStep('reason');
      setError(err instanceof Error ? err.message : 'Failed to delete user');
    }
  };

  if (!isSuperAdmin) {
    return null;
  }

  return (
    <>
      <Button variant="destructive" onClick={handleOpen} className="gap-2">
        <Trash2 className="h-4 w-4" />
        Delete Account
      </Button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={handleClose}
          />

          {/* Modal */}
          <div className="relative w-full max-w-md rounded-lg bg-card border border-border p-6 shadow-xl">
            {/* Close button */}
            <button
              onClick={handleClose}
              className="absolute right-4 top-4 text-text-tertiary hover:text-text-primary"
            >
              <X className="h-5 w-5" />
            </button>

            {/* Header */}
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10">
                <AlertTriangle className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <h3 className="font-semibold text-text-primary">
                  Delete User Account
                </h3>
                <p className="text-sm text-text-tertiary">{userName}</p>
              </div>
            </div>

            {/* Step: Confirm */}
            {step === 'confirm' && (
              <div className="space-y-4">
                <p className="text-sm text-text-secondary">
                  Are you sure you want to delete this user account? This action
                  is <strong className="text-red-500">permanent</strong> and
                  cannot be undone.
                </p>
                <p className="text-sm text-text-secondary">
                  All user data will be permanently deleted, including:
                </p>
                <ul className="list-inside list-disc text-sm text-text-tertiary">
                  <li>Profile information</li>
                  <li>Meal history</li>
                  <li>Health metrics</li>
                  <li>Activity data</li>
                  <li>Subscription records</li>
                </ul>
                <div className="flex gap-3 pt-2">
                  <Button
                    variant="outline"
                    onClick={handleClose}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleConfirm}
                    className="flex-1"
                  >
                    Continue
                  </Button>
                </div>
              </div>
            )}

            {/* Step: Verify Email */}
            {step === 'verify' && (
              <div className="space-y-4">
                <p className="text-sm text-text-secondary">
                  To confirm deletion, please type the user&apos;s email address:
                </p>
                <p className="font-mono text-sm text-text-primary">
                  {userEmail}
                </p>
                <input
                  type="text"
                  value={emailConfirmation}
                  onChange={(e) => setEmailConfirmation(e.target.value)}
                  placeholder="Type email to confirm"
                  className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary placeholder:text-text-disabled focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
                />
                {error && <p className="text-sm text-red-500">{error}</p>}
                <div className="flex gap-3 pt-2">
                  <Button
                    variant="outline"
                    onClick={() => setStep('confirm')}
                    className="flex-1"
                  >
                    Back
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleVerify}
                    className="flex-1"
                    disabled={emailConfirmation !== userEmail}
                  >
                    Verify
                  </Button>
                </div>
              </div>
            )}

            {/* Step: Reason */}
            {step === 'reason' && (
              <div className="space-y-4">
                <p className="text-sm text-text-secondary">
                  Please provide a reason for deleting this account. This will be
                  recorded in the audit log.
                </p>
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Reason for deletion (min. 10 characters)"
                  rows={3}
                  className="w-full rounded-md border border-border bg-background-elevated px-4 py-2 text-text-primary placeholder:text-text-disabled focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
                />
                <p className="text-xs text-text-tertiary">
                  {reason.length}/10 characters minimum
                </p>
                {error && <p className="text-sm text-red-500">{error}</p>}
                <div className="flex gap-3 pt-2">
                  <Button
                    variant="outline"
                    onClick={() => setStep('verify')}
                    className="flex-1"
                  >
                    Back
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleDelete}
                    className="flex-1"
                    disabled={reason.length < 10}
                  >
                    Delete Account
                  </Button>
                </div>
              </div>
            )}

            {/* Step: Deleting */}
            {step === 'deleting' && (
              <div className="flex flex-col items-center gap-4 py-6">
                <Loader2 className="h-8 w-8 animate-spin text-red-500" />
                <p className="text-sm text-text-secondary">
                  Deleting user account...
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
