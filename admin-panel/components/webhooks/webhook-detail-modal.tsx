'use client';

import { X, RotateCw, Loader2, Copy, Check, User, AlertTriangle } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatDate } from '@/lib/utils';
import type { WebhookEventDetail, WebhookEventStatus } from '@/lib/hooks/useWebhooks';

interface WebhookDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  webhook: WebhookEventDetail | null;
  isLoading: boolean;
  onRetry: (id: string) => void;
  isRetrying: boolean;
  isSuperAdmin: boolean;
}

function getStatusBadgeVariant(status: WebhookEventStatus): 'success' | 'warning' | 'danger' | 'muted' {
  switch (status) {
    case 'SUCCESS':
      return 'success';
    case 'PENDING':
      return 'warning';
    case 'FAILED':
      return 'danger';
    default:
      return 'muted';
  }
}

export function WebhookDetailModal({
  isOpen,
  onClose,
  webhook,
  isLoading,
  onRetry,
  isRetrying,
  isSuperAdmin,
}: WebhookDetailModalProps) {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleCopyPayload = async () => {
    if (!webhook?.payload) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(webhook.payload, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  const canRetry = isSuperAdmin && webhook?.status === 'FAILED';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative w-full max-w-3xl max-h-[90vh] overflow-hidden rounded-lg bg-card border border-border shadow-xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <h3 className="font-semibold text-text-primary text-lg">
              Webhook Event Details
            </h3>
            {webhook && (
              <Badge variant={getStatusBadgeVariant(webhook.status)}>
                {webhook.status}
              </Badge>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-text-tertiary hover:text-text-primary"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : webhook ? (
            <div className="space-y-6">
              {/* Overview */}
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-border bg-background-secondary p-4">
                  <h4 className="text-sm font-medium text-text-tertiary mb-3">
                    Event Information
                  </h4>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">Type</dt>
                      <dd className="text-text-primary font-medium">
                        {webhook.notificationType}
                      </dd>
                    </div>
                    {webhook.subtype && (
                      <div className="flex justify-between">
                        <dt className="text-text-tertiary">Subtype</dt>
                        <dd className="text-text-secondary">{webhook.subtype}</dd>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">Event ID</dt>
                      <dd className="font-mono text-xs text-text-secondary">
                        {webhook.id}
                      </dd>
                    </div>
                    {webhook.bundleId && (
                      <div className="flex justify-between">
                        <dt className="text-text-tertiary">Bundle ID</dt>
                        <dd className="text-text-secondary">{webhook.bundleId}</dd>
                      </div>
                    )}
                  </dl>
                </div>

                <div className="rounded-lg border border-border bg-background-secondary p-4">
                  <h4 className="text-sm font-medium text-text-tertiary mb-3">
                    Transaction Details
                  </h4>
                  <dl className="space-y-2 text-sm">
                    {webhook.originalTransactionId && (
                      <div className="flex justify-between">
                        <dt className="text-text-tertiary">Original Txn ID</dt>
                        <dd className="font-mono text-xs text-text-secondary">
                          {webhook.originalTransactionId}
                        </dd>
                      </div>
                    )}
                    {webhook.transactionId && (
                      <div className="flex justify-between">
                        <dt className="text-text-tertiary">Transaction ID</dt>
                        <dd className="font-mono text-xs text-text-secondary">
                          {webhook.transactionId}
                        </dd>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">Received At</dt>
                      <dd className="text-text-secondary">
                        {formatDate(webhook.receivedAt, {
                          dateStyle: 'medium',
                          timeStyle: 'medium',
                        })}
                      </dd>
                    </div>
                    {webhook.processedAt && (
                      <div className="flex justify-between">
                        <dt className="text-text-tertiary">Processed At</dt>
                        <dd className="text-text-secondary">
                          {formatDate(webhook.processedAt, {
                            dateStyle: 'medium',
                            timeStyle: 'medium',
                          })}
                        </dd>
                      </div>
                    )}
                  </dl>
                </div>
              </div>

              {/* User Info */}
              {webhook.user && (
                <div className="rounded-lg border border-border bg-background-secondary p-4">
                  <h4 className="text-sm font-medium text-text-tertiary mb-3 flex items-center gap-2">
                    <User className="h-4 w-4" />
                    Associated User
                  </h4>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">Name</dt>
                      <dd className="text-text-primary">{webhook.user.name}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">Email</dt>
                      <dd className="text-text-secondary">{webhook.user.email}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-text-tertiary">User ID</dt>
                      <dd className="font-mono text-xs text-text-secondary">
                        {webhook.user.id}
                      </dd>
                    </div>
                  </dl>
                </div>
              )}

              {/* Error Info */}
              {webhook.status === 'FAILED' && webhook.errorMessage && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4">
                  <h4 className="text-sm font-medium text-red-500 mb-2 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4" />
                    Error Details
                  </h4>
                  <p className="text-sm text-red-400">{webhook.errorMessage}</p>
                  <div className="mt-2 flex gap-4 text-xs text-red-400/70">
                    <span>Retry Count: {webhook.retryCount}</span>
                    {webhook.lastRetryAt && (
                      <span>
                        Last Retry:{' '}
                        {formatDate(webhook.lastRetryAt, {
                          dateStyle: 'short',
                          timeStyle: 'short',
                        })}
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Payload */}
              <div className="rounded-lg border border-border bg-background-secondary p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-text-tertiary">
                    Raw Payload
                  </h4>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCopyPayload}
                    className="gap-1.5"
                  >
                    {copied ? (
                      <>
                        <Check className="h-4 w-4 text-green-500" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-4 w-4" />
                        Copy
                      </>
                    )}
                  </Button>
                </div>
                <pre className="overflow-x-auto rounded bg-background-elevated p-4 text-xs text-text-secondary">
                  {JSON.stringify(webhook.payload, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-text-disabled">
              Webhook event not found
            </div>
          )}
        </div>

        {/* Footer */}
        {webhook && (
          <div className="flex justify-end gap-3 p-6 border-t border-border">
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
            {canRetry && (
              <Button onClick={() => onRetry(webhook.id)} disabled={isRetrying}>
                {isRetrying ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Retrying...
                  </>
                ) : (
                  <>
                    <RotateCw className="h-4 w-4 mr-2" />
                    Retry Webhook
                  </>
                )}
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
