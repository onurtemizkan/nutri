'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { ArrowLeft, Save, Send, XCircle, Users, Calendar, Eye, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  useEmailCampaign,
  useUpdateEmailCampaign,
  useSendEmailCampaign,
  useCancelEmailCampaign,
} from '@/lib/hooks/useEmailCampaigns';
import { useEmailTemplates } from '@/lib/hooks/useEmailTemplates';
import { format } from 'date-fns';
import * as Dialog from '@radix-ui/react-dialog';

const SUBSCRIPTION_TIERS = ['FREE', 'PRO', 'PRO_TRIAL'];
const ACTIVITY_LEVELS = [
  { value: 'active_7d', label: 'Active in last 7 days' },
  { value: 'active_30d', label: 'Active in last 30 days' },
  { value: 'inactive_7d', label: 'Inactive for 7+ days' },
  { value: 'inactive_14d', label: 'Inactive for 14+ days' },
  { value: 'inactive_30d', label: 'Inactive for 30+ days' },
];

const STATUS_CONFIG: Record<string, { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'muted' }> = {
  DRAFT: { label: 'Draft', variant: 'muted' },
  SCHEDULED: { label: 'Scheduled', variant: 'info' },
  SENDING: { label: 'Sending', variant: 'warning' },
  SENT: { label: 'Sent', variant: 'success' },
  CANCELLED: { label: 'Cancelled', variant: 'danger' },
};

export default function EditCampaignPage() {
  const router = useRouter();
  const params = useParams();
  const campaignId = params.id as string;

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [templateId, setTemplateId] = useState('');
  const [scheduleType, setScheduleType] = useState<'now' | 'scheduled'>('now');
  const [scheduledAt, setScheduledAt] = useState('');
  const [subscriptionTier, setSubscriptionTier] = useState<string>('');
  const [activityLevel, setActivityLevel] = useState<string>('');
  const [error, setError] = useState('');
  const [sendDialogOpen, setSendDialogOpen] = useState(false);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);

  const { data: campaign, isLoading } = useEmailCampaign(campaignId);
  const { data: templatesData } = useEmailTemplates({ category: 'MARKETING', isActive: true });
  const updateMutation = useUpdateEmailCampaign();
  const sendMutation = useSendEmailCampaign();
  const cancelMutation = useCancelEmailCampaign();

  const templates = templatesData?.templates || [];
  const isEditable = campaign?.status === 'DRAFT';

  // Populate form with campaign data
  useEffect(() => {
    if (campaign) {
      setName(campaign.name);
      setDescription(campaign.description || '');
      setTemplateId(campaign.templateId);
      if (campaign.scheduledAt) {
        setScheduleType('scheduled');
        setScheduledAt(new Date(campaign.scheduledAt).toISOString().slice(0, 16));
      }
      const criteria = campaign.segmentCriteria as Record<string, string>;
      if (criteria?.subscriptionTier) setSubscriptionTier(criteria.subscriptionTier);
      if (criteria?.activityLevel) setActivityLevel(criteria.activityLevel);
    }
  }, [campaign]);

  const handleSave = async () => {
    setError('');

    if (!name.trim()) {
      setError('Campaign name is required');
      return;
    }
    if (!templateId) {
      setError('Please select a template');
      return;
    }

    const segmentCriteria: Record<string, unknown> = {};
    if (subscriptionTier) segmentCriteria.subscriptionTier = subscriptionTier;
    if (activityLevel) segmentCriteria.activityLevel = activityLevel;

    try {
      await updateMutation.mutateAsync({
        id: campaignId,
        data: {
          name: name.trim(),
          description: description.trim() || undefined,
          templateId,
          scheduledAt: scheduleType === 'scheduled' ? scheduledAt : undefined,
          segmentCriteria,
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update campaign');
    }
  };

  const handleSend = async () => {
    try {
      await sendMutation.mutateAsync(campaignId);
      setSendDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send campaign');
    }
  };

  const handleCancel = async () => {
    try {
      await cancelMutation.mutateAsync(campaignId);
      setCancelDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel campaign');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!campaign) {
    return (
      <div className="text-center py-12">
        <p className="text-text-muted">Campaign not found</p>
      </div>
    );
  }

  const statusInfo = STATUS_CONFIG[campaign.status] || STATUS_CONFIG.DRAFT;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push('/dashboard/email/campaigns')}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-semibold text-text-primary">{campaign.name}</h1>
              <Badge variant={statusInfo.variant}>{statusInfo.label}</Badge>
            </div>
            {campaign.description && (
              <p className="text-text-secondary mt-1">{campaign.description}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {campaign.status === 'DRAFT' && (
            <>
              <Button
                variant="outline"
                onClick={handleSave}
                disabled={updateMutation.isPending}
              >
                <Save className="w-4 h-4 mr-2" />
                {updateMutation.isPending ? 'Saving...' : 'Save'}
              </Button>
              <Button onClick={() => setSendDialogOpen(true)}>
                <Send className="w-4 h-4 mr-2" />
                Send Campaign
              </Button>
            </>
          )}
          {(campaign.status === 'SCHEDULED' || campaign.status === 'SENDING') && (
            <Button
              variant="destructive"
              onClick={() => setCancelDialogOpen(true)}
            >
              <XCircle className="w-4 h-4 mr-2" />
              Cancel Campaign
            </Button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-md text-red-400">
          {error}
        </div>
      )}

      {/* Campaign Stats (for sent campaigns) */}
      {campaign.status === 'SENT' && (
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-background-secondary border border-border rounded-lg p-4">
            <p className="text-sm text-text-muted">Sent</p>
            <p className="text-2xl font-semibold text-text-primary">
              {campaign.actualSent.toLocaleString()}
            </p>
          </div>
          <div className="bg-background-secondary border border-border rounded-lg p-4">
            <p className="text-sm text-text-muted">Open Rate</p>
            <p className="text-2xl font-semibold text-text-primary">--</p>
          </div>
          <div className="bg-background-secondary border border-border rounded-lg p-4">
            <p className="text-sm text-text-muted">Click Rate</p>
            <p className="text-2xl font-semibold text-text-primary">--</p>
          </div>
          <div className="bg-background-secondary border border-border rounded-lg p-4">
            <p className="text-sm text-text-muted">Completed</p>
            <p className="text-2xl font-semibold text-text-primary">
              {campaign.completedAt
                ? format(new Date(campaign.completedAt), 'MMM d, h:mm a')
                : '--'}
            </p>
          </div>
        </div>
      )}

      {/* Form (editable for drafts) */}
      <div className="bg-background-secondary border border-border rounded-lg p-6 space-y-6">
        {/* Basic Info */}
        <div className="space-y-4">
          <h2 className="text-lg font-medium text-text-primary">Campaign Details</h2>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Campaign Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={!isEditable}
              className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={!isEditable}
              rows={2}
              className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Email Template *
            </label>
            <div className="flex gap-2">
              <select
                value={templateId}
                onChange={(e) => setTemplateId(e.target.value)}
                disabled={!isEditable}
                className="flex-1 px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
              >
                <option value="">Select a template</option>
                {templates.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.name}
                  </option>
                ))}
              </select>
              {templateId && (
                <Button
                  variant="outline"
                  onClick={() =>
                    window.open(`/dashboard/email/templates/${templateId}/edit`, '_blank')
                  }
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Preview
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Audience Targeting */}
        <div className="space-y-4 pt-4 border-t border-border">
          <h2 className="text-lg font-medium text-text-primary flex items-center gap-2">
            <Users className="w-5 h-5" />
            Audience Targeting
          </h2>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Subscription Tier
              </label>
              <select
                value={subscriptionTier}
                onChange={(e) => setSubscriptionTier(e.target.value)}
                disabled={!isEditable}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
              >
                <option value="">All Tiers</option>
                {SUBSCRIPTION_TIERS.map((tier) => (
                  <option key={tier} value={tier}>
                    {tier.replace('_', ' ')}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Activity Level
              </label>
              <select
                value={activityLevel}
                onChange={(e) => setActivityLevel(e.target.value)}
                disabled={!isEditable}
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
              >
                <option value="">All Users</option>
                {ACTIVITY_LEVELS.map((level) => (
                  <option key={level.value} value={level.value}>
                    {level.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="p-3 bg-primary/10 border border-primary/20 rounded-md">
            <p className="text-sm text-text-secondary">
              Estimated audience:{' '}
              <span className="font-semibold text-text-primary">
                {campaign.estimatedAudience?.toLocaleString() || 'N/A'} users
              </span>
            </p>
          </div>
        </div>

        {/* Scheduling */}
        {isEditable && (
          <div className="space-y-4 pt-4 border-t border-border">
            <h2 className="text-lg font-medium text-text-primary flex items-center gap-2">
              <Calendar className="w-5 h-5" />
              Schedule
            </h2>

            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="scheduleType"
                  value="now"
                  checked={scheduleType === 'now'}
                  onChange={() => setScheduleType('now')}
                  className="w-4 h-4 text-primary"
                />
                <span className="text-text-primary">Send immediately</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="scheduleType"
                  value="scheduled"
                  checked={scheduleType === 'scheduled'}
                  onChange={() => setScheduleType('scheduled')}
                  className="w-4 h-4 text-primary"
                />
                <span className="text-text-primary">Schedule for later</span>
              </label>
            </div>

            {scheduleType === 'scheduled' && (
              <input
                type="datetime-local"
                value={scheduledAt}
                onChange={(e) => setScheduledAt(e.target.value)}
                min={new Date().toISOString().slice(0, 16)}
                className="w-full max-w-xs px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            )}
          </div>
        )}
      </div>

      {/* Send Confirmation Dialog */}
      <Dialog.Root open={sendDialogOpen} onOpenChange={setSendDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <Dialog.Title className="text-lg font-semibold text-text-primary flex items-center gap-2">
              <Send className="w-5 h-5 text-primary" />
              Send Campaign
            </Dialog.Title>
            <div className="mt-4 space-y-3">
              <p className="text-text-secondary">
                You are about to send this campaign to approximately{' '}
                <strong>{campaign.estimatedAudience?.toLocaleString()}</strong> users.
              </p>
              {scheduleType === 'scheduled' && scheduledAt ? (
                <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-md text-blue-400 text-sm">
                  <Calendar className="w-4 h-4 inline mr-2" />
                  Scheduled for {format(new Date(scheduledAt), 'PPp')}
                </div>
              ) : (
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-md text-yellow-400 text-sm flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  This will send immediately and cannot be undone.
                </div>
              )}
            </div>
            <div className="flex gap-3 mt-6 justify-end">
              <Button variant="outline" onClick={() => setSendDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSend} disabled={sendMutation.isPending}>
                {sendMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Confirm Send
                  </>
                )}
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      {/* Cancel Confirmation Dialog */}
      <Dialog.Root open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <Dialog.Title className="text-lg font-semibold text-text-primary">
              Cancel Campaign
            </Dialog.Title>
            <p className="mt-2 text-text-secondary">
              Are you sure you want to cancel this campaign? Any emails still in the queue will not be sent.
            </p>
            <div className="flex gap-3 mt-6 justify-end">
              <Button variant="outline" onClick={() => setCancelDialogOpen(false)}>
                Keep Running
              </Button>
              <Button
                variant="destructive"
                onClick={handleCancel}
                disabled={cancelMutation.isPending}
              >
                {cancelMutation.isPending ? 'Cancelling...' : 'Cancel Campaign'}
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
