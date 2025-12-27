'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Save, Users, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCreateEmailCampaign } from '@/lib/hooks/useEmailCampaigns';
import { useEmailTemplates } from '@/lib/hooks/useEmailTemplates';

// Segment criteria options
const SUBSCRIPTION_TIERS = ['FREE', 'PRO', 'PRO_TRIAL'];
const ACTIVITY_LEVELS = [
  { value: 'active_7d', label: 'Active in last 7 days' },
  { value: 'active_30d', label: 'Active in last 30 days' },
  { value: 'inactive_7d', label: 'Inactive for 7+ days' },
  { value: 'inactive_14d', label: 'Inactive for 14+ days' },
  { value: 'inactive_30d', label: 'Inactive for 30+ days' },
];

export default function NewCampaignPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [templateId, setTemplateId] = useState('');
  const [scheduleType, setScheduleType] = useState<'now' | 'scheduled'>('now');
  const [scheduledAt, setScheduledAt] = useState('');
  const [subscriptionTier, setSubscriptionTier] = useState<string>('');
  const [activityLevel, setActivityLevel] = useState<string>('');
  const [estimatedAudience, setEstimatedAudience] = useState<number | null>(null);
  const [error, setError] = useState('');

  const { data: templatesData } = useEmailTemplates({ category: 'MARKETING', isActive: true });
  const createMutation = useCreateEmailCampaign();

  const templates = templatesData?.templates || [];

  // Estimate audience (simplified - would call backend in real implementation)
  useEffect(() => {
    // This would be a real API call
    const baseAudience = 1000;
    let multiplier = 1;

    if (subscriptionTier) {
      multiplier *= subscriptionTier === 'PRO' ? 0.2 : subscriptionTier === 'PRO_TRIAL' ? 0.1 : 0.7;
    }

    if (activityLevel) {
      if (activityLevel.includes('inactive')) {
        multiplier *= 0.3;
      } else {
        multiplier *= 0.7;
      }
    }

    setEstimatedAudience(Math.round(baseAudience * multiplier));
  }, [subscriptionTier, activityLevel]);

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
      const campaign = await createMutation.mutateAsync({
        name: name.trim(),
        description: description.trim() || undefined,
        templateId,
        scheduledAt: scheduleType === 'scheduled' ? scheduledAt : undefined,
        segmentCriteria,
      });

      router.push(`/dashboard/email/campaigns/${campaign.id}/edit`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create campaign');
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.push('/dashboard/email/campaigns')}
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>
        <h1 className="text-2xl font-semibold text-text-primary">New Campaign</h1>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-md text-red-400">
          {error}
        </div>
      )}

      {/* Form */}
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
              placeholder="e.g., Spring Promotion 2024"
              className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Internal notes about this campaign..."
              rows={2}
              className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1">
              Email Template *
            </label>
            <select
              value={templateId}
              onChange={(e) => setTemplateId(e.target.value)}
              className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              <option value="">Select a template</option>
              {templates.map((template) => (
                <option key={template.id} value={template.id}>
                  {template.name}
                </option>
              ))}
            </select>
            {templates.length === 0 && (
              <p className="mt-1 text-sm text-text-muted">
                No marketing templates available.{' '}
                <a
                  href="/dashboard/email/templates/new"
                  className="text-primary hover:underline"
                >
                  Create one
                </a>
              </p>
            )}
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
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
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
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
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

          {estimatedAudience !== null && (
            <div className="p-3 bg-primary/10 border border-primary/20 rounded-md">
              <p className="text-sm text-text-secondary">
                Estimated audience:{' '}
                <span className="font-semibold text-text-primary">
                  {estimatedAudience.toLocaleString()} users
                </span>
              </p>
            </div>
          )}
        </div>

        {/* Scheduling */}
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
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-3">
        <Button variant="outline" onClick={() => router.push('/dashboard/email/campaigns')}>
          Cancel
        </Button>
        <Button onClick={handleSave} disabled={createMutation.isPending}>
          <Save className="w-4 h-4 mr-2" />
          {createMutation.isPending ? 'Creating...' : 'Create Campaign'}
        </Button>
      </div>
    </div>
  );
}
