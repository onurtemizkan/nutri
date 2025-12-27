'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Plus, Search, Mail, Users, Play, XCircle, Clock, CheckCircle, Edit, Trash2, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useEmailCampaigns, useDeleteEmailCampaign } from '@/lib/hooks/useEmailCampaigns';
import { format } from 'date-fns';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Dialog from '@radix-ui/react-dialog';

const STATUS_CONFIG: Record<
  string,
  { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'muted'; icon: React.ReactNode }
> = {
  DRAFT: { label: 'Draft', variant: 'muted', icon: <Edit className="w-3 h-3" /> },
  SCHEDULED: { label: 'Scheduled', variant: 'info', icon: <Clock className="w-3 h-3" /> },
  SENDING: { label: 'Sending', variant: 'warning', icon: <Play className="w-3 h-3" /> },
  SENT: { label: 'Sent', variant: 'success', icon: <CheckCircle className="w-3 h-3" /> },
  CANCELLED: { label: 'Cancelled', variant: 'danger', icon: <XCircle className="w-3 h-3" /> },
};

export default function EmailCampaignsPage() {
  const router = useRouter();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [campaignToDelete, setCampaignToDelete] = useState<string | null>(null);

  const { data, isLoading, error } = useEmailCampaigns({
    status: statusFilter || undefined,
  });

  const deleteMutation = useDeleteEmailCampaign();

  const handleDelete = async () => {
    if (!campaignToDelete) return;

    try {
      await deleteMutation.mutateAsync(campaignToDelete);
      setDeleteDialogOpen(false);
      setCampaignToDelete(null);
    } catch (err) {
      console.error('Failed to delete campaign:', err);
    }
  };

  const campaigns = data?.campaigns || [];

  // Filter by search
  const filteredCampaigns = campaigns.filter((campaign) =>
    campaign.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Email Campaigns</h1>
          <p className="text-text-secondary mt-1">
            Create and manage marketing email campaigns
          </p>
        </div>
        <Button onClick={() => router.push('/dashboard/email/campaigns/new')}>
          <Plus className="w-4 h-4 mr-2" />
          New Campaign
        </Button>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search campaigns..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-4 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">All Statuses</option>
          <option value="DRAFT">Draft</option>
          <option value="SCHEDULED">Scheduled</option>
          <option value="SENDING">Sending</option>
          <option value="SENT">Sent</option>
          <option value="CANCELLED">Cancelled</option>
        </select>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-md text-red-400">
          Failed to load campaigns. Please try again.
        </div>
      )}

      {/* Campaigns List */}
      {!isLoading && !error && (
        <div className="space-y-4">
          {filteredCampaigns.length === 0 ? (
            <div className="text-center py-12 text-text-muted">
              <Mail className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">No campaigns found</p>
              <p className="text-sm mt-1">
                {search ? 'Try adjusting your search' : 'Create your first email campaign'}
              </p>
            </div>
          ) : (
            <div className="bg-background-secondary border border-border rounded-lg overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left px-4 py-3 text-sm font-medium text-text-secondary">
                      Campaign
                    </th>
                    <th className="text-left px-4 py-3 text-sm font-medium text-text-secondary">
                      Status
                    </th>
                    <th className="text-left px-4 py-3 text-sm font-medium text-text-secondary">
                      Audience
                    </th>
                    <th className="text-left px-4 py-3 text-sm font-medium text-text-secondary">
                      Scheduled
                    </th>
                    <th className="text-left px-4 py-3 text-sm font-medium text-text-secondary">
                      Performance
                    </th>
                    <th className="w-12"></th>
                  </tr>
                </thead>
                <tbody>
                  {filteredCampaigns.map((campaign) => {
                    const statusInfo = STATUS_CONFIG[campaign.status] || STATUS_CONFIG.DRAFT;
                    const openRate = campaign.actualSent > 0 ? 0 : 0; // Would need opens data

                    return (
                      <tr
                        key={campaign.id}
                        className="border-b border-border last:border-0 hover:bg-background-primary/50 transition-colors"
                      >
                        <td className="px-4 py-4">
                          <div>
                            <p className="font-medium text-text-primary">
                              {campaign.name}
                            </p>
                            {campaign.template && (
                              <p className="text-sm text-text-muted mt-0.5">
                                Template: {campaign.template.name}
                              </p>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <Badge variant={statusInfo.variant} className="gap-1">
                            {statusInfo.icon}
                            {statusInfo.label}
                          </Badge>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex items-center gap-2 text-sm text-text-secondary">
                            <Users className="w-4 h-4" />
                            {campaign.estimatedAudience?.toLocaleString() || 'N/A'}
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          {campaign.scheduledAt ? (
                            <div className="text-sm">
                              <p className="text-text-primary">
                                {format(new Date(campaign.scheduledAt), 'MMM d, yyyy')}
                              </p>
                              <p className="text-text-muted">
                                {format(new Date(campaign.scheduledAt), 'h:mm a')}
                              </p>
                            </div>
                          ) : (
                            <span className="text-sm text-text-muted">Not scheduled</span>
                          )}
                        </td>
                        <td className="px-4 py-4">
                          {campaign.status === 'SENT' ? (
                            <div className="text-sm">
                              <p className="text-text-primary">
                                {campaign.actualSent.toLocaleString()} sent
                              </p>
                              <p className="text-text-muted">
                                {openRate.toFixed(1)}% opened
                              </p>
                            </div>
                          ) : (
                            <span className="text-sm text-text-muted">—</span>
                          )}
                        </td>
                        <td className="px-4 py-4">
                          <DropdownMenu.Root>
                            <DropdownMenu.Trigger asChild>
                              <button className="p-2 hover:bg-background-primary rounded-md">
                                <MoreHorizontal className="w-4 h-4 text-text-muted" />
                              </button>
                            </DropdownMenu.Trigger>
                            <DropdownMenu.Portal>
                              <DropdownMenu.Content
                                className="min-w-[160px] bg-background-secondary border border-border rounded-md shadow-lg p-1 z-50"
                                sideOffset={5}
                                align="end"
                              >
                                <DropdownMenu.Item
                                  className="px-3 py-2 text-sm text-text-primary hover:bg-background-primary rounded cursor-pointer outline-none"
                                  onClick={() => router.push(`/dashboard/email/campaigns/${campaign.id}/edit`)}
                                >
                                  <Edit className="w-4 h-4 mr-2 inline" />
                                  Edit
                                </DropdownMenu.Item>
                                {campaign.status === 'DRAFT' && (
                                  <DropdownMenu.Item
                                    className="px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded cursor-pointer outline-none"
                                    onClick={() => {
                                      setCampaignToDelete(campaign.id);
                                      setDeleteDialogOpen(true);
                                    }}
                                  >
                                    <Trash2 className="w-4 h-4 mr-2 inline" />
                                    Delete
                                  </DropdownMenu.Item>
                                )}
                              </DropdownMenu.Content>
                            </DropdownMenu.Portal>
                          </DropdownMenu.Root>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog.Root open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <Dialog.Title className="text-lg font-semibold text-text-primary">
              Delete Campaign
            </Dialog.Title>
            <p className="mt-2 text-text-secondary">
              Are you sure you want to delete this campaign? This action cannot be undone.
            </p>
            <div className="flex gap-3 mt-6 justify-end">
              <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={deleteMutation.isPending}
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
