'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Plus, Search, Edit2, Copy, TestTube, Trash2, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useEmailTemplates, useDeleteEmailTemplate } from '@/lib/hooks/useEmailTemplates';
import { format } from 'date-fns';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@radix-ui/react-dropdown-menu';
import * as AlertDialog from '@radix-ui/react-alert-dialog';

export default function EmailTemplatesPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<'TRANSACTIONAL' | 'MARKETING' | ''>('');
  const [deleteTemplateId, setDeleteTemplateId] = useState<string | null>(null);

  const { data, isLoading, error } = useEmailTemplates({
    search: searchQuery || undefined,
    category: categoryFilter || undefined,
  });

  const deleteTemplateMutation = useDeleteEmailTemplate();

  const handleDelete = async () => {
    if (!deleteTemplateId) return;
    try {
      await deleteTemplateMutation.mutateAsync(deleteTemplateId);
      setDeleteTemplateId(null);
    } catch (err) {
      console.error('Failed to delete template:', err);
    }
  };

  const handleDuplicate = (templateId: string) => {
    // Navigate to new template page with source template
    router.push(`/dashboard/email/templates/new?duplicate=${templateId}`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Email Templates</h1>
          <p className="text-text-tertiary mt-1">
            Manage your email templates for transactional and marketing emails
          </p>
        </div>
        <Link href="/dashboard/email/templates/new">
          <Button>
            <Plus className="w-4 h-4 mr-2" />
            Create Template
          </Button>
        </Link>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary" />
          <input
            type="text"
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value as '' | 'TRANSACTIONAL' | 'MARKETING')}
          className="px-4 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">All Categories</option>
          <option value="TRANSACTIONAL">Transactional</option>
          <option value="MARKETING">Marketing</option>
        </select>
      </div>

      {/* Templates table */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-text-tertiary">Loading templates...</div>
        </div>
      ) : error ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-red-400">Error loading templates</div>
        </div>
      ) : !data?.templates?.length ? (
        <div className="flex flex-col items-center justify-center h-64 bg-background-secondary rounded-lg border border-border">
          <p className="text-text-tertiary mb-4">No email templates found</p>
          <Link href="/dashboard/email/templates/new">
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Create Your First Template
            </Button>
          </Link>
        </div>
      ) : (
        <div className="bg-background-secondary rounded-lg border border-border overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-4 py-3 text-sm font-medium text-text-tertiary">
                  Name
                </th>
                <th className="text-left px-4 py-3 text-sm font-medium text-text-tertiary">
                  Category
                </th>
                <th className="text-left px-4 py-3 text-sm font-medium text-text-tertiary">
                  Subject
                </th>
                <th className="text-left px-4 py-3 text-sm font-medium text-text-tertiary">
                  Status
                </th>
                <th className="text-left px-4 py-3 text-sm font-medium text-text-tertiary">
                  Updated
                </th>
                <th className="text-right px-4 py-3 text-sm font-medium text-text-tertiary">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {data.templates.map((template) => (
                <tr
                  key={template.id}
                  className="border-b border-border last:border-b-0 hover:bg-background-tertiary/50"
                >
                  <td className="px-4 py-3">
                    <div>
                      <p className="font-medium text-text-primary">{template.name}</p>
                      <p className="text-xs text-text-tertiary">{template.slug}</p>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge
                      variant={template.category === 'TRANSACTIONAL' ? 'default' : 'secondary'}
                    >
                      {template.category}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <p className="text-sm text-text-secondary truncate max-w-xs">
                      {template.subject}
                    </p>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={template.isActive ? 'default' : 'outline'}>
                      {template.isActive ? 'Active' : 'Inactive'}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-sm text-text-tertiary">
                    {format(new Date(template.updatedAt), 'MMM d, yyyy')}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-2">
                      <Link href={`/dashboard/email/templates/${template.id}/edit`}>
                        <Button variant="ghost" size="sm">
                          <Edit2 className="w-4 h-4" />
                        </Button>
                      </Link>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="bg-background-secondary border border-border rounded-md shadow-lg p-1 min-w-[160px]"
                        >
                          <DropdownMenuItem
                            onClick={() => handleDuplicate(template.id)}
                            className="flex items-center gap-2 px-3 py-2 text-sm text-text-primary hover:bg-background-tertiary rounded cursor-pointer"
                          >
                            <Copy className="w-4 h-4" />
                            Duplicate
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => router.push(`/dashboard/email/templates/${template.id}/edit?test=true`)}
                            className="flex items-center gap-2 px-3 py-2 text-sm text-text-primary hover:bg-background-tertiary rounded cursor-pointer"
                          >
                            <TestTube className="w-4 h-4" />
                            Send Test
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => setDeleteTemplateId(template.id)}
                            className="flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded cursor-pointer"
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Delete confirmation dialog */}
      <AlertDialog.Root open={!!deleteTemplateId} onOpenChange={() => setDeleteTemplateId(null)}>
        <AlertDialog.Portal>
          <AlertDialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <AlertDialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <AlertDialog.Title className="text-lg font-semibold text-text-primary">
              Delete Template
            </AlertDialog.Title>
            <AlertDialog.Description className="text-sm text-text-tertiary mt-2">
              Are you sure you want to delete this template? This action cannot be undone.
            </AlertDialog.Description>
            <div className="flex gap-3 mt-6 justify-end">
              <AlertDialog.Cancel asChild>
                <Button variant="outline">Cancel</Button>
              </AlertDialog.Cancel>
              <AlertDialog.Action asChild>
                <Button
                  variant="destructive"
                  onClick={handleDelete}
                  disabled={deleteTemplateMutation.isPending}
                >
                  {deleteTemplateMutation.isPending ? 'Deleting...' : 'Delete'}
                </Button>
              </AlertDialog.Action>
            </div>
          </AlertDialog.Content>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  );
}
