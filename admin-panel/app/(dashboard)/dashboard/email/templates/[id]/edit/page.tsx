'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, useParams, useSearchParams } from 'next/navigation';
import { ArrowLeft, Save, Send, Eye, EyeOff, Trash2, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { MJMLEditor } from '@/components/email/MJMLEditor';
import { EmailPreview } from '@/components/email/EmailPreview';
import { VariablePicker } from '@/components/email/VariablePicker';
import {
  useEmailTemplate,
  useUpdateEmailTemplate,
  useDeleteEmailTemplate,
  useSendTestEmail,
} from '@/lib/hooks/useEmailTemplates';
import * as Dialog from '@radix-ui/react-dialog';
import * as AlertDialog from '@radix-ui/react-alert-dialog';

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
}

export default function EditEmailTemplatePage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const templateId = params.id as string;
  const openTestDialog = searchParams.get('test') === 'true';

  const [name, setName] = useState('');
  const [slug, setSlug] = useState('');
  const [category, setCategory] = useState<'TRANSACTIONAL' | 'MARKETING'>('TRANSACTIONAL');
  const [subject, setSubject] = useState('');
  const [mjmlContent, setMjmlContent] = useState('');
  const [mjmlErrors, setMjmlErrors] = useState<string[]>([]);
  const [showPreview, setShowPreview] = useState(true);
  const [testEmailDialogOpen, setTestEmailDialogOpen] = useState(openTestDialog);
  const [testEmail, setTestEmail] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [hasChanges, setHasChanges] = useState(false);

  // Load template data
  const { data: template, isLoading, error: loadError } = useEmailTemplate(templateId);

  const updateTemplateMutation = useUpdateEmailTemplate();
  const deleteTemplateMutation = useDeleteEmailTemplate();
  const sendTestEmailMutation = useSendTestEmail();

  // Populate form with template data
  useEffect(() => {
    if (template) {
      setName(template.name);
      setSlug(template.slug);
      setCategory(template.category);
      setSubject(template.subject);
      setMjmlContent(template.mjmlContent);
      setHasChanges(false);
    }
  }, [template]);

  const handleChange = useCallback((setter: (value: string) => void) => {
    return (value: string) => {
      setter(value);
      setHasChanges(true);
    };
  }, []);

  const handleSlugChange = useCallback((value: string) => {
    setSlug(slugify(value));
    setHasChanges(true);
  }, []);

  const handleCategoryChange = useCallback((value: 'TRANSACTIONAL' | 'MARKETING') => {
    setCategory(value);
    setHasChanges(true);
  }, []);

  const handleInsertVariable = useCallback((variable: string) => {
    navigator.clipboard.writeText(variable);
  }, []);

  const handleSave = async () => {
    setError('');
    setSuccess('');

    // Validation
    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    if (!slug.trim()) {
      setError('Slug is required');
      return;
    }
    if (!subject.trim()) {
      setError('Subject is required');
      return;
    }
    if (!mjmlContent.trim()) {
      setError('Template content is required');
      return;
    }
    if (mjmlErrors.length > 0) {
      setError('Please fix MJML errors before saving');
      return;
    }

    try {
      await updateTemplateMutation.mutateAsync({
        id: templateId,
        data: {
          name: name.trim(),
          slug: slug.trim(),
          category,
          subject: subject.trim(),
          mjmlContent,
        },
      });

      setSuccess('Template saved successfully');
      setHasChanges(false);
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save template');
    }
  };

  const handleDelete = async () => {
    try {
      await deleteTemplateMutation.mutateAsync(templateId);
      router.push('/dashboard/email/templates');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete template');
      setDeleteDialogOpen(false);
    }
  };

  const handleDuplicate = () => {
    router.push(`/dashboard/email/templates/new?duplicate=${templateId}`);
  };

  const handleSendTestEmail = async () => {
    if (!testEmail.trim()) return;

    try {
      await sendTestEmailMutation.mutateAsync({
        id: templateId,
        email: testEmail.trim(),
      });

      setTestEmailDialogOpen(false);
      setSuccess('Test email sent successfully');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send test email');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-text-tertiary">Loading template...</div>
      </div>
    );
  }

  if (loadError || !template) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <p className="text-red-400 mb-4">Template not found</p>
        <Button onClick={() => router.push('/dashboard/email/templates')}>
          Back to Templates
        </Button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-border">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push('/dashboard/email/templates')}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <h1 className="text-xl font-semibold text-text-primary">Edit Template</h1>
          <Badge variant={template.isActive ? 'default' : 'outline'}>
            {template.isActive ? 'Active' : 'Inactive'}
          </Badge>
          {hasChanges && (
            <Badge variant="secondary" className="text-yellow-400 bg-yellow-400/10">
              Unsaved changes
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? (
              <>
                <EyeOff className="w-4 h-4 mr-2" />
                Hide Preview
              </>
            ) : (
              <>
                <Eye className="w-4 h-4 mr-2" />
                Show Preview
              </>
            )}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleDuplicate}>
            <Copy className="w-4 h-4 mr-2" />
            Duplicate
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setDeleteDialogOpen(true)}
            className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
          <Button
            variant="outline"
            onClick={() => setTestEmailDialogOpen(true)}
          >
            <Send className="w-4 h-4 mr-2" />
            Test Email
          </Button>
          <Button
            onClick={handleSave}
            disabled={updateTemplateMutation.isPending || !hasChanges}
          >
            <Save className="w-4 h-4 mr-2" />
            {updateTemplateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      {/* Messages */}
      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-md text-red-400 text-sm">
          {error}
        </div>
      )}
      {success && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-md text-green-400 text-sm">
          {success}
        </div>
      )}

      {/* Form and preview */}
      <div className={`flex-1 mt-4 grid gap-4 ${showPreview ? 'grid-cols-2' : 'grid-cols-1'}`}>
        {/* Left side - Form */}
        <div className="flex flex-col gap-4 overflow-y-auto">
          {/* Basic fields */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => handleChange(setName)(e.target.value)}
                placeholder="Welcome Email"
                className="w-full px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Slug
              </label>
              <input
                type="text"
                value={slug}
                onChange={(e) => handleSlugChange(e.target.value)}
                placeholder="welcome-email"
                className="w-full px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Category
              </label>
              <select
                value={category}
                onChange={(e) => handleCategoryChange(e.target.value as 'TRANSACTIONAL' | 'MARKETING')}
                className="w-full px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
              >
                <option value="TRANSACTIONAL">Transactional</option>
                <option value="MARKETING">Marketing</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Subject
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={subject}
                  onChange={(e) => handleChange(setSubject)(e.target.value)}
                  placeholder="Welcome to Nutri, {{userName}}!"
                  className="flex-1 px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
                <VariablePicker onInsert={handleInsertVariable} buttonText="+" />
              </div>
            </div>
          </div>

          {/* MJML Editor */}
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <label className="text-sm font-medium text-text-secondary">
                MJML Content
              </label>
              <VariablePicker
                onInsert={handleInsertVariable}
                buttonText="Insert Variable"
              />
            </div>
            <MJMLEditor
              value={mjmlContent}
              onChange={(value) => {
                setMjmlContent(value);
                setHasChanges(true);
              }}
              onValidate={setMjmlErrors}
              height="calc(100vh - 420px)"
            />
            {mjmlErrors.length > 0 && (
              <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-yellow-400 text-xs">
                {mjmlErrors.map((err, i) => (
                  <p key={i}>{err}</p>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right side - Preview */}
        {showPreview && (
          <div className="border border-border rounded-lg overflow-hidden flex flex-col">
            <EmailPreview
              mjmlContent={mjmlContent}
              subject={subject}
              className="flex-1"
            />
          </div>
        )}
      </div>

      {/* Test email dialog */}
      <Dialog.Root open={testEmailDialogOpen} onOpenChange={setTestEmailDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <Dialog.Title className="text-lg font-semibold text-text-primary">
              Send Test Email
            </Dialog.Title>
            <div className="mt-4">
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Email Address
              </label>
              <input
                type="email"
                value={testEmail}
                onChange={(e) => setTestEmail(e.target.value)}
                placeholder="test@example.com"
                className="w-full px-3 py-2 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <div className="flex gap-3 mt-6 justify-end">
              <Button variant="outline" onClick={() => setTestEmailDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleSendTestEmail}
                disabled={!testEmail.trim() || sendTestEmailMutation.isPending}
              >
                {sendTestEmailMutation.isPending ? 'Sending...' : 'Send Test'}
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      {/* Delete confirmation dialog */}
      <AlertDialog.Root open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialog.Portal>
          <AlertDialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <AlertDialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-background-secondary border border-border rounded-lg shadow-xl p-6 max-w-md w-full z-50">
            <AlertDialog.Title className="text-lg font-semibold text-text-primary">
              Delete Template
            </AlertDialog.Title>
            <AlertDialog.Description className="text-sm text-text-tertiary mt-2">
              Are you sure you want to delete &quot;{template.name}&quot;? This action cannot be undone.
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
                  {deleteTemplateMutation.isPending ? 'Deleting...' : 'Delete Template'}
                </Button>
              </AlertDialog.Action>
            </div>
          </AlertDialog.Content>
        </AlertDialog.Portal>
      </AlertDialog.Root>
    </div>
  );
}
