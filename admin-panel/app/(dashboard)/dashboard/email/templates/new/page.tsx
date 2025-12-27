'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, Save, Send, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { MJMLEditor, DEFAULT_MJML_TEMPLATE } from '@/components/email/MJMLEditor';
import { EmailPreview } from '@/components/email/EmailPreview';
import { VariablePicker } from '@/components/email/VariablePicker';
import {
  useCreateEmailTemplate,
  useEmailTemplate,
  useSendTestEmail,
} from '@/lib/hooks/useEmailTemplates';
import * as Dialog from '@radix-ui/react-dialog';

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .trim();
}

export default function NewEmailTemplatePage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const duplicateId = searchParams.get('duplicate');

  const [name, setName] = useState('');
  const [slug, setSlug] = useState('');
  const [slugEdited, setSlugEdited] = useState(false);
  const [category, setCategory] = useState<'TRANSACTIONAL' | 'MARKETING'>('TRANSACTIONAL');
  const [subject, setSubject] = useState('');
  const [mjmlContent, setMjmlContent] = useState(DEFAULT_MJML_TEMPLATE);
  const [mjmlErrors, setMjmlErrors] = useState<string[]>([]);
  const [showPreview, setShowPreview] = useState(true);
  const [testEmailDialogOpen, setTestEmailDialogOpen] = useState(false);
  const [testEmail, setTestEmail] = useState('');
  const [error, setError] = useState('');

  // Load template to duplicate
  const { data: sourceTemplate } = useEmailTemplate(duplicateId || '');

  useEffect(() => {
    if (sourceTemplate) {
      setName(`${sourceTemplate.name} (Copy)`);
      setSlug(`${sourceTemplate.slug}-copy`);
      setSlugEdited(true);
      setCategory(sourceTemplate.category);
      setSubject(sourceTemplate.subject);
      setMjmlContent(sourceTemplate.mjmlContent);
    }
  }, [sourceTemplate]);

  const createTemplateMutation = useCreateEmailTemplate();
  const sendTestEmailMutation = useSendTestEmail();

  // Auto-generate slug from name
  useEffect(() => {
    if (!slugEdited && name) {
      setSlug(slugify(name));
    }
  }, [name, slugEdited]);

  const handleSlugChange = useCallback((value: string) => {
    setSlug(slugify(value));
    setSlugEdited(true);
  }, []);

  const handleInsertVariable = useCallback((variable: string) => {
    // This would insert at cursor position in a real implementation
    // For now, we'll append to the subject if focused, or copy to clipboard
    navigator.clipboard.writeText(variable);
  }, []);

  const handleSave = async () => {
    setError('');

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
      const template = await createTemplateMutation.mutateAsync({
        name: name.trim(),
        slug: slug.trim(),
        category,
        subject: subject.trim(),
        mjmlContent,
      });

      router.push(`/dashboard/email/templates/${template.id}/edit`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create template');
    }
  };

  const handleSendTestEmail = async () => {
    if (!testEmail.trim()) return;

    try {
      // We need to save the template first before sending test email
      setError('Please save the template first before sending a test email');
      setTestEmailDialogOpen(false);
    } catch (err) {
      console.error('Failed to send test email:', err);
    }
  };

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
          <h1 className="text-xl font-semibold text-text-primary">
            {duplicateId ? 'Duplicate Template' : 'New Email Template'}
          </h1>
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
          <Button
            variant="outline"
            onClick={() => setTestEmailDialogOpen(true)}
            disabled={!mjmlContent || createTemplateMutation.isPending}
          >
            <Send className="w-4 h-4 mr-2" />
            Test Email
          </Button>
          <Button
            onClick={() => handleSave()}
            disabled={createTemplateMutation.isPending}
          >
            <Save className="w-4 h-4 mr-2" />
            {createTemplateMutation.isPending ? 'Saving...' : 'Save Template'}
          </Button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-md text-red-400 text-sm">
          {error}
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
                onChange={(e) => setName(e.target.value)}
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
                onChange={(e) => setCategory(e.target.value as 'TRANSACTIONAL' | 'MARKETING')}
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
                  onChange={(e) => setSubject(e.target.value)}
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
              onChange={setMjmlContent}
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
    </div>
  );
}
