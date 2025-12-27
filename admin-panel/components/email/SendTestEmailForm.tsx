'use client';

import { useState } from 'react';
import { Send, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { SAMPLE_DATA_PRESETS, getSampleDataByKey } from '@/lib/emailSampleData';

interface SendTestEmailFormProps {
  onSend: (email: string, sampleData: Record<string, string | number | boolean>) => Promise<void>;
  disabled?: boolean;
}

export function SendTestEmailForm({ onSend, disabled }: SendTestEmailFormProps) {
  const [email, setEmail] = useState('');
  const [selectedPreset, setSelectedPreset] = useState('active_user');
  const [customData] = useState<Record<string, string>>({});
  const [showCustomData, setShowCustomData] = useState(false);
  const [status, setStatus] = useState<'idle' | 'sending' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleSend = async () => {
    if (!email.trim()) return;

    setStatus('sending');
    setErrorMessage('');

    try {
      const sampleData = getSampleDataByKey(selectedPreset);
      // Merge custom data overrides
      const mergedData = { ...sampleData, ...customData };

      await onSend(email, mergedData);
      setStatus('success');

      // Reset to idle after 3 seconds
      setTimeout(() => setStatus('idle'), 3000);
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to send test email');
    }
  };

  const selectedPresetData = SAMPLE_DATA_PRESETS.find(p => p.key === selectedPreset);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-text-secondary mb-1">
          Recipient Email
        </label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="test@example.com"
          disabled={disabled || status === 'sending'}
          className="w-full px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-text-secondary mb-1">
          Sample Data Preset
        </label>
        <select
          value={selectedPreset}
          onChange={(e) => setSelectedPreset(e.target.value)}
          disabled={disabled || status === 'sending'}
          className="w-full px-3 py-2 bg-background-secondary border border-border rounded-md text-text-primary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
        >
          {SAMPLE_DATA_PRESETS.map((preset) => (
            <option key={preset.key} value={preset.key}>
              {preset.label}
            </option>
          ))}
        </select>
        {selectedPresetData && (
          <p className="mt-1 text-xs text-text-muted">
            {selectedPresetData.description}
          </p>
        )}
      </div>

      <div>
        <button
          type="button"
          onClick={() => setShowCustomData(!showCustomData)}
          className="text-sm text-primary hover:text-primary/80"
        >
          {showCustomData ? 'Hide' : 'Show'} data preview
        </button>

        {showCustomData && selectedPresetData && (
          <div className="mt-2 p-3 bg-background-secondary rounded-md border border-border max-h-48 overflow-y-auto">
            <pre className="text-xs text-text-secondary font-mono">
              {JSON.stringify(selectedPresetData.data, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {status === 'error' && errorMessage && (
        <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-md text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {errorMessage}
        </div>
      )}

      {status === 'success' && (
        <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/20 rounded-md text-green-400 text-sm">
          <CheckCircle className="w-4 h-4 flex-shrink-0" />
          Test email sent successfully!
        </div>
      )}

      <Button
        onClick={handleSend}
        disabled={disabled || !email.trim() || status === 'sending'}
        className="w-full"
      >
        {status === 'sending' ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Sending...
          </>
        ) : (
          <>
            <Send className="w-4 h-4 mr-2" />
            Send Test Email
          </>
        )}
      </Button>
    </div>
  );
}

export default SendTestEmailForm;
