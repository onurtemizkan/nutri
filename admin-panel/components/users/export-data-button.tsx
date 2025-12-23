'use client';

import { useState } from 'react';
import { Download, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { adminApi } from '@/lib/api';

interface ExportDataButtonProps {
  userId: string;
  userName?: string;
}

export function ExportDataButton({ userId }: ExportDataButtonProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleExport = async () => {
    setIsExporting(true);
    setStatus('idle');
    setErrorMessage(null);

    try {
      const response = await adminApi.exportUserData(userId);

      // Create download link
      const blob = new Blob([JSON.stringify(response, null, 2)], {
        type: 'application/json',
      });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `user-${userId}-export-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setStatus('success');
      setTimeout(() => setStatus('idle'), 3000);
    } catch (error) {
      setStatus('error');
      setErrorMessage(
        error instanceof Error ? error.message : 'Export failed'
      );
      setTimeout(() => {
        setStatus('idle');
        setErrorMessage(null);
      }, 5000);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="relative">
      <Button
        variant="outline"
        onClick={handleExport}
        disabled={isExporting}
        className="gap-2"
      >
        {isExporting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Exporting...
          </>
        ) : status === 'success' ? (
          <>
            <CheckCircle className="h-4 w-4 text-green-500" />
            Exported
          </>
        ) : status === 'error' ? (
          <>
            <XCircle className="h-4 w-4 text-red-500" />
            Failed
          </>
        ) : (
          <>
            <Download className="h-4 w-4" />
            Export Data (GDPR)
          </>
        )}
      </Button>
      {errorMessage && (
        <p className="absolute left-0 mt-1 text-xs text-red-500">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
