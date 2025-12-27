'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Monitor, Smartphone, Moon, Sun, RefreshCw, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface EmailPreviewProps {
  mjmlContent: string;
  subject?: string;
  testData?: Record<string, unknown>;
  className?: string;
}

// Client-side MJML compilation
async function compileMJML(mjml: string): Promise<{ html: string; errors: string[] }> {
  try {
    // Dynamic import for client-side only
    const mjml2html = (await import('mjml-browser')).default;
    const result = mjml2html(mjml, {
      validationLevel: 'soft',
      minify: false,
    });

    return {
      html: result.html,
      errors: result.errors?.map((e: { message: string }) => e.message) || [],
    };
  } catch (error) {
    return {
      html: '',
      errors: [error instanceof Error ? error.message : 'Failed to compile MJML'],
    };
  }
}

// Replace template variables with test data
function replaceVariables(
  html: string,
  testData: Record<string, unknown>
): string {
  let result = html;

  // Replace {{variable}} patterns
  for (const [key, value] of Object.entries(testData)) {
    const regex = new RegExp(`\\{\\{${key}\\}\\}`, 'g');
    result = result.replace(regex, String(value));
  }

  // Replace any remaining {{variable}} with placeholder
  result = result.replace(/\{\{(\w+)\}\}/g, '[$1]');

  return result;
}

// Default test data for preview
const DEFAULT_TEST_DATA: Record<string, unknown> = {
  userName: 'John Doe',
  email: 'john@example.com',
  firstName: 'John',
  lastName: 'Doe',
  goalProgress: '75%',
  caloriesConsumed: '1850',
  caloriesGoal: '2000',
  proteinConsumed: '120g',
  proteinGoal: '150g',
  actionUrl: 'https://nutri.app/action',
  unsubscribeUrl: 'https://nutri.app/unsubscribe',
  preferencesUrl: 'https://nutri.app/preferences',
  appName: 'Nutri',
};

export function EmailPreview({
  mjmlContent,
  subject,
  testData = DEFAULT_TEST_DATA,
  className = '',
}: EmailPreviewProps) {
  const [viewMode, setViewMode] = useState<'desktop' | 'mobile'>('desktop');
  const [darkMode, setDarkMode] = useState(false);
  const [compiledHtml, setCompiledHtml] = useState('');
  const [errors, setErrors] = useState<string[]>([]);
  const [isCompiling, setIsCompiling] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  // Compile MJML with debounce
  useEffect(() => {
    if (!mjmlContent) {
      setCompiledHtml('');
      setErrors([]);
      return;
    }

    const timeoutId = setTimeout(async () => {
      setIsCompiling(true);
      const result = await compileMJML(mjmlContent);
      setCompiledHtml(result.html);
      setErrors(result.errors);
      setIsCompiling(false);
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [mjmlContent, refreshKey]);

  // Apply test data to HTML
  const previewHtml = useMemo(() => {
    if (!compiledHtml) return '';
    return replaceVariables(compiledHtml, { ...DEFAULT_TEST_DATA, ...testData });
  }, [compiledHtml, testData]);

  // Calculate iframe dimensions based on view mode
  const iframeWidth = viewMode === 'desktop' ? '100%' : '375px';
  const iframeMaxWidth = viewMode === 'desktop' ? '600px' : '375px';

  const handleRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  const handleOpenInNewTab = useCallback(() => {
    const blob = new Blob([previewHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [previewHtml]);

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-background-secondary border-b border-border">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode('desktop')}
            className={viewMode === 'desktop' ? 'bg-background-tertiary' : ''}
          >
            <Monitor className="w-4 h-4 mr-1" />
            Desktop
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode('mobile')}
            className={viewMode === 'mobile' ? 'bg-background-tertiary' : ''}
          >
            <Smartphone className="w-4 h-4 mr-1" />
            Mobile
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleRefresh}>
            <RefreshCw className={`w-4 h-4 ${isCompiling ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleOpenInNewTab}
            disabled={!previewHtml}
          >
            <ExternalLink className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Subject preview */}
      {subject && (
        <div className="px-4 py-2 bg-background-secondary border-b border-border">
          <span className="text-xs text-text-tertiary">Subject: </span>
          <span className="text-sm text-text-primary font-medium">
            {replaceVariables(subject, { ...DEFAULT_TEST_DATA, ...testData })}
          </span>
        </div>
      )}

      {/* Errors */}
      {errors.length > 0 && (
        <div className="p-3 bg-red-500/10 border-b border-red-500/20">
          <p className="text-xs font-medium text-red-400 mb-1">MJML Errors:</p>
          <ul className="text-xs text-red-300 list-disc list-inside">
            {errors.map((error, i) => (
              <li key={i}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Preview */}
      <div
        className={`flex-1 overflow-auto p-4 ${
          darkMode ? 'bg-gray-900' : 'bg-gray-100'
        }`}
      >
        <div
          className="mx-auto transition-all duration-200"
          style={{ maxWidth: iframeMaxWidth, width: iframeWidth }}
        >
          {isCompiling ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-text-tertiary">Compiling...</div>
            </div>
          ) : previewHtml ? (
            <iframe
              srcDoc={previewHtml}
              className="w-full bg-white rounded shadow-lg"
              style={{ minHeight: '500px', height: '100%' }}
              title="Email Preview"
              sandbox="allow-same-origin"
            />
          ) : (
            <div className="flex items-center justify-center h-64 bg-white rounded shadow-lg">
              <p className="text-text-tertiary">Enter MJML content to see preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default EmailPreview;
