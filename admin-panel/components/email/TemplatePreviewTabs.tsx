'use client';

import { useState, useCallback } from 'react';
import { Monitor, Smartphone, Moon, Sun, User } from 'lucide-react';
import { EmailPreview } from './EmailPreview';
import { SAMPLE_DATA_PRESETS, getSampleDataByKey } from '@/lib/emailSampleData';

interface TemplatePreviewTabsProps {
  mjmlContent: string;
  subject: string;
  className?: string;
}

type DeviceMode = 'desktop' | 'mobile';
type ThemeMode = 'light' | 'dark';

export function TemplatePreviewTabs({
  mjmlContent,
  subject,
  className,
}: TemplatePreviewTabsProps) {
  const [deviceMode, setDeviceMode] = useState<DeviceMode>('desktop');
  const [themeMode, setThemeMode] = useState<ThemeMode>('light');
  const [selectedPreset, setSelectedPreset] = useState('active_user');

  const sampleData = getSampleDataByKey(selectedPreset);

  // Replace template variables with sample data
  const processedContent = useCallback(() => {
    let content = mjmlContent;
    let processedSubject = subject;

    Object.entries(sampleData).forEach(([key, value]) => {
      const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
      content = content.replace(regex, String(value));
      processedSubject = processedSubject.replace(regex, String(value));
    });

    return { content, subject: processedSubject };
  }, [mjmlContent, subject, sampleData]);

  const { content: processedMjml, subject: processedSubject } = processedContent();

  return (
    <div className={`flex flex-col ${className || ''}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-background-secondary">
        {/* Device & Theme toggles */}
        <div className="flex items-center gap-2">
          <div className="flex rounded-md border border-border overflow-hidden">
            <button
              onClick={() => setDeviceMode('desktop')}
              className={`px-3 py-1.5 flex items-center gap-1.5 text-xs ${
                deviceMode === 'desktop'
                  ? 'bg-primary text-white'
                  : 'bg-background-primary text-text-secondary hover:bg-background-secondary'
              }`}
            >
              <Monitor className="w-3.5 h-3.5" />
              Desktop
            </button>
            <button
              onClick={() => setDeviceMode('mobile')}
              className={`px-3 py-1.5 flex items-center gap-1.5 text-xs border-l border-border ${
                deviceMode === 'mobile'
                  ? 'bg-primary text-white'
                  : 'bg-background-primary text-text-secondary hover:bg-background-secondary'
              }`}
            >
              <Smartphone className="w-3.5 h-3.5" />
              Mobile
            </button>
          </div>

          <button
            onClick={() => setThemeMode(themeMode === 'light' ? 'dark' : 'light')}
            className="p-1.5 rounded-md border border-border hover:bg-background-secondary"
            title={`Switch to ${themeMode === 'light' ? 'dark' : 'light'} mode`}
          >
            {themeMode === 'light' ? (
              <Moon className="w-4 h-4 text-text-secondary" />
            ) : (
              <Sun className="w-4 h-4 text-text-secondary" />
            )}
          </button>
        </div>

        {/* Sample Data Selector */}
        <div className="flex items-center gap-2">
          <User className="w-4 h-4 text-text-muted" />
          <select
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value)}
            className="text-xs px-2 py-1 bg-background-primary border border-border rounded-md text-text-primary focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            {SAMPLE_DATA_PRESETS.map((preset) => (
              <option key={preset.key} value={preset.key}>
                {preset.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Subject line preview */}
      <div className="px-4 py-2 border-b border-border bg-background-secondary">
        <div className="text-xs text-text-muted mb-1">Subject:</div>
        <div className="text-sm text-text-primary font-medium">
          {processedSubject || '(No subject)'}
        </div>
      </div>

      {/* Preview */}
      <div
        className={`flex-1 overflow-auto ${
          themeMode === 'dark' ? 'bg-gray-900' : 'bg-white'
        }`}
      >
        <div
          className={`mx-auto transition-all duration-300 ${
            deviceMode === 'mobile' ? 'max-w-[375px]' : 'max-w-full'
          }`}
        >
          <EmailPreview
            mjmlContent={processedMjml}
            subject={processedSubject}
            className="h-full"
          />
        </div>
      </div>
    </div>
  );
}

export default TemplatePreviewTabs;
