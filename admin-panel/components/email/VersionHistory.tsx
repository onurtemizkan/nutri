'use client';

import { useState } from 'react';
import { History, ChevronDown, ChevronRight, Eye, RotateCcw, GitCompare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { formatDistanceToNow } from 'date-fns';

export interface TemplateVersion {
  id: string;
  version: number;
  mjmlContent: string;
  subject: string;
  createdAt: string;
  createdBy?: {
    id: string;
    email: string;
  };
  changeNotes?: string;
}

interface VersionHistoryProps {
  versions: TemplateVersion[];
  currentVersion: number;
  onPreview: (version: TemplateVersion) => void;
  onRestore: (version: TemplateVersion) => void;
  onCompare: (versionA: TemplateVersion, versionB: TemplateVersion) => void;
  isLoading?: boolean;
}

export function VersionHistory({
  versions,
  currentVersion,
  onPreview,
  onRestore,
  onCompare,
  isLoading,
}: VersionHistoryProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [selectedForCompare, setSelectedForCompare] = useState<TemplateVersion | null>(null);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        <History className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>No version history available</p>
      </div>
    );
  }

  const handleCompareClick = (version: TemplateVersion) => {
    if (selectedForCompare) {
      if (selectedForCompare.id !== version.id) {
        onCompare(selectedForCompare, version);
      }
      setSelectedForCompare(null);
    } else {
      setSelectedForCompare(version);
    }
  };

  return (
    <div className="space-y-2">
      {selectedForCompare && (
        <div className="p-2 bg-blue-500/10 border border-blue-500/20 rounded-md text-blue-400 text-sm flex items-center justify-between">
          <span>
            Select another version to compare with v{selectedForCompare.version}
          </span>
          <button
            onClick={() => setSelectedForCompare(null)}
            className="text-blue-400 hover:text-blue-300"
          >
            Cancel
          </button>
        </div>
      )}

      {versions.map((version) => {
        const isExpanded = expandedId === version.id;
        const isCurrent = version.version === currentVersion;
        const isSelected = selectedForCompare?.id === version.id;

        return (
          <div
            key={version.id}
            className={`border rounded-md transition-colors ${
              isSelected
                ? 'border-blue-500/50 bg-blue-500/5'
                : isCurrent
                ? 'border-primary/50 bg-primary/5'
                : 'border-border bg-background-secondary'
            }`}
          >
            <button
              onClick={() => setExpandedId(isExpanded ? null : version.id)}
              className="w-full p-3 flex items-center justify-between text-left"
            >
              <div className="flex items-center gap-3">
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-text-muted" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-text-muted" />
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-text-primary">
                      Version {version.version}
                    </span>
                    {isCurrent && (
                      <Badge variant="success">Current</Badge>
                    )}
                  </div>
                  <p className="text-xs text-text-muted mt-0.5">
                    {formatDistanceToNow(new Date(version.createdAt), { addSuffix: true })}
                    {version.createdBy && ` by ${version.createdBy.email}`}
                  </p>
                </div>
              </div>
            </button>

            {isExpanded && (
              <div className="px-3 pb-3 border-t border-border mt-2 pt-3">
                {version.changeNotes && (
                  <p className="text-sm text-text-secondary mb-3">
                    {version.changeNotes}
                  </p>
                )}
                <div className="text-xs text-text-muted mb-3">
                  <strong>Subject:</strong> {version.subject}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onPreview(version)}
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    Preview
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleCompareClick(version)}
                    className={isSelected ? 'bg-blue-500/10' : ''}
                  >
                    <GitCompare className="w-3 h-3 mr-1" />
                    {isSelected ? 'Comparing...' : 'Compare'}
                  </Button>
                  {!isCurrent && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onRestore(version)}
                    >
                      <RotateCcw className="w-3 h-3 mr-1" />
                      Restore
                    </Button>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default VersionHistory;
