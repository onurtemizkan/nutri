'use client';

import { useMemo } from 'react';
import { X, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { TemplateVersion } from './VersionHistory';

interface TemplateDiffProps {
  versionA: TemplateVersion;
  versionB: TemplateVersion;
  onClose: () => void;
}

interface DiffLine {
  type: 'added' | 'removed' | 'unchanged';
  content: string;
  lineNumber: { a: number | null; b: number | null };
}

function computeDiff(textA: string, textB: string): DiffLine[] {
  const linesA = textA.split('\n');
  const linesB = textB.split('\n');
  const result: DiffLine[] = [];

  // Simple line-by-line diff algorithm
  // For production, consider using a proper diff library like 'diff'
  let indexA = 0;
  let indexB = 0;

  while (indexA < linesA.length || indexB < linesB.length) {
    if (indexA >= linesA.length) {
      // Remaining lines in B are additions
      result.push({
        type: 'added',
        content: linesB[indexB],
        lineNumber: { a: null, b: indexB + 1 },
      });
      indexB++;
    } else if (indexB >= linesB.length) {
      // Remaining lines in A are removals
      result.push({
        type: 'removed',
        content: linesA[indexA],
        lineNumber: { a: indexA + 1, b: null },
      });
      indexA++;
    } else if (linesA[indexA] === linesB[indexB]) {
      // Lines match
      result.push({
        type: 'unchanged',
        content: linesA[indexA],
        lineNumber: { a: indexA + 1, b: indexB + 1 },
      });
      indexA++;
      indexB++;
    } else {
      // Look ahead to find if line was moved
      const lookAheadB = linesB.slice(indexB + 1, indexB + 5).indexOf(linesA[indexA]);
      const lookAheadA = linesA.slice(indexA + 1, indexA + 5).indexOf(linesB[indexB]);

      if (lookAheadB !== -1 && (lookAheadA === -1 || lookAheadB <= lookAheadA)) {
        // Current line from A appears later in B, so lines between are additions
        for (let i = 0; i <= lookAheadB; i++) {
          result.push({
            type: 'added',
            content: linesB[indexB + i],
            lineNumber: { a: null, b: indexB + i + 1 },
          });
        }
        indexB += lookAheadB + 1;
      } else if (lookAheadA !== -1) {
        // Current line from B appears later in A, so lines between are removals
        for (let i = 0; i <= lookAheadA; i++) {
          result.push({
            type: 'removed',
            content: linesA[indexA + i],
            lineNumber: { a: indexA + i + 1, b: null },
          });
        }
        indexA += lookAheadA + 1;
      } else {
        // Lines are different
        result.push({
          type: 'removed',
          content: linesA[indexA],
          lineNumber: { a: indexA + 1, b: null },
        });
        result.push({
          type: 'added',
          content: linesB[indexB],
          lineNumber: { a: null, b: indexB + 1 },
        });
        indexA++;
        indexB++;
      }
    }
  }

  return result;
}

export function TemplateDiff({ versionA, versionB, onClose }: TemplateDiffProps) {
  const diff = useMemo(
    () => computeDiff(versionA.mjmlContent, versionB.mjmlContent),
    [versionA.mjmlContent, versionB.mjmlContent]
  );

  const stats = useMemo(() => {
    const added = diff.filter((d) => d.type === 'added').length;
    const removed = diff.filter((d) => d.type === 'removed').length;
    return { added, removed };
  }, [diff]);

  return (
    <div className="fixed inset-0 bg-background-primary z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
          <h2 className="text-lg font-semibold text-text-primary">
            Template Diff
          </h2>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <Badge variant="outline" className="font-mono">
              v{versionA.version}
            </Badge>
            <ArrowRight className="w-4 h-4 text-text-muted" />
            <Badge variant="outline" className="font-mono">
              v{versionB.version}
            </Badge>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-green-400">+{stats.added} added</span>
            <span className="text-red-400">-{stats.removed} removed</span>
          </div>
        </div>
      </div>

      {/* Subject comparison */}
      {versionA.subject !== versionB.subject && (
        <div className="px-4 py-2 border-b border-border bg-background-secondary">
          <div className="text-xs text-text-muted mb-1">Subject changed:</div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-red-400 line-through">{versionA.subject}</span>
            <ArrowRight className="w-4 h-4 text-text-muted flex-shrink-0" />
            <span className="text-green-400">{versionB.subject}</span>
          </div>
        </div>
      )}

      {/* Diff content */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm font-mono">
          <tbody>
            {diff.map((line, index) => (
              <tr
                key={index}
                className={
                  line.type === 'added'
                    ? 'bg-green-500/10'
                    : line.type === 'removed'
                    ? 'bg-red-500/10'
                    : ''
                }
              >
                <td className="w-12 text-right px-2 py-0.5 text-text-muted border-r border-border select-none">
                  {line.lineNumber.a || ''}
                </td>
                <td className="w-12 text-right px-2 py-0.5 text-text-muted border-r border-border select-none">
                  {line.lineNumber.b || ''}
                </td>
                <td className="w-6 text-center py-0.5 select-none">
                  {line.type === 'added' && (
                    <span className="text-green-400">+</span>
                  )}
                  {line.type === 'removed' && (
                    <span className="text-red-400">-</span>
                  )}
                </td>
                <td className="px-2 py-0.5 whitespace-pre">
                  <span
                    className={
                      line.type === 'added'
                        ? 'text-green-400'
                        : line.type === 'removed'
                        ? 'text-red-400'
                        : 'text-text-secondary'
                    }
                  >
                    {line.content}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TemplateDiff;
