'use client';

import { useRef, useCallback } from 'react';
import Editor, { OnMount, OnChange } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';

interface MJMLEditorProps {
  value: string;
  onChange: (value: string) => void;
  onValidate?: (errors: string[]) => void;
  height?: string;
  readOnly?: boolean;
}

// Default MJML template for new templates
export const DEFAULT_MJML_TEMPLATE = `<mjml>
  <mj-head>
    <mj-title>Email Title</mj-title>
    <mj-preview>Preview text shown in email clients</mj-preview>
    <mj-attributes>
      <mj-all font-family="Arial, sans-serif" />
      <mj-text font-size="14px" color="#333333" line-height="1.5" />
      <mj-section background-color="#ffffff" padding="20px" />
    </mj-attributes>
    <mj-style>
      .link-button {
        color: #ffffff !important;
        text-decoration: none;
      }
    </mj-style>
  </mj-head>
  <mj-body background-color="#f4f4f4">
    <mj-section background-color="#10b981" padding="20px 0">
      <mj-column>
        <mj-text align="center" color="#ffffff" font-size="24px" font-weight="bold">
          Nutri
        </mj-text>
      </mj-column>
    </mj-section>

    <mj-section>
      <mj-column>
        <mj-text>
          Hello {{userName}},
        </mj-text>
        <mj-text>
          Your email content goes here. You can use variables like {{userName}}, {{email}}, etc.
        </mj-text>
        <mj-button background-color="#10b981" href="{{actionUrl}}">
          Take Action
        </mj-button>
      </mj-column>
    </mj-section>

    <mj-section background-color="#333333" padding="20px">
      <mj-column>
        <mj-text align="center" color="#999999" font-size="12px">
          You're receiving this email because you signed up for Nutri.
          <br />
          <a href="{{unsubscribeUrl}}" style="color: #999999;">Unsubscribe</a>
        </mj-text>
      </mj-column>
    </mj-section>
  </mj-body>
</mjml>`;

export function MJMLEditor({
  value,
  onChange,
  onValidate,
  height = '500px',
  readOnly = false,
}: MJMLEditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const handleEditorDidMount: OnMount = useCallback((editor) => {
    editorRef.current = editor;

    // Configure editor
    editor.updateOptions({
      minimap: { enabled: false },
      fontSize: 13,
      lineNumbers: 'on',
      wordWrap: 'on',
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 2,
    });
  }, []);

  const handleChange: OnChange = useCallback(
    (newValue) => {
      if (newValue !== undefined) {
        onChange(newValue);

        // Basic MJML validation
        if (onValidate) {
          const errors: string[] = [];

          if (!newValue.includes('<mjml>')) {
            errors.push('Missing <mjml> root tag');
          }
          if (!newValue.includes('<mj-body>')) {
            errors.push('Missing <mj-body> tag');
          }
          if (!newValue.includes('</mjml>')) {
            errors.push('Missing closing </mjml> tag');
          }

          // Check for unclosed tags (basic check)
          const openTags = (newValue.match(/<mj-[a-z-]+[^/>]*>/g) || []).length;
          const closeTags = (newValue.match(/<\/mj-[a-z-]+>/g) || []).length;
          const selfClosing = (newValue.match(/<mj-[a-z-]+[^>]*\/>/g) || []).length;

          if (openTags !== closeTags + selfClosing) {
            errors.push('Possible unclosed MJML tags');
          }

          onValidate(errors);
        }
      }
    },
    [onChange, onValidate]
  );

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Editor
        height={height}
        defaultLanguage="html"
        value={value}
        onChange={handleChange}
        onMount={handleEditorDidMount}
        theme="vs-dark"
        options={{
          readOnly,
          minimap: { enabled: false },
        }}
      />
    </div>
  );
}

// Export insert helper for external use
export function useMJMLEditorRef() {
  const editorRef = useRef<{
    insertAtCursor: (text: string) => void;
  } | null>(null);

  return editorRef;
}

export default MJMLEditor;
