'use client';

import { useState, useEffect, useRef } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';

interface MFAFormState {
  code: string;
  isLoading: boolean;
  error: string | null;
}

export default function MFAPage() {
  const router = useRouter();
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const [pendingToken, setPendingToken] = useState<string | null>(null);
  const [qrCode, setQrCode] = useState<string | null>(null);
  const [isSetup, setIsSetup] = useState(false);

  const [formState, setFormState] = useState<MFAFormState>({
    code: '',
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    // Get pending token from session storage
    const token = sessionStorage.getItem('mfa_pending_token');
    const qr = sessionStorage.getItem('mfa_qr_code');

    if (!token) {
      // No pending token, redirect to login
      router.push('/login');
      return;
    }

    setPendingToken(token);
    if (qr) {
      setQrCode(qr);
      setIsSetup(true);
    }
  }, [router]);

  const handleCodeChange = (index: number, value: string) => {
    // Only allow digits
    const digit = value.replace(/\D/g, '').slice(-1);

    // Update the code
    const newCode = formState.code.split('');
    newCode[index] = digit;
    const updatedCode = newCode.join('').slice(0, 6);

    setFormState((prev) => ({ ...prev, code: updatedCode, error: null }));

    // Auto-focus next input
    if (digit && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    // Handle backspace to go to previous input
    if (e.key === 'Backspace' && !formState.code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedText = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    setFormState((prev) => ({ ...prev, code: pastedText, error: null }));

    // Focus the appropriate input based on pasted length
    const nextIndex = Math.min(pastedText.length, 5);
    inputRefs.current[nextIndex]?.focus();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formState.code.length !== 6) {
      setFormState((prev) => ({
        ...prev,
        error: 'Please enter a 6-digit code',
      }));
      return;
    }

    setFormState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const result = await signIn('credentials', {
        pendingToken: pendingToken,
        mfaCode: formState.code,
        redirect: false,
      });

      if (result?.error) {
        setFormState((prev) => ({
          ...prev,
          isLoading: false,
          error: 'Invalid verification code',
          code: '', // Clear the code on error
        }));
        // Focus the first input
        inputRefs.current[0]?.focus();
        return;
      }

      if (result?.ok) {
        // Clear session storage
        sessionStorage.removeItem('mfa_pending_token');
        sessionStorage.removeItem('mfa_qr_code');
        router.push('/dashboard');
      }
    } catch (error) {
      setFormState((prev) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Verification failed',
      }));
    }
  };

  const handleBackToLogin = () => {
    sessionStorage.removeItem('mfa_pending_token');
    sessionStorage.removeItem('mfa_qr_code');
    router.push('/login');
  };

  if (!pendingToken) {
    return (
      <div className="bg-card rounded-lg border border-border p-8 shadow-xl text-center">
        <p className="text-text-tertiary">Redirecting to login...</p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-lg border border-border p-8 shadow-xl">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-bold text-text-primary">
          {isSetup ? 'Set Up Two-Factor Authentication' : 'Two-Factor Authentication'}
        </h1>
        <p className="mt-2 text-sm text-text-tertiary">
          {isSetup
            ? 'Scan the QR code with your authenticator app, then enter the code'
            : 'Enter the 6-digit code from your authenticator app'}
        </p>
      </div>

      {isSetup && qrCode && (
        <div className="mb-8 flex justify-center">
          <div className="bg-white p-4 rounded-lg">
            <Image
              src={qrCode}
              alt="MFA QR Code"
              width={200}
              height={200}
              className="rounded"
            />
          </div>
        </div>
      )}

      {isSetup && (
        <div className="mb-6 p-4 rounded-md bg-warning/10 border border-warning/20">
          <p className="text-sm text-warning">
            <strong>Important:</strong> After scanning, enter the 6-digit code from your
            authenticator app to complete setup. You will need this app to log in.
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {formState.error && (
          <div className="rounded-md bg-error/10 border border-error/20 p-3 text-sm text-error">
            {formState.error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-4 text-center">
            Verification Code
          </label>
          <div
            className="flex justify-center gap-2"
            onPaste={handlePaste}
          >
            {[0, 1, 2, 3, 4, 5].map((index) => (
              <input
                key={index}
                ref={(el) => {
                  inputRefs.current[index] = el;
                }}
                type="text"
                inputMode="numeric"
                maxLength={1}
                value={formState.code[index] || ''}
                onChange={(e) => handleCodeChange(index, e.target.value)}
                onKeyDown={(e) => handleKeyDown(index, e)}
                className="w-12 h-14 text-center text-2xl font-bold rounded-md border border-border bg-background-elevated text-text-primary focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                autoFocus={index === 0}
                disabled={formState.isLoading}
              />
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={formState.isLoading || formState.code.length !== 6}
          className="w-full rounded-md bg-primary px-4 py-2.5 font-medium text-white hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {formState.isLoading ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Verifying...
            </span>
          ) : (
            'Verify'
          )}
        </button>

        <button
          type="button"
          onClick={handleBackToLogin}
          className="w-full text-sm text-text-tertiary hover:text-text-secondary transition-colors"
        >
          Back to login
        </button>
      </form>
    </div>
  );
}
