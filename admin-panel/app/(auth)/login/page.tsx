'use client';

import { useState, Suspense } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter, useSearchParams } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';

interface LoginFormState {
  email: string;
  password: string;
  isLoading: boolean;
  error: string | null;
}

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get('callbackUrl') || '/dashboard';
  const urlError = searchParams.get('error');

  const [formState, setFormState] = useState<LoginFormState>({
    email: '',
    password: '',
    isLoading: false,
    error: urlError === 'CredentialsSignin' ? 'Invalid credentials' : null,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      // Step 1: Call the backend API directly to check for MFA requirement
      const loginResponse = await fetch(`${API_URL}/api/admin/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: formState.email,
          password: formState.password,
        }),
      });

      const loginData = await loginResponse.json();

      if (!loginResponse.ok) {
        setFormState((prev) => ({
          ...prev,
          isLoading: false,
          error: loginData.error || 'Invalid credentials',
        }));
        return;
      }

      // Step 2: Check if MFA is required
      if (loginData.requiresMFA) {
        // Store MFA data and redirect to MFA page
        sessionStorage.setItem('mfa_pending_token', loginData.pendingToken);
        sessionStorage.setItem('mfa_email', formState.email);
        if (loginData.qrCode) {
          sessionStorage.setItem('mfa_qr_code', loginData.qrCode);
        }
        sessionStorage.setItem(
          'mfa_setup_required',
          String(loginData.mfaSetupRequired || false)
        );
        router.push('/mfa');
        return;
      }

      // Step 3: No MFA required - create NextAuth session
      // This shouldn't normally happen for super admins, but handle it anyway
      const result = await signIn('credentials', {
        email: formState.email,
        password: formState.password,
        redirect: false,
      });

      if (result?.error) {
        setFormState((prev) => ({
          ...prev,
          isLoading: false,
          error: result.error || 'Login failed',
        }));
        return;
      }

      if (result?.ok) {
        router.push(callbackUrl);
      }
    } catch (error) {
      setFormState((prev) => ({
        ...prev,
        isLoading: false,
        error:
          error instanceof Error ? error.message : 'An unexpected error occurred',
      }));
    }
  };

  return (
    <div className="bg-card rounded-lg border border-border p-8 shadow-xl">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-bold text-text-primary">Nutri Admin</h1>
        <p className="mt-2 text-sm text-text-tertiary">
          Sign in to access the admin panel
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {formState.error && (
          <div className="rounded-md bg-error/10 border border-error/20 p-3 text-sm text-error">
            {formState.error}
          </div>
        )}

        <div>
          <label
            htmlFor="email"
            className="block text-sm font-medium text-text-secondary mb-2"
          >
            Email
          </label>
          <input
            id="email"
            type="email"
            value={formState.email}
            onChange={(e) =>
              setFormState((prev) => ({ ...prev, email: e.target.value }))
            }
            required
            autoComplete="email"
            className="w-full rounded-md border border-border bg-background-elevated px-4 py-2.5 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="admin@nutri.app"
          />
        </div>

        <div>
          <label
            htmlFor="password"
            className="block text-sm font-medium text-text-secondary mb-2"
          >
            Password
          </label>
          <input
            id="password"
            type="password"
            value={formState.password}
            onChange={(e) =>
              setFormState((prev) => ({ ...prev, password: e.target.value }))
            }
            required
            autoComplete="current-password"
            className="w-full rounded-md border border-border bg-background-elevated px-4 py-2.5 text-text-primary placeholder:text-text-disabled focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="••••••••"
          />
        </div>

        <button
          type="submit"
          disabled={formState.isLoading}
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
              Signing in...
            </span>
          ) : (
            'Sign In'
          )}
        </button>
      </form>

      <p className="mt-6 text-center text-xs text-text-disabled">
        Protected admin area. Unauthorized access is prohibited.
      </p>
    </div>
  );
}

function LoginFallback() {
  return (
    <div className="bg-card rounded-lg border border-border p-8 shadow-xl">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-bold text-text-primary">Nutri Admin</h1>
        <p className="mt-2 text-sm text-text-tertiary">Loading...</p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<LoginFallback />}>
      <LoginForm />
    </Suspense>
  );
}
