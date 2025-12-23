import NextAuth, { type NextAuthConfig } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import { AdminRole, SessionUser } from './types';

/**
 * NextAuth v5 configuration for admin panel
 * Uses credentials provider with backend API
 */

// Extend NextAuth types
declare module 'next-auth' {
  interface User {
    id: string;
    email: string;
    name: string;
    role: AdminRole;
    accessToken: string;
    requiresMFA?: boolean;
    mfaSetupRequired?: boolean;
    pendingToken?: string;
    qrCode?: string;
  }

  interface Session {
    user: SessionUser;
    accessToken: string;
  }
}

declare module '@auth/core/jwt' {
  interface JWT {
    id: string;
    email: string;
    name: string;
    role: AdminRole;
    accessToken: string;
  }
}

const API_URL = process.env.API_URL || 'http://localhost:3000';

export const authConfig: NextAuthConfig = {
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
        pendingToken: { label: 'Pending Token', type: 'text' },
        mfaCode: { label: 'MFA Code', type: 'text' },
      },
      async authorize(credentials) {
        if (!credentials) {
          return null;
        }

        const { email, password, pendingToken, mfaCode } = credentials;

        try {
          // If we have a pending token and MFA code, verify MFA
          if (pendingToken && mfaCode) {
            const mfaResponse = await fetch(
              `${API_URL}/api/admin/auth/mfa/verify`,
              {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  pendingToken,
                  code: mfaCode,
                }),
              }
            );

            if (!mfaResponse.ok) {
              const error = await mfaResponse.json();
              throw new Error(error.error || 'MFA verification failed');
            }

            const mfaData = await mfaResponse.json();

            // After MFA, get user info
            const meResponse = await fetch(`${API_URL}/api/admin/auth/me`, {
              headers: {
                Authorization: `Bearer ${mfaData.token}`,
              },
            });

            if (!meResponse.ok) {
              throw new Error('Failed to get user info');
            }

            const userData = await meResponse.json();

            return {
              id: userData.id,
              email: userData.email,
              name: userData.name,
              role: userData.role as AdminRole,
              accessToken: mfaData.token,
            };
          }

          // Standard login flow
          if (!email || !password) {
            return null;
          }

          const response = await fetch(`${API_URL}/api/admin/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          });

          if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Invalid credentials');
          }

          const data = await response.json();

          // If MFA is required, return special object that signals MFA flow
          if (data.requiresMFA) {
            // Return a special object to indicate MFA is required
            // The frontend will handle this and redirect to MFA page
            return {
              id: 'pending-mfa',
              email: email as string,
              name: 'Pending MFA',
              role: 'VIEWER' as AdminRole,
              accessToken: '',
              requiresMFA: true,
              mfaSetupRequired: data.mfaSetupRequired || false,
              pendingToken: data.pendingToken,
              qrCode: data.qrCode,
            };
          }

          // No MFA required, get user info
          const meResponse = await fetch(`${API_URL}/api/admin/auth/me`, {
            headers: {
              Authorization: `Bearer ${data.token}`,
            },
          });

          if (!meResponse.ok) {
            throw new Error('Failed to get user info');
          }

          const userData = await meResponse.json();

          return {
            id: userData.id,
            email: userData.email,
            name: userData.name,
            role: userData.role as AdminRole,
            accessToken: data.token,
          };
        } catch (error) {
          console.error('Auth error:', error);
          throw error;
        }
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      // Initial sign-in
      if (user) {
        token.id = user.id;
        token.email = user.email ?? '';
        token.name = user.name ?? '';
        token.role = user.role;
        token.accessToken = user.accessToken;
      }
      return token;
    },
    async session({ session, token }) {
      // Extend session with our custom properties
      return {
        ...session,
        user: {
          ...session.user,
          id: token.id,
          email: token.email,
          name: token.name,
          role: token.role,
        },
        accessToken: token.accessToken,
      };
    },
    async authorized({ auth, request }) {
      const isLoggedIn = !!auth?.user;
      const isOnDashboard = request.nextUrl.pathname.startsWith('/dashboard');
      const isOnLogin = request.nextUrl.pathname === '/login';

      if (isOnDashboard) {
        if (isLoggedIn) return true;
        return false; // Redirect to login
      } else if (isOnLogin && isLoggedIn) {
        // Redirect logged-in users from login to dashboard
        return Response.redirect(new URL('/dashboard', request.nextUrl));
      }

      return true;
    },
  },
  pages: {
    signIn: '/login',
    error: '/login',
  },
  session: {
    strategy: 'jwt',
    maxAge: 8 * 60 * 60, // 8 hours
  },
  cookies: {
    sessionToken: {
      name: 'admin-session-token',
      options: {
        httpOnly: true,
        sameSite: 'strict',
        path: '/',
        secure: process.env.NODE_ENV === 'production',
      },
    },
  },
  trustHost: true,
};

export const {
  handlers: { GET, POST },
  auth,
  signIn,
  signOut,
} = NextAuth(authConfig);
