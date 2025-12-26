import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { User } from '../types';
import { authApi } from '../api/auth';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  appleSignIn: (data: {
    identityToken: string;
    authorizationCode: string;
    user?: {
      email?: string;
      name?: {
        firstName?: string;
        lastName?: string;
      };
    };
  }) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (data: Partial<User>) => Promise<void>;
  deleteAccount: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Refs to prevent race conditions
  const isMountedRef = useRef(true);
  const isAuthOperationInProgressRef = useRef(false);

  const checkAuthStatus = useCallback(async () => {
    // Skip if another operation is in progress
    if (isAuthOperationInProgressRef.current) {
      return;
    }

    isAuthOperationInProgressRef.current = true;

    try {
      const token = await authApi.getToken();
      if (token && isMountedRef.current) {
        const userProfile = await authApi.getProfile();
        // Only update state if still mounted
        if (isMountedRef.current) {
          setUser(userProfile);
        }
      }
    } catch {
      // Auth check failed - clear token but don't throw
      // This is expected when token is expired/invalid
      if (isMountedRef.current) {
        await authApi.logout();
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoading(false);
      }
      isAuthOperationInProgressRef.current = false;
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    checkAuthStatus();

    // Cleanup: mark as unmounted to prevent state updates
    return () => {
      isMountedRef.current = false;
    };
  }, [checkAuthStatus]);

  const login = async (email: string, password: string) => {
    const response = await authApi.login(email, password);
    if (isMountedRef.current) {
      setUser(response.user);
    }
  };

  const register = async (email: string, password: string, name: string) => {
    const response = await authApi.register(email, password, name);
    if (isMountedRef.current) {
      setUser(response.user);
    }
  };

  const appleSignIn = async (data: {
    identityToken: string;
    authorizationCode: string;
    user?: {
      email?: string;
      name?: {
        firstName?: string;
        lastName?: string;
      };
    };
  }) => {
    const response = await authApi.appleSignIn(data);
    if (isMountedRef.current) {
      setUser(response.user);
    }
  };

  const logout = async () => {
    try {
      await authApi.logout();
      if (isMountedRef.current) {
        setUser(null);
      }
    } catch {
      // Logout failed - still clear local user state
      // Token will be cleared anyway, user can log in again
      if (isMountedRef.current) {
        setUser(null);
      }
    }
  };

  const updateUser = async (data: Partial<User>) => {
    const updatedUser = await authApi.updateProfile(data);
    if (isMountedRef.current) {
      setUser(updatedUser);
    }
  };

  const deleteAccount = async () => {
    await authApi.deleteAccount();
    if (isMountedRef.current) {
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        appleSignIn,
        logout,
        updateUser,
        deleteAccount,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
