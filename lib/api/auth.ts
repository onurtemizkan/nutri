import api from './client';
import * as SecureStore from 'expo-secure-store';
import { AuthResponse, User } from '../types';

export const authApi = {
  async register(email: string, password: string, name: string): Promise<AuthResponse> {
    const response = await api.post<AuthResponse>('/auth/register', {
      email,
      password,
      name,
    });

    if (response.data.token) {
      await SecureStore.setItemAsync('authToken', response.data.token);
    }

    return response.data;
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await api.post<AuthResponse>('/auth/login', {
      email,
      password,
    });

    if (response.data.token) {
      await SecureStore.setItemAsync('authToken', response.data.token);
    }

    return response.data;
  },

  async logout(): Promise<void> {
    await SecureStore.deleteItemAsync('authToken');
  },

  async getProfile(): Promise<User> {
    const response = await api.get<User>('/auth/profile');
    return response.data;
  },

  async updateProfile(data: Partial<User>): Promise<User> {
    const response = await api.put<User>('/auth/profile', data);
    return response.data;
  },

  async getToken(): Promise<string | null> {
    return await SecureStore.getItemAsync('authToken');
  },

  async forgotPassword(email: string): Promise<{ message: string; resetToken?: string }> {
    const response = await api.post('/auth/forgot-password', { email });
    return response.data;
  },

  async resetPassword(token: string, newPassword: string): Promise<{ message: string }> {
    const response = await api.post('/auth/reset-password', {
      token,
      newPassword,
    });
    return response.data;
  },

  async verifyResetToken(token: string): Promise<{ valid: boolean; email: string }> {
    const response = await api.post('/auth/verify-reset-token', { token });
    return response.data;
  },

  async appleSignIn(data: {
    identityToken: string;
    authorizationCode: string;
    user?: {
      email?: string;
      name?: {
        firstName?: string;
        lastName?: string;
      };
    };
  }): Promise<AuthResponse> {
    const response = await api.post<AuthResponse>('/auth/apple-signin', data);

    if (response.data.token) {
      await SecureStore.setItemAsync('authToken', response.data.token);
    }

    return response.data;
  },

  async deleteAccount(): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>('/auth/account');
    // Clear the token after successful deletion
    await SecureStore.deleteItemAsync('authToken');
    return response.data;
  },
};
