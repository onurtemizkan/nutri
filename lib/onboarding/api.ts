/**
 * Onboarding API Client
 *
 * API methods for interacting with the onboarding backend endpoints
 */

import api from '../api/client';
import {
  OnboardingStatus,
  OnboardingStep1Data,
  OnboardingStep2Data,
  OnboardingStep3Data,
  OnboardingStep4Data,
  OnboardingStep5Data,
  OnboardingStep6Data,
  OnboardingStepData,
} from './types';

/**
 * Full onboarding data response
 */
export interface OnboardingDataResponse {
  user: {
    id: string;
    name: string;
    email: string;
    dateOfBirth: string | null;
    biologicalSex: string | null;
    height: number | null;
    currentWeight: number | null;
    goalWeight: number | null;
    activityLevel: string;
    primaryGoal: string | null;
    dietaryPreferences: string[] | null;
    goalCalories: number;
    goalProtein: number;
    goalCarbs: number;
    goalFat: number;
  } | null;
  status: OnboardingStatus | null;
  permissions: {
    notificationsEnabled: boolean;
    notificationTypes: string[] | null;
    healthKitEnabled: boolean;
    healthKitScopes: string[] | null;
    healthConnectEnabled: boolean;
    healthConnectScopes: string[] | null;
    shareAnonymousData: boolean;
  } | null;
  healthBackground: {
    chronicConditions: unknown[] | null;
    medications: unknown[] | null;
    supplements: unknown[] | null;
    allergies: unknown[] | null;
  } | null;
  lifestyle: {
    nicotineUse: string | null;
    nicotineType: string | null;
    alcoholUse: string | null;
    caffeineDaily: number | null;
    typicalBedtime: string | null;
    typicalWakeTime: string | null;
    sleepQuality: number | null;
    stressLevel: number | null;
    workSchedule: string | null;
  } | null;
}

/**
 * Onboarding API methods
 */
export const onboardingApi = {
  /**
   * Start or resume onboarding
   */
  async start(version: string = '1.0'): Promise<OnboardingStatus> {
    const response = await api.post<OnboardingStatus>('/onboarding/start', { version });
    return response.data;
  },

  /**
   * Get current onboarding status
   */
  async getStatus(): Promise<OnboardingStatus | null> {
    try {
      const response = await api.get<OnboardingStatus>('/onboarding/status');
      return response.data;
    } catch (error: unknown) {
      // Handle 404 (onboarding not started) gracefully
      if (
        error !== null &&
        typeof error === 'object' &&
        'response' in error &&
        (error as { response?: { status?: number } }).response?.status === 404
      ) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get all onboarding data
   */
  async getData(): Promise<OnboardingDataResponse> {
    const response = await api.get<OnboardingDataResponse>('/onboarding/data');
    return response.data;
  },

  /**
   * Save step data (generic method)
   */
  async saveStep(stepNumber: number, data: OnboardingStepData): Promise<OnboardingStatus> {
    const response = await api.put<OnboardingStatus>(`/onboarding/step/${stepNumber}`, data);
    return response.data;
  },

  /**
   * Save Step 1: Profile Basics
   */
  async saveStep1(data: OnboardingStep1Data): Promise<OnboardingStatus> {
    return this.saveStep(1, data);
  },

  /**
   * Save Step 2: Health Goals
   */
  async saveStep2(data: OnboardingStep2Data): Promise<OnboardingStatus> {
    return this.saveStep(2, data);
  },

  /**
   * Save Step 3: Permissions
   */
  async saveStep3(data: OnboardingStep3Data): Promise<OnboardingStatus> {
    return this.saveStep(3, data);
  },

  /**
   * Save Step 4: Health Background
   */
  async saveStep4(data: OnboardingStep4Data): Promise<OnboardingStatus> {
    return this.saveStep(4, data);
  },

  /**
   * Save Step 5: Lifestyle
   */
  async saveStep5(data: OnboardingStep5Data): Promise<OnboardingStatus> {
    return this.saveStep(5, data);
  },

  /**
   * Save Step 6: Completion
   */
  async saveStep6(data: OnboardingStep6Data): Promise<OnboardingStatus> {
    return this.saveStep(6, data);
  },

  /**
   * Skip a step (only allowed for certain steps)
   */
  async skipStep(stepNumber: number): Promise<OnboardingStatus> {
    const response = await api.post<OnboardingStatus>(`/onboarding/skip/${stepNumber}`);
    return response.data;
  },

  /**
   * Complete onboarding
   */
  async complete(skipRemaining: boolean = false): Promise<OnboardingStatus> {
    const response = await api.post<OnboardingStatus>('/onboarding/complete', { skipRemaining });
    return response.data;
  },
};
