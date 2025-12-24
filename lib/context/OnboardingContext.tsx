import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  OnboardingStatus,
  OnboardingData,
  OnboardingStepData,
} from '../onboarding/types';
import { onboardingApi, OnboardingDataResponse } from '../onboarding/api';
import {
  TOTAL_ONBOARDING_STEPS,
  canSkipStep,
  getNextStepId,
  getPreviousStepId,
  DEFAULT_STEP1_DATA,
  DEFAULT_STEP2_DATA,
  DEFAULT_STEP3_DATA,
  DEFAULT_STEP4_DATA,
  DEFAULT_STEP5_DATA,
} from '../onboarding/config';

// ============================================================================
// STORAGE KEYS
// ============================================================================

const STORAGE_KEYS = {
  ONBOARDING_STATUS: '@nutri/onboarding_status',
  ONBOARDING_DATA: '@nutri/onboarding_data',
  ONBOARDING_DRAFT: '@nutri/onboarding_draft',
} as const;

// ============================================================================
// CONTEXT TYPES
// ============================================================================

interface OnboardingContextType {
  // Status
  status: OnboardingStatus | null;
  isLoading: boolean;
  isComplete: boolean;
  currentStep: number;
  progress: number;

  // Saved data from server
  savedData: OnboardingDataResponse | null;

  // Draft data (local, unsaved)
  draftData: OnboardingData;

  // Actions
  startOnboarding: () => Promise<void>;
  refreshStatus: () => Promise<void>;
  saveStep: (stepNumber: number, data: OnboardingStepData) => Promise<void>;
  skipStep: (stepNumber: number) => Promise<void>;
  completeOnboarding: (skipRemaining?: boolean) => Promise<void>;
  goToNextStep: () => void;
  goToPreviousStep: () => void;
  goToStep: (stepNumber: number) => void;

  // Draft management
  updateDraft: (stepNumber: number, data: Partial<OnboardingStepData>) => void;
  clearDraft: () => void;
  getDraftForStep: <T extends OnboardingStepData>(stepNumber: number) => T | undefined;

  // Utility
  canGoBack: boolean;
  canGoForward: boolean;
  isStepSkippable: (stepNumber: number) => boolean;
  hasCompletedStep: (stepNumber: number) => boolean;
}

const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined);

// ============================================================================
// PROVIDER
// ============================================================================

export function OnboardingProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<OnboardingStatus | null>(null);
  const [savedData, setSavedData] = useState<OnboardingDataResponse | null>(null);
  const [draftData, setDraftData] = useState<OnboardingData>({});
  const [isLoading, setIsLoading] = useState(true);
  const [localStep, setLocalStep] = useState(1);

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  useEffect(() => {
    initializeOnboarding();
  }, []);

  const initializeOnboarding = async () => {
    try {
      setIsLoading(true);

      // Load cached status from AsyncStorage
      const cachedStatus = await loadFromStorage<OnboardingStatus>(STORAGE_KEYS.ONBOARDING_STATUS);
      if (cachedStatus) {
        setStatus(cachedStatus);
        setLocalStep(cachedStatus.currentStep);
      }

      // Load draft data from AsyncStorage
      const cachedDraft = await loadFromStorage<OnboardingData>(STORAGE_KEYS.ONBOARDING_DRAFT);
      if (cachedDraft) {
        setDraftData(cachedDraft);
      }

      // Try to fetch fresh status from server
      try {
        const serverStatus = await onboardingApi.getStatus();
        if (serverStatus) {
          setStatus(serverStatus);
          setLocalStep(serverStatus.currentStep);
          await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, serverStatus);

          // Also fetch saved data if onboarding exists
          const data = await onboardingApi.getData();
          setSavedData(data);
          await saveToStorage(STORAGE_KEYS.ONBOARDING_DATA, data);
        }
      } catch {
        // Server fetch failed, continue with cached data
        console.log('Could not fetch onboarding status from server');
      }
    } catch (error) {
      console.error('Failed to initialize onboarding:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // ============================================================================
  // ACTIONS
  // ============================================================================

  const startOnboarding = useCallback(async () => {
    try {
      setIsLoading(true);
      const newStatus = await onboardingApi.start();
      setStatus(newStatus);
      setLocalStep(newStatus.currentStep);
      await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, newStatus);

      // Initialize draft with defaults
      setDraftData({
        step1: DEFAULT_STEP1_DATA,
        step2: DEFAULT_STEP2_DATA,
        step3: DEFAULT_STEP3_DATA,
        step4: DEFAULT_STEP4_DATA,
        step5: DEFAULT_STEP5_DATA,
      });
    } catch (error) {
      console.error('Failed to start onboarding:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const serverStatus = await onboardingApi.getStatus();
      if (serverStatus) {
        setStatus(serverStatus);
        setLocalStep(serverStatus.currentStep);
        await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, serverStatus);

        const data = await onboardingApi.getData();
        setSavedData(data);
        await saveToStorage(STORAGE_KEYS.ONBOARDING_DATA, data);
      }
    } catch (error) {
      console.error('Failed to refresh onboarding status:', error);
    }
  }, []);

  const saveStep = useCallback(
    async (stepNumber: number, data: OnboardingStepData) => {
      try {
        setIsLoading(true);

        // Save to server
        const newStatus = await onboardingApi.saveStep(stepNumber, data);
        setStatus(newStatus);
        setLocalStep(newStatus.currentStep);
        await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, newStatus);

        // Clear draft for this step since it's now saved
        setDraftData((prev) => {
          const updated = { ...prev };
          delete updated[`step${stepNumber}` as keyof OnboardingData];
          return updated;
        });
        await saveDraftToStorage(draftData);

        // Refresh saved data
        const data_ = await onboardingApi.getData();
        setSavedData(data_);
        await saveToStorage(STORAGE_KEYS.ONBOARDING_DATA, data_);
      } catch (error) {
        console.error(`Failed to save step ${stepNumber}:`, error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [draftData]
  );

  const skipStep = useCallback(async (stepNumber: number) => {
    try {
      setIsLoading(true);

      if (!canSkipStep(stepNumber)) {
        throw new Error(`Step ${stepNumber} cannot be skipped`);
      }

      const newStatus = await onboardingApi.skipStep(stepNumber);
      setStatus(newStatus);
      setLocalStep(newStatus.currentStep);
      await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, newStatus);
    } catch (error) {
      console.error(`Failed to skip step ${stepNumber}:`, error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const completeOnboarding = useCallback(async (skipRemaining: boolean = false) => {
    try {
      setIsLoading(true);

      const newStatus = await onboardingApi.complete(skipRemaining);
      setStatus(newStatus);
      await saveToStorage(STORAGE_KEYS.ONBOARDING_STATUS, newStatus);

      // Clear all draft data
      setDraftData({});
      await AsyncStorage.removeItem(STORAGE_KEYS.ONBOARDING_DRAFT);
    } catch (error) {
      console.error('Failed to complete onboarding:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // ============================================================================
  // NAVIGATION
  // ============================================================================

  const goToNextStep = useCallback(() => {
    const nextStep = getNextStepId(localStep);
    if (nextStep !== null) {
      setLocalStep(nextStep);
    }
  }, [localStep]);

  const goToPreviousStep = useCallback(() => {
    const prevStep = getPreviousStepId(localStep);
    if (prevStep !== null) {
      setLocalStep(prevStep);
    }
  }, [localStep]);

  const goToStep = useCallback((stepNumber: number) => {
    if (stepNumber >= 1 && stepNumber <= TOTAL_ONBOARDING_STEPS) {
      setLocalStep(stepNumber);
    }
  }, []);

  // ============================================================================
  // DRAFT MANAGEMENT
  // ============================================================================

  const updateDraft = useCallback(
    (stepNumber: number, data: Partial<OnboardingStepData>) => {
      setDraftData((prev) => {
        const stepKey = `step${stepNumber}` as keyof OnboardingData;
        const existing = prev[stepKey] || {};
        const updated = {
          ...prev,
          [stepKey]: { ...existing, ...data },
        };
        saveDraftToStorage(updated);
        return updated;
      });
    },
    []
  );

  const clearDraft = useCallback(() => {
    setDraftData({});
    AsyncStorage.removeItem(STORAGE_KEYS.ONBOARDING_DRAFT);
  }, []);

  const getDraftForStep = useCallback(
    <T extends OnboardingStepData>(stepNumber: number): T | undefined => {
      const stepKey = `step${stepNumber}` as keyof OnboardingData;
      return draftData[stepKey] as T | undefined;
    },
    [draftData]
  );

  // ============================================================================
  // COMPUTED VALUES
  // ============================================================================

  const isComplete = status?.isComplete ?? false;
  const currentStep = localStep;
  const progress = status?.progress ?? 0;

  const canGoBack = localStep > 1;
  const canGoForward = localStep < TOTAL_ONBOARDING_STEPS;

  const isStepSkippable = useCallback((stepNumber: number) => {
    return canSkipStep(stepNumber);
  }, []);

  const hasCompletedStep = useCallback(
    (stepNumber: number) => {
      if (!status) return false;
      return stepNumber < status.currentStep;
    },
    [status]
  );

  // ============================================================================
  // CONTEXT VALUE
  // ============================================================================

  const contextValue: OnboardingContextType = {
    // Status
    status,
    isLoading,
    isComplete,
    currentStep,
    progress,

    // Data
    savedData,
    draftData,

    // Actions
    startOnboarding,
    refreshStatus,
    saveStep,
    skipStep,
    completeOnboarding,
    goToNextStep,
    goToPreviousStep,
    goToStep,

    // Draft management
    updateDraft,
    clearDraft,
    getDraftForStep,

    // Utility
    canGoBack,
    canGoForward,
    isStepSkippable,
    hasCompletedStep,
  };

  return <OnboardingContext.Provider value={contextValue}>{children}</OnboardingContext.Provider>;
}

// ============================================================================
// HOOK
// ============================================================================

export function useOnboarding() {
  const context = useContext(OnboardingContext);
  if (context === undefined) {
    throw new Error('useOnboarding must be used within an OnboardingProvider');
  }
  return context;
}

// ============================================================================
// STORAGE HELPERS
// ============================================================================

async function loadFromStorage<T>(key: string): Promise<T | null> {
  try {
    const value = await AsyncStorage.getItem(key);
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

async function saveToStorage<T>(key: string, value: T): Promise<void> {
  try {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Failed to save to storage (${key}):`, error);
  }
}

async function saveDraftToStorage(draft: OnboardingData): Promise<void> {
  await saveToStorage(STORAGE_KEYS.ONBOARDING_DRAFT, draft);
}

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type { OnboardingContextType };
