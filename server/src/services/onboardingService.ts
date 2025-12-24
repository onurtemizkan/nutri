import prisma from '../config/database';
import { Prisma, BiologicalSex, PrimaryGoal, NicotineUseLevel, AlcoholUseLevel } from '@prisma/client';
import {
  OnboardingStep1Data,
  OnboardingStep2Data,
  OnboardingStep3Data,
  OnboardingStep4Data,
  OnboardingStep5Data,
} from '../validation/schemas';

/**
 * Onboarding step configuration
 */
export const ONBOARDING_STEPS = {
  PROFILE_BASICS: 1,
  HEALTH_GOALS: 2,
  PERMISSIONS: 3,
  HEALTH_BACKGROUND: 4,
  LIFESTYLE: 5,
  COMPLETION: 6,
} as const;

const TOTAL_STEPS = 6;

/**
 * Steps that can be skipped
 */
const SKIPPABLE_STEPS: readonly number[] = [
  ONBOARDING_STEPS.HEALTH_BACKGROUND,
  ONBOARDING_STEPS.LIFESTYLE,
];

export interface OnboardingStatus {
  id: string;
  userId: string;
  currentStep: number;
  totalSteps: number;
  completedAt: Date | null;
  skippedSteps: number[];
  version: string;
  isComplete: boolean;
  progress: number;
}

export class OnboardingService {
  /**
   * Start or resume onboarding for a user
   */
  async startOnboarding(userId: string, version: string = '1.0'): Promise<OnboardingStatus> {
    // Check if onboarding already exists
    let onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    if (!onboarding) {
      // Create new onboarding record
      onboarding = await prisma.userOnboarding.create({
        data: {
          userId,
          version,
          totalSteps: TOTAL_STEPS,
        },
      });
    }

    return this.formatOnboardingStatus(onboarding);
  }

  /**
   * Get current onboarding status for a user
   */
  async getStatus(userId: string): Promise<OnboardingStatus | null> {
    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    if (!onboarding) {
      return null;
    }

    return this.formatOnboardingStatus(onboarding);
  }

  /**
   * Save data for Step 1: Profile Basics
   */
  async saveStep1(userId: string, data: OnboardingStep1Data): Promise<OnboardingStatus> {
    const dateOfBirth = new Date(data.dateOfBirth);

    await prisma.$transaction([
      // Update user profile
      prisma.user.update({
        where: { id: userId },
        data: {
          name: data.name,
          dateOfBirth,
          biologicalSex: data.biologicalSex as BiologicalSex,
          height: data.height,
          currentWeight: data.currentWeight,
          activityLevel: data.activityLevel,
        },
      }),
      // Update onboarding progress
      prisma.userOnboarding.update({
        where: { userId },
        data: {
          currentStep: ONBOARDING_STEPS.HEALTH_GOALS,
        },
      }),
    ]);

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    return this.formatOnboardingStatus(onboarding!);
  }

  /**
   * Save data for Step 2: Health Goals
   */
  async saveStep2(userId: string, data: OnboardingStep2Data): Promise<OnboardingStatus> {
    const updateData: Prisma.UserUpdateInput = {
      primaryGoal: data.primaryGoal as PrimaryGoal,
      dietaryPreferences: data.dietaryPreferences ?? [],
    };

    if (data.goalWeight !== undefined) {
      updateData.goalWeight = data.goalWeight;
    }

    // Handle custom macros
    if (data.customMacros) {
      if (data.customMacros.goalCalories !== undefined) {
        updateData.goalCalories = data.customMacros.goalCalories;
      }
      if (data.customMacros.goalProtein !== undefined) {
        updateData.goalProtein = data.customMacros.goalProtein;
      }
      if (data.customMacros.goalCarbs !== undefined) {
        updateData.goalCarbs = data.customMacros.goalCarbs;
      }
      if (data.customMacros.goalFat !== undefined) {
        updateData.goalFat = data.customMacros.goalFat;
      }
    }

    await prisma.$transaction([
      prisma.user.update({
        where: { id: userId },
        data: updateData,
      }),
      prisma.userOnboarding.update({
        where: { userId },
        data: {
          currentStep: ONBOARDING_STEPS.PERMISSIONS,
        },
      }),
    ]);

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    return this.formatOnboardingStatus(onboarding!);
  }

  /**
   * Save data for Step 3: Permissions
   */
  async saveStep3(userId: string, data: OnboardingStep3Data): Promise<OnboardingStatus> {
    await prisma.$transaction([
      // Upsert user permissions
      prisma.userPermissions.upsert({
        where: { userId },
        create: {
          userId,
          notificationsEnabled: data.notificationsEnabled,
          notificationTypes: data.notificationTypes,
          healthKitEnabled: data.healthKitEnabled,
          healthKitScopes: data.healthKitScopes,
          healthConnectEnabled: data.healthConnectEnabled ?? false,
          healthConnectScopes: data.healthConnectScopes ?? [],
        },
        update: {
          notificationsEnabled: data.notificationsEnabled,
          notificationTypes: data.notificationTypes,
          healthKitEnabled: data.healthKitEnabled,
          healthKitScopes: data.healthKitScopes,
          healthConnectEnabled: data.healthConnectEnabled ?? false,
          healthConnectScopes: data.healthConnectScopes ?? [],
        },
      }),
      // Update onboarding progress
      prisma.userOnboarding.update({
        where: { userId },
        data: {
          currentStep: ONBOARDING_STEPS.HEALTH_BACKGROUND,
        },
      }),
    ]);

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    return this.formatOnboardingStatus(onboarding!);
  }

  /**
   * Save data for Step 4: Health Background
   */
  async saveStep4(userId: string, data: OnboardingStep4Data): Promise<OnboardingStatus> {
    await prisma.$transaction([
      // Upsert health background
      prisma.userHealthBackground.upsert({
        where: { userId },
        create: {
          userId,
          chronicConditions: data.chronicConditions ?? [],
          medications: data.medications ?? [],
          supplements: data.supplements ?? [],
          allergies: [...(data.allergies ?? []), ...(data.allergyNotes ? [data.allergyNotes] : [])],
        },
        update: {
          chronicConditions: data.chronicConditions ?? [],
          medications: data.medications ?? [],
          supplements: data.supplements ?? [],
          allergies: [...(data.allergies ?? []), ...(data.allergyNotes ? [data.allergyNotes] : [])],
        },
      }),
      // Update onboarding progress
      prisma.userOnboarding.update({
        where: { userId },
        data: {
          currentStep: ONBOARDING_STEPS.LIFESTYLE,
        },
      }),
    ]);

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    return this.formatOnboardingStatus(onboarding!);
  }

  /**
   * Save data for Step 5: Lifestyle
   */
  async saveStep5(userId: string, data: OnboardingStep5Data): Promise<OnboardingStatus> {
    await prisma.$transaction([
      // Upsert lifestyle data
      prisma.userLifestyle.upsert({
        where: { userId },
        create: {
          userId,
          nicotineUse: data.nicotineUse as NicotineUseLevel | undefined,
          nicotineType: data.nicotineType,
          alcoholUse: data.alcoholUse as AlcoholUseLevel | undefined,
          caffeineDaily: data.caffeineDaily,
          typicalBedtime: data.typicalBedtime,
          typicalWakeTime: data.typicalWakeTime,
          sleepQuality: data.sleepQuality,
          stressLevel: data.stressLevel,
          workSchedule: data.workSchedule,
        },
        update: {
          nicotineUse: data.nicotineUse as NicotineUseLevel | undefined,
          nicotineType: data.nicotineType,
          alcoholUse: data.alcoholUse as AlcoholUseLevel | undefined,
          caffeineDaily: data.caffeineDaily,
          typicalBedtime: data.typicalBedtime,
          typicalWakeTime: data.typicalWakeTime,
          sleepQuality: data.sleepQuality,
          stressLevel: data.stressLevel,
          workSchedule: data.workSchedule,
        },
      }),
      // Update onboarding progress
      prisma.userOnboarding.update({
        where: { userId },
        data: {
          currentStep: ONBOARDING_STEPS.COMPLETION,
        },
      }),
    ]);

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    return this.formatOnboardingStatus(onboarding!);
  }

  /**
   * Skip a step (only allowed for certain steps)
   */
  async skipStep(userId: string, stepNumber: number): Promise<OnboardingStatus> {
    if (!SKIPPABLE_STEPS.includes(stepNumber)) {
      throw new Error(`Step ${stepNumber} cannot be skipped`);
    }

    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    if (!onboarding) {
      throw new Error('Onboarding not found');
    }

    if (onboarding.currentStep !== stepNumber) {
      throw new Error(`Cannot skip step ${stepNumber} - current step is ${onboarding.currentStep}`);
    }

    const updatedSkippedSteps = [...(onboarding.skippedSteps || []), stepNumber];

    const updated = await prisma.userOnboarding.update({
      where: { userId },
      data: {
        currentStep: stepNumber + 1,
        skippedSteps: updatedSkippedSteps,
      },
    });

    return this.formatOnboardingStatus(updated);
  }

  /**
   * Complete onboarding
   */
  async completeOnboarding(userId: string, skipRemaining: boolean = false): Promise<OnboardingStatus> {
    const onboarding = await prisma.userOnboarding.findUnique({
      where: { userId },
    });

    if (!onboarding) {
      throw new Error('Onboarding not found');
    }

    // If skipping remaining steps, mark them as skipped
    let skippedSteps = [...(onboarding.skippedSteps || [])];
    if (skipRemaining) {
      for (let step = onboarding.currentStep; step < TOTAL_STEPS; step++) {
        if (SKIPPABLE_STEPS.includes(step) && !skippedSteps.includes(step)) {
          skippedSteps.push(step);
        }
      }
    }

    const updated = await prisma.userOnboarding.update({
      where: { userId },
      data: {
        currentStep: TOTAL_STEPS,
        completedAt: new Date(),
        skippedSteps,
      },
    });

    return this.formatOnboardingStatus(updated);
  }

  /**
   * Get full onboarding data for a user
   */
  async getOnboardingData(userId: string) {
    const [user, onboarding, permissions, healthBackground, lifestyle] = await Promise.all([
      prisma.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          name: true,
          email: true,
          dateOfBirth: true,
          biologicalSex: true,
          height: true,
          currentWeight: true,
          goalWeight: true,
          activityLevel: true,
          primaryGoal: true,
          dietaryPreferences: true,
          goalCalories: true,
          goalProtein: true,
          goalCarbs: true,
          goalFat: true,
        },
      }),
      prisma.userOnboarding.findUnique({
        where: { userId },
      }),
      prisma.userPermissions.findUnique({
        where: { userId },
      }),
      prisma.userHealthBackground.findUnique({
        where: { userId },
      }),
      prisma.userLifestyle.findUnique({
        where: { userId },
      }),
    ]);

    return {
      user,
      status: onboarding ? this.formatOnboardingStatus(onboarding) : null,
      permissions,
      healthBackground,
      lifestyle,
    };
  }

  /**
   * Format onboarding record into status response
   */
  private formatOnboardingStatus(onboarding: Prisma.UserOnboardingGetPayload<object>): OnboardingStatus {
    const isComplete = onboarding.completedAt !== null;
    const progress = isComplete ? 100 : Math.round(((onboarding.currentStep - 1) / TOTAL_STEPS) * 100);

    return {
      id: onboarding.id,
      userId: onboarding.userId,
      currentStep: onboarding.currentStep,
      totalSteps: onboarding.totalSteps,
      completedAt: onboarding.completedAt,
      skippedSteps: onboarding.skippedSteps,
      version: onboarding.version,
      isComplete,
      progress,
    };
  }
}

export const onboardingService = new OnboardingService();
