/**
 * Onboarding Types
 * Type definitions for the user onboarding flow
 */

// ============================================================================
// ENUMS
// ============================================================================

export type BiologicalSex = 'MALE' | 'FEMALE' | 'OTHER' | 'PREFER_NOT_TO_SAY';

export type PrimaryGoal =
  | 'WEIGHT_LOSS'
  | 'MUSCLE_GAIN'
  | 'MAINTENANCE'
  | 'GENERAL_HEALTH'
  | 'ATHLETIC_PERFORMANCE'
  | 'BETTER_SLEEP'
  | 'STRESS_REDUCTION';

export type ActivityLevel = 'sedentary' | 'light' | 'moderate' | 'active' | 'veryActive';

export type NicotineUseLevel = 'NONE' | 'OCCASIONAL' | 'DAILY' | 'HEAVY';

export type AlcoholUseLevel = 'NONE' | 'OCCASIONAL' | 'MODERATE' | 'FREQUENT';

export type DietaryPreference =
  | 'vegetarian'
  | 'vegan'
  | 'pescatarian'
  | 'keto'
  | 'paleo'
  | 'gluten_free'
  | 'dairy_free'
  | 'nut_free'
  | 'low_carb'
  | 'low_fat'
  | 'mediterranean'
  | 'halal'
  | 'kosher'
  | 'none';

// ============================================================================
// STEP DATA TYPES
// ============================================================================

/**
 * Step 1: Profile Basics
 */
export interface OnboardingStep1Data {
  name: string;
  dateOfBirth: string; // YYYY-MM-DD
  biologicalSex: BiologicalSex;
  height: number; // cm
  currentWeight: number; // kg
  activityLevel: ActivityLevel;
}

/**
 * Step 2: Health Goals
 */
export interface OnboardingStep2Data {
  primaryGoal: PrimaryGoal;
  goalWeight?: number; // kg
  dietaryPreferences: DietaryPreference[];
  customMacros?: {
    goalCalories?: number;
    goalProtein?: number;
    goalCarbs?: number;
    goalFat?: number;
  };
}

/**
 * Step 3: Permissions
 */
export interface OnboardingStep3Data {
  notificationsEnabled: boolean;
  notificationTypes: NotificationType[];
  healthKitEnabled: boolean;
  healthKitScopes: HealthKitScope[];
  healthConnectEnabled: boolean;
  healthConnectScopes: string[];
}

export type NotificationType = 'meal_reminders' | 'insights' | 'goals' | 'weekly_summary';

export type HealthKitScope =
  | 'heartRate'
  | 'restingHeartRate'
  | 'hrv'
  | 'steps'
  | 'activeEnergy'
  | 'sleep'
  | 'weight'
  | 'bodyFat'
  | 'workouts'
  | 'vo2Max'
  | 'respiratoryRate';

/**
 * Step 4: Health Background
 */
export interface OnboardingStep4Data {
  chronicConditions: ChronicCondition[];
  medications: Medication[];
  supplements: Supplement[];
  allergies: Allergy[];
  allergyNotes?: string;
}

export type ChronicConditionType =
  | 'diabetes_type1'
  | 'diabetes_type2'
  | 'prediabetic'
  | 'hypertension'
  | 'heart_disease'
  | 'thyroid_hypothyroid'
  | 'thyroid_hyperthyroid'
  | 'pcos'
  | 'ibs'
  | 'celiac'
  | 'crohns'
  | 'ulcerative_colitis'
  | 'asthma'
  | 'arthritis'
  | 'osteoporosis'
  | 'depression'
  | 'anxiety'
  | 'eating_disorder'
  | 'other';

export interface ChronicCondition {
  type: ChronicConditionType;
  customType?: string;
  diagnosedYear?: number;
  notes?: string;
}

export type MedicationFrequency = 'as_needed' | 'daily' | 'twice_daily' | 'weekly' | 'monthly';

export type MedicationCategory =
  | 'blood_pressure'
  | 'diabetes'
  | 'thyroid'
  | 'mental_health'
  | 'pain'
  | 'heart'
  | 'cholesterol'
  | 'hormones'
  | 'digestive'
  | 'other';

export interface Medication {
  name: string;
  dosage?: string;
  frequency?: MedicationFrequency;
  category?: MedicationCategory;
}

export type SupplementName =
  | 'vitamin_a'
  | 'vitamin_b_complex'
  | 'vitamin_b12'
  | 'vitamin_c'
  | 'vitamin_d'
  | 'vitamin_e'
  | 'vitamin_k'
  | 'iron'
  | 'magnesium'
  | 'zinc'
  | 'calcium'
  | 'potassium'
  | 'omega3_fish_oil'
  | 'probiotics'
  | 'creatine'
  | 'protein_powder'
  | 'collagen'
  | 'melatonin'
  | 'ashwagandha'
  | 'turmeric_curcumin'
  | 'coq10'
  | 'fiber'
  | 'multivitamin'
  | 'other';

export type SupplementFrequency = 'daily' | 'twice_daily' | 'weekly' | 'as_needed';

export interface Supplement {
  name: SupplementName;
  customName?: string;
  dosage?: string;
  frequency?: SupplementFrequency;
}

export type Allergy =
  | 'peanuts'
  | 'tree_nuts'
  | 'milk'
  | 'eggs'
  | 'wheat'
  | 'soy'
  | 'fish'
  | 'shellfish'
  | 'sesame'
  | 'sulfites'
  | 'other';

/**
 * Step 5: Lifestyle
 */
export interface OnboardingStep5Data {
  nicotineUse?: NicotineUseLevel;
  nicotineType?: NicotineType;
  alcoholUse?: AlcoholUseLevel;
  caffeineDaily?: number;
  typicalBedtime?: string; // HH:MM
  typicalWakeTime?: string; // HH:MM
  sleepQuality?: number; // 1-10
  stressLevel?: number; // 1-10
  workSchedule?: WorkSchedule;
}

export type NicotineType = 'cigarettes' | 'vape' | 'chewing' | 'patches' | 'other';

export type WorkSchedule = 'regular' | 'shift' | 'irregular' | 'remote' | 'not_working';

/**
 * Step 6: Completion
 */
export interface OnboardingStep6Data {
  acknowledged: true;
}

// ============================================================================
// ONBOARDING STATUS
// ============================================================================

export interface OnboardingStatus {
  id: string;
  userId: string;
  currentStep: number;
  totalSteps: number;
  completedAt: string | null;
  skippedSteps: number[];
  version: string;
  isComplete: boolean;
  progress: number;
}

// ============================================================================
// STEP CONFIGURATION
// ============================================================================

export type OnboardingStepData =
  | OnboardingStep1Data
  | OnboardingStep2Data
  | OnboardingStep3Data
  | OnboardingStep4Data
  | OnboardingStep5Data
  | OnboardingStep6Data;

export interface OnboardingStepConfig {
  id: number;
  key: string;
  title: string;
  description: string;
  required: boolean;
  skippable: boolean;
  icon: string;
}

export interface OnboardingData {
  step1?: OnboardingStep1Data;
  step2?: OnboardingStep2Data;
  step3?: OnboardingStep3Data;
  step4?: OnboardingStep4Data;
  step5?: OnboardingStep5Data;
  step6?: OnboardingStep6Data;
}
