/**
 * Onboarding Configuration
 *
 * Configuration-driven onboarding flow that supports:
 * - Easy reordering of steps
 * - Adding/removing steps
 * - Defining required vs optional steps
 * - Skip conditions
 * - Validation rules
 */

import {
  OnboardingStepConfig,
  BiologicalSex,
  PrimaryGoal,
  ActivityLevel,
  DietaryPreference,
  NotificationType,
  HealthKitScope,
  ChronicConditionType,
  MedicationCategory,
  SupplementName,
  Allergy,
  NicotineUseLevel,
  AlcoholUseLevel,
  WorkSchedule,
} from './types';

// ============================================================================
// STEP CONFIGURATION
// ============================================================================

/**
 * Onboarding Steps Configuration
 * Each step is defined with its properties and can be easily reordered
 */
export const ONBOARDING_STEPS: OnboardingStepConfig[] = [
  {
    id: 1,
    key: 'profile',
    title: 'Your Profile',
    description: 'Tell us about yourself so we can personalize your experience',
    required: true,
    skippable: false,
    icon: 'person-outline',
  },
  {
    id: 2,
    key: 'goals',
    title: 'Your Goals',
    description: 'What do you want to achieve with Nutri?',
    required: true,
    skippable: false,
    icon: 'target',
  },
  {
    id: 3,
    key: 'permissions',
    title: 'Permissions',
    description: 'Enable features to get the most out of Nutri',
    required: true,
    skippable: false,
    icon: 'shield-checkmark-outline',
  },
  {
    id: 4,
    key: 'health-background',
    title: 'Health Background',
    description: 'Help us understand your health history for better insights',
    required: false,
    skippable: true,
    icon: 'medical-outline',
  },
  {
    id: 5,
    key: 'lifestyle',
    title: 'Lifestyle',
    description: 'Tell us about your daily habits',
    required: false,
    skippable: true,
    icon: 'cafe-outline',
  },
  {
    id: 6,
    key: 'completion',
    title: "You're All Set!",
    description: "Let's start your nutrition journey",
    required: true,
    skippable: false,
    icon: 'checkmark-circle-outline',
  },
];

export const TOTAL_ONBOARDING_STEPS = ONBOARDING_STEPS.length;

export const SKIPPABLE_STEP_IDS = ONBOARDING_STEPS.filter((step) => step.skippable).map(
  (step) => step.id
);

// ============================================================================
// OPTION LISTS FOR FORMS
// ============================================================================

/**
 * Biological Sex Options
 */
export const BIOLOGICAL_SEX_OPTIONS: { value: BiologicalSex; label: string }[] = [
  { value: 'MALE', label: 'Male' },
  { value: 'FEMALE', label: 'Female' },
  { value: 'OTHER', label: 'Other' },
  { value: 'PREFER_NOT_TO_SAY', label: 'Prefer not to say' },
];

/**
 * Activity Level Options
 */
export const ACTIVITY_LEVEL_OPTIONS: { value: ActivityLevel; label: string; description: string }[] =
  [
    { value: 'sedentary', label: 'Sedentary', description: 'Little to no exercise' },
    { value: 'light', label: 'Lightly Active', description: 'Light exercise 1-3 days/week' },
    { value: 'moderate', label: 'Moderately Active', description: 'Moderate exercise 3-5 days/week' },
    { value: 'active', label: 'Very Active', description: 'Hard exercise 6-7 days/week' },
    { value: 'veryActive', label: 'Extra Active', description: 'Very hard exercise & physical job' },
  ];

/**
 * Primary Goal Options
 */
export const PRIMARY_GOAL_OPTIONS: { value: PrimaryGoal; label: string; icon: string }[] = [
  { value: 'WEIGHT_LOSS', label: 'Lose Weight', icon: 'trending-down-outline' },
  { value: 'MUSCLE_GAIN', label: 'Build Muscle', icon: 'barbell-outline' },
  { value: 'MAINTENANCE', label: 'Maintain Weight', icon: 'scale-outline' },
  { value: 'GENERAL_HEALTH', label: 'General Health', icon: 'heart-outline' },
  { value: 'ATHLETIC_PERFORMANCE', label: 'Athletic Performance', icon: 'trophy-outline' },
  { value: 'BETTER_SLEEP', label: 'Better Sleep', icon: 'moon-outline' },
  { value: 'STRESS_REDUCTION', label: 'Reduce Stress', icon: 'leaf-outline' },
];

/**
 * Dietary Preference Options
 */
export const DIETARY_PREFERENCE_OPTIONS: { value: DietaryPreference; label: string }[] = [
  { value: 'none', label: 'No restrictions' },
  { value: 'vegetarian', label: 'Vegetarian' },
  { value: 'vegan', label: 'Vegan' },
  { value: 'pescatarian', label: 'Pescatarian' },
  { value: 'keto', label: 'Keto' },
  { value: 'paleo', label: 'Paleo' },
  { value: 'gluten_free', label: 'Gluten-free' },
  { value: 'dairy_free', label: 'Dairy-free' },
  { value: 'nut_free', label: 'Nut-free' },
  { value: 'low_carb', label: 'Low carb' },
  { value: 'low_fat', label: 'Low fat' },
  { value: 'mediterranean', label: 'Mediterranean' },
  { value: 'halal', label: 'Halal' },
  { value: 'kosher', label: 'Kosher' },
];

/**
 * Notification Type Options
 */
export const NOTIFICATION_TYPE_OPTIONS: {
  value: NotificationType;
  label: string;
  description: string;
}[] = [
  {
    value: 'meal_reminders',
    label: 'Meal Reminders',
    description: 'Get reminded to log your meals',
  },
  { value: 'insights', label: 'Health Insights', description: 'Receive personalized health insights' },
  { value: 'goals', label: 'Goal Updates', description: 'Track your progress towards goals' },
  {
    value: 'weekly_summary',
    label: 'Weekly Summary',
    description: 'Get a weekly nutrition summary',
  },
];

/**
 * HealthKit Scope Options
 */
export const HEALTHKIT_SCOPE_OPTIONS: {
  value: HealthKitScope;
  label: string;
  description: string;
}[] = [
  { value: 'heartRate', label: 'Heart Rate', description: 'Track your heart rate patterns' },
  {
    value: 'restingHeartRate',
    label: 'Resting Heart Rate',
    description: 'Monitor your resting heart rate',
  },
  {
    value: 'hrv',
    label: 'Heart Rate Variability',
    description: 'Track stress and recovery',
  },
  { value: 'steps', label: 'Steps', description: 'Count your daily steps' },
  {
    value: 'activeEnergy',
    label: 'Active Calories',
    description: 'Track calories burned',
  },
  { value: 'sleep', label: 'Sleep', description: 'Analyze your sleep patterns' },
  { value: 'weight', label: 'Weight', description: 'Sync your weight measurements' },
  { value: 'bodyFat', label: 'Body Fat', description: 'Track body composition' },
  { value: 'workouts', label: 'Workouts', description: 'Sync your workout data' },
  { value: 'vo2Max', label: 'VO2 Max', description: 'Monitor cardio fitness' },
  {
    value: 'respiratoryRate',
    label: 'Respiratory Rate',
    description: 'Track breathing patterns',
  },
];

/**
 * Chronic Condition Options
 */
export const CHRONIC_CONDITION_OPTIONS: { value: ChronicConditionType; label: string }[] = [
  { value: 'diabetes_type1', label: 'Type 1 Diabetes' },
  { value: 'diabetes_type2', label: 'Type 2 Diabetes' },
  { value: 'prediabetic', label: 'Pre-diabetic' },
  { value: 'hypertension', label: 'High Blood Pressure' },
  { value: 'heart_disease', label: 'Heart Disease' },
  { value: 'thyroid_hypothyroid', label: 'Hypothyroidism' },
  { value: 'thyroid_hyperthyroid', label: 'Hyperthyroidism' },
  { value: 'pcos', label: 'PCOS' },
  { value: 'ibs', label: 'IBS' },
  { value: 'celiac', label: 'Celiac Disease' },
  { value: 'crohns', label: "Crohn's Disease" },
  { value: 'ulcerative_colitis', label: 'Ulcerative Colitis' },
  { value: 'asthma', label: 'Asthma' },
  { value: 'arthritis', label: 'Arthritis' },
  { value: 'osteoporosis', label: 'Osteoporosis' },
  { value: 'depression', label: 'Depression' },
  { value: 'anxiety', label: 'Anxiety' },
  { value: 'eating_disorder', label: 'Eating Disorder' },
  { value: 'other', label: 'Other' },
];

/**
 * Medication Category Options
 */
export const MEDICATION_CATEGORY_OPTIONS: { value: MedicationCategory; label: string }[] = [
  { value: 'blood_pressure', label: 'Blood Pressure' },
  { value: 'diabetes', label: 'Diabetes' },
  { value: 'thyroid', label: 'Thyroid' },
  { value: 'mental_health', label: 'Mental Health' },
  { value: 'pain', label: 'Pain Relief' },
  { value: 'heart', label: 'Heart' },
  { value: 'cholesterol', label: 'Cholesterol' },
  { value: 'hormones', label: 'Hormones' },
  { value: 'digestive', label: 'Digestive' },
  { value: 'other', label: 'Other' },
];

/**
 * Supplement Options
 */
export const SUPPLEMENT_OPTIONS: { value: SupplementName; label: string; category: string }[] = [
  { value: 'multivitamin', label: 'Multivitamin', category: 'General' },
  { value: 'vitamin_a', label: 'Vitamin A', category: 'Vitamins' },
  { value: 'vitamin_b_complex', label: 'Vitamin B Complex', category: 'Vitamins' },
  { value: 'vitamin_b12', label: 'Vitamin B12', category: 'Vitamins' },
  { value: 'vitamin_c', label: 'Vitamin C', category: 'Vitamins' },
  { value: 'vitamin_d', label: 'Vitamin D', category: 'Vitamins' },
  { value: 'vitamin_e', label: 'Vitamin E', category: 'Vitamins' },
  { value: 'vitamin_k', label: 'Vitamin K', category: 'Vitamins' },
  { value: 'iron', label: 'Iron', category: 'Minerals' },
  { value: 'magnesium', label: 'Magnesium', category: 'Minerals' },
  { value: 'zinc', label: 'Zinc', category: 'Minerals' },
  { value: 'calcium', label: 'Calcium', category: 'Minerals' },
  { value: 'potassium', label: 'Potassium', category: 'Minerals' },
  { value: 'omega3_fish_oil', label: 'Fish Oil / Omega-3', category: 'Other' },
  { value: 'probiotics', label: 'Probiotics', category: 'Other' },
  { value: 'creatine', label: 'Creatine', category: 'Sports' },
  { value: 'protein_powder', label: 'Protein Powder', category: 'Sports' },
  { value: 'collagen', label: 'Collagen', category: 'Other' },
  { value: 'melatonin', label: 'Melatonin', category: 'Other' },
  { value: 'ashwagandha', label: 'Ashwagandha', category: 'Herbal' },
  { value: 'turmeric_curcumin', label: 'Turmeric / Curcumin', category: 'Herbal' },
  { value: 'coq10', label: 'CoQ10', category: 'Other' },
  { value: 'fiber', label: 'Fiber', category: 'Other' },
  { value: 'other', label: 'Other', category: 'Other' },
];

/**
 * Allergy Options
 */
export const ALLERGY_OPTIONS: { value: Allergy; label: string }[] = [
  { value: 'peanuts', label: 'Peanuts' },
  { value: 'tree_nuts', label: 'Tree Nuts' },
  { value: 'milk', label: 'Milk / Dairy' },
  { value: 'eggs', label: 'Eggs' },
  { value: 'wheat', label: 'Wheat' },
  { value: 'soy', label: 'Soy' },
  { value: 'fish', label: 'Fish' },
  { value: 'shellfish', label: 'Shellfish' },
  { value: 'sesame', label: 'Sesame' },
  { value: 'sulfites', label: 'Sulfites' },
  { value: 'other', label: 'Other' },
];

/**
 * Nicotine Use Level Options
 */
export const NICOTINE_USE_OPTIONS: { value: NicotineUseLevel; label: string }[] = [
  { value: 'NONE', label: 'None' },
  { value: 'OCCASIONAL', label: 'Occasional' },
  { value: 'DAILY', label: 'Daily' },
  { value: 'HEAVY', label: 'Heavy' },
];

/**
 * Alcohol Use Level Options
 */
export const ALCOHOL_USE_OPTIONS: { value: AlcoholUseLevel; label: string }[] = [
  { value: 'NONE', label: 'None' },
  { value: 'OCCASIONAL', label: 'Occasional (1-2 drinks/week)' },
  { value: 'MODERATE', label: 'Moderate (3-7 drinks/week)' },
  { value: 'FREQUENT', label: 'Frequent (8+ drinks/week)' },
];

/**
 * Work Schedule Options
 */
export const WORK_SCHEDULE_OPTIONS: { value: WorkSchedule; label: string }[] = [
  { value: 'regular', label: 'Regular (9-5)' },
  { value: 'shift', label: 'Shift Work' },
  { value: 'irregular', label: 'Irregular Hours' },
  { value: 'remote', label: 'Remote / Flexible' },
  { value: 'not_working', label: 'Not Currently Working' },
];

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get step configuration by ID
 */
export function getStepConfig(stepId: number): OnboardingStepConfig | undefined {
  return ONBOARDING_STEPS.find((step) => step.id === stepId);
}

/**
 * Get step configuration by key
 */
export function getStepConfigByKey(key: string): OnboardingStepConfig | undefined {
  return ONBOARDING_STEPS.find((step) => step.key === key);
}

/**
 * Check if a step can be skipped
 */
export function canSkipStep(stepId: number): boolean {
  return SKIPPABLE_STEP_IDS.includes(stepId);
}

/**
 * Get the next step ID
 */
export function getNextStepId(currentStepId: number): number | null {
  const currentIndex = ONBOARDING_STEPS.findIndex((step) => step.id === currentStepId);
  if (currentIndex === -1 || currentIndex >= ONBOARDING_STEPS.length - 1) {
    return null;
  }
  return ONBOARDING_STEPS[currentIndex + 1].id;
}

/**
 * Get the previous step ID
 */
export function getPreviousStepId(currentStepId: number): number | null {
  const currentIndex = ONBOARDING_STEPS.findIndex((step) => step.id === currentStepId);
  if (currentIndex <= 0) {
    return null;
  }
  return ONBOARDING_STEPS[currentIndex - 1].id;
}

/**
 * Calculate progress percentage
 */
export function calculateProgress(currentStep: number, totalSteps: number): number {
  if (totalSteps <= 0) return 0;
  return Math.round(((currentStep - 1) / totalSteps) * 100);
}

// ============================================================================
// DEFAULT VALUES
// ============================================================================

export const DEFAULT_STEP1_DATA = {
  name: '',
  dateOfBirth: '',
  biologicalSex: 'PREFER_NOT_TO_SAY' as BiologicalSex,
  height: 170,
  currentWeight: 70,
  activityLevel: 'moderate' as ActivityLevel,
};

export const DEFAULT_STEP2_DATA = {
  primaryGoal: 'GENERAL_HEALTH' as PrimaryGoal,
  dietaryPreferences: [] as DietaryPreference[],
};

export const DEFAULT_STEP3_DATA = {
  notificationsEnabled: false,
  notificationTypes: [] as NotificationType[],
  healthKitEnabled: false,
  healthKitScopes: [] as HealthKitScope[],
  healthConnectEnabled: false,
  healthConnectScopes: [] as string[],
};

export const DEFAULT_STEP4_DATA = {
  chronicConditions: [],
  medications: [],
  supplements: [],
  allergies: [],
};

export const DEFAULT_STEP5_DATA = {
  nicotineUse: 'NONE' as NicotineUseLevel,
  alcoholUse: 'NONE' as AlcoholUseLevel,
  caffeineDaily: 0,
};
