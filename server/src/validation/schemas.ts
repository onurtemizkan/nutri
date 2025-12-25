import { z } from 'zod';
import { MIN_PASSWORD_LENGTH, MAX_PASSWORD_LENGTH, PASSWORD_ERRORS } from '../config/constants';

/**
 * Centralized Zod validation schemas
 * Eliminates duplication across controllers and ensures consistency
 */

// ============================================================================
// ENUM SCHEMAS
// ============================================================================

/**
 * Health Metric Types
 * Matches Prisma HealthMetricType enum
 */
export const healthMetricTypeSchema = z.enum([
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
  'HEART_RATE_VARIABILITY_RMSSD',
  'BLOOD_PRESSURE_SYSTOLIC',
  'BLOOD_PRESSURE_DIASTOLIC',
  'RESPIRATORY_RATE',
  'OXYGEN_SATURATION',
  'VO2_MAX',
  'SLEEP_DURATION',
  'DEEP_SLEEP_DURATION',
  'REM_SLEEP_DURATION',
  'SLEEP_EFFICIENCY',
  'SLEEP_SCORE',
  'STEPS',
  'ACTIVE_CALORIES',
  'TOTAL_CALORIES',
  'EXERCISE_MINUTES',
  'STANDING_HOURS',
  'RECOVERY_SCORE',
  'STRAIN_SCORE',
  'READINESS_SCORE',
  'BODY_FAT_PERCENTAGE',
  'MUSCLE_MASS',
  'BONE_MASS',
  'WATER_PERCENTAGE',
  'SKIN_TEMPERATURE',
  'BLOOD_GLUCOSE',
  'STRESS_LEVEL',
]);

/**
 * Activity Types
 * Matches Prisma ActivityType enum
 */
export const activityTypeSchema = z.enum([
  'RUNNING',
  'CYCLING',
  'SWIMMING',
  'WALKING',
  'HIKING',
  'ROWING',
  'ELLIPTICAL',
  'WEIGHT_TRAINING',
  'BODYWEIGHT',
  'CROSSFIT',
  'POWERLIFTING',
  'BASKETBALL',
  'SOCCER',
  'TENNIS',
  'GOLF',
  'YOGA',
  'PILATES',
  'STRETCHING',
  'MARTIAL_ARTS',
  'DANCE',
  'OTHER',
]);

/**
 * Activity Intensity Levels
 * Matches Prisma ActivityIntensity enum
 */
export const activityIntensitySchema = z.enum(['LOW', 'MODERATE', 'HIGH', 'MAXIMUM']);

/**
 * Meal Types
 */
export const mealTypeSchema = z.enum(['breakfast', 'lunch', 'dinner', 'snack']);

/**
 * Health Metric Sources
 */
export const healthMetricSourceSchema = z.enum([
  'apple_health',
  'fitbit',
  'garmin',
  'oura',
  'whoop',
  'manual',
]);

/**
 * Activity Sources
 */
export const activitySourceSchema = z.enum(['apple_health', 'strava', 'garmin', 'manual']);

// ============================================================================
// COMMON FIELD SCHEMAS
// ============================================================================

/**
 * ISO 8601 datetime string
 */
export const datetimeSchema = z.string().datetime();

/**
 * Positive number (for quantities, calories, etc.)
 */
export const positiveNumberSchema = z.number().min(0);

/**
 * Positive integer (for steps, duration, etc.)
 */
export const positiveIntSchema = z.number().int().min(0);

/**
 * Non-empty string
 */
export const nonEmptyStringSchema = z.string().min(1);

/**
 * Email address
 */
export const emailSchema = z.string().email();

/**
 * Password schema with complexity requirements
 * - Minimum 12 characters (OWASP recommendation)
 * - Maximum 128 characters
 * - At least one uppercase letter
 * - At least one lowercase letter
 * - At least one number
 */
export const passwordSchema = z
  .string()
  .min(MIN_PASSWORD_LENGTH, PASSWORD_ERRORS.TOO_SHORT)
  .max(MAX_PASSWORD_LENGTH, PASSWORD_ERRORS.TOO_LONG)
  .refine((val) => /[A-Z]/.test(val), {
    message: PASSWORD_ERRORS.MISSING_UPPERCASE,
  })
  .refine((val) => /[a-z]/.test(val), {
    message: PASSWORD_ERRORS.MISSING_LOWERCASE,
  })
  .refine((val) => /[0-9]/.test(val), {
    message: PASSWORD_ERRORS.MISSING_NUMBER,
  });

/**
 * Non-negative number schema (for micronutrients that can be 0)
 */
const nonNegativeNumberSchema = z.number().min(0, 'Value must be 0 or greater');

// ============================================================================
// ENTITY SCHEMAS
// ============================================================================

/**
 * Health Metric Creation Schema
 */
export const createHealthMetricSchema = z.object({
  metricType: healthMetricTypeSchema,
  value: z.number(),
  unit: nonEmptyStringSchema,
  recordedAt: datetimeSchema,
  source: healthMetricSourceSchema,
  sourceId: z.string().optional(),
  metadata: z.record(z.any()).optional(),
});

/**
 * Bulk Health Metrics Creation Schema
 */
export const bulkCreateHealthMetricsSchema = z.object({
  metrics: z.array(createHealthMetricSchema).min(1, 'At least one metric is required'),
});

/**
 * Activity Creation Schema
 */
export const createActivitySchema = z.object({
  activityType: activityTypeSchema,
  intensity: activityIntensitySchema,
  startedAt: datetimeSchema,
  endedAt: datetimeSchema,
  duration: positiveIntSchema.min(1, 'Duration must be at least 1 minute'),
  caloriesBurned: positiveNumberSchema.optional(),
  averageHeartRate: positiveNumberSchema.optional(),
  maxHeartRate: positiveNumberSchema.optional(),
  distance: positiveNumberSchema.optional(),
  steps: positiveIntSchema.optional(),
  source: activitySourceSchema,
  sourceId: z.string().optional(),
  notes: z.string().optional(),
});

/**
 * Activity Update Schema
 */
export const updateActivitySchema = z.object({
  activityType: activityTypeSchema.optional(),
  intensity: activityIntensitySchema.optional(),
  startedAt: datetimeSchema.optional(),
  endedAt: datetimeSchema.optional(),
  duration: positiveIntSchema.min(1).optional(),
  caloriesBurned: positiveNumberSchema.optional(),
  averageHeartRate: positiveNumberSchema.optional(),
  maxHeartRate: positiveNumberSchema.optional(),
  distance: positiveNumberSchema.optional(),
  steps: positiveIntSchema.optional(),
  notes: z.string().optional(),
});

/**
 * Bulk Activities Creation Schema
 */
export const bulkCreateActivitiesSchema = z.object({
  activities: z.array(createActivitySchema).min(1, 'At least one activity is required'),
});

/**
 * Meal Creation Schema
 */
export const createMealSchema = z.object({
  mealType: mealTypeSchema,
  name: nonEmptyStringSchema,
  calories: positiveNumberSchema,
  protein: positiveNumberSchema,
  carbs: positiveNumberSchema,
  fat: positiveNumberSchema,
  fiber: positiveNumberSchema.optional(),
  sugar: positiveNumberSchema.optional(),

  // Fat breakdown (optional)
  saturatedFat: nonNegativeNumberSchema.optional(),
  transFat: nonNegativeNumberSchema.optional(),
  cholesterol: nonNegativeNumberSchema.optional(),

  // Minerals (optional) - all in mg
  sodium: nonNegativeNumberSchema.optional(),
  potassium: nonNegativeNumberSchema.optional(),
  calcium: nonNegativeNumberSchema.optional(),
  iron: nonNegativeNumberSchema.optional(),
  magnesium: nonNegativeNumberSchema.optional(),
  zinc: nonNegativeNumberSchema.optional(),
  phosphorus: nonNegativeNumberSchema.optional(),

  // Vitamins (optional) - various units
  vitaminA: nonNegativeNumberSchema.optional(), // mcg RAE
  vitaminC: nonNegativeNumberSchema.optional(), // mg
  vitaminD: nonNegativeNumberSchema.optional(), // mcg
  vitaminE: nonNegativeNumberSchema.optional(), // mg
  vitaminK: nonNegativeNumberSchema.optional(), // mcg
  vitaminB6: nonNegativeNumberSchema.optional(), // mg
  vitaminB12: nonNegativeNumberSchema.optional(), // mcg
  folate: nonNegativeNumberSchema.optional(), // mcg DFE
  thiamin: nonNegativeNumberSchema.optional(), // mg (B1)
  riboflavin: nonNegativeNumberSchema.optional(), // mg (B2)
  niacin: nonNegativeNumberSchema.optional(), // mg (B3)

  servingSize: z.string().optional(),
  notes: z.string().optional(),
  imageUrl: z.string().url().optional(),
  consumedAt: datetimeSchema.optional(),
});

/**
 * Meal Update Schema
 */
export const updateMealSchema = createMealSchema.partial();

// ============================================================================
// AUTH SCHEMAS
// ============================================================================

/**
 * User Registration Schema
 */
export const registerSchema = z.object({
  name: nonEmptyStringSchema,
  email: emailSchema,
  password: passwordSchema,
});

/**
 * User Login Schema
 */
export const loginSchema = z.object({
  email: emailSchema,
  password: nonEmptyStringSchema,
});

/**
 * Password Reset Request Schema
 */
export const forgotPasswordSchema = z.object({
  email: emailSchema,
});

/**
 * Password Reset Schema
 */
export const resetPasswordSchema = z.object({
  token: nonEmptyStringSchema,
  newPassword: passwordSchema,
});

/**
 * Apple Sign-In Schema
 */
export const appleSignInSchema = z.object({
  identityToken: nonEmptyStringSchema,
  authorizationCode: nonEmptyStringSchema,
  user: z
    .object({
      email: emailSchema.optional(),
      name: z
        .object({
          firstName: z.string().optional(),
          lastName: z.string().optional(),
        })
        .optional(),
    })
    .optional(),
});

// ============================================================================
// SUPPLEMENT SCHEMAS
// ============================================================================

/**
 * Supplement Frequency
 * Matches Prisma SupplementFrequency enum
 */
export const supplementFrequencySchema = z.enum([
  'DAILY',
  'TWICE_DAILY',
  'THREE_TIMES_DAILY',
  'WEEKLY',
  'EVERY_OTHER_DAY',
  'AS_NEEDED',
]);

/**
 * Supplement Time of Day
 * Matches Prisma SupplementTimeOfDay enum
 */
export const supplementTimeOfDaySchema = z.enum([
  'MORNING',
  'AFTERNOON',
  'EVENING',
  'BEFORE_BED',
  'WITH_BREAKFAST',
  'WITH_LUNCH',
  'WITH_DINNER',
  'EMPTY_STOMACH',
]);

/**
 * Supplement Creation Schema
 */
export const createSupplementSchema = z.object({
  name: nonEmptyStringSchema.max(100, 'Name must be 100 characters or less'),
  brand: z.string().max(100).optional(),
  dosageAmount: positiveNumberSchema,
  dosageUnit: nonEmptyStringSchema.max(20, 'Unit must be 20 characters or less'),
  frequency: supplementFrequencySchema.optional(),
  timesPerDay: z.number().int().min(1).max(10).optional(),
  timeOfDay: z.array(supplementTimeOfDaySchema).optional(),
  withFood: z.boolean().optional(),
  isActive: z.boolean().optional(),
  startDate: datetimeSchema.optional(),
  endDate: datetimeSchema.optional().nullable(),
  notes: z.string().max(500).optional(),
  color: z
    .string()
    .regex(/^#[0-9A-Fa-f]{6}$/, 'Invalid hex color')
    .optional(),

  // Micronutrient content (estimated or from barcode)
  // Vitamins (optional) - values with reasonable max limits
  vitaminA: nonNegativeNumberSchema.max(50000).optional(), // mcg RAE
  vitaminC: nonNegativeNumberSchema.max(10000).optional(), // mg
  vitaminD: nonNegativeNumberSchema.max(500).optional(), // mcg
  vitaminE: nonNegativeNumberSchema.max(1000).optional(), // mg
  vitaminK: nonNegativeNumberSchema.max(5000).optional(), // mcg
  vitaminB6: nonNegativeNumberSchema.max(500).optional(), // mg
  vitaminB12: nonNegativeNumberSchema.max(50000).optional(), // mcg
  folate: nonNegativeNumberSchema.max(10000).optional(), // mcg DFE
  thiamin: nonNegativeNumberSchema.max(500).optional(), // mg (B1)
  riboflavin: nonNegativeNumberSchema.max(500).optional(), // mg (B2)
  niacin: nonNegativeNumberSchema.max(2000).optional(), // mg (B3)

  // Minerals (optional)
  calcium: nonNegativeNumberSchema.max(5000).optional(), // mg
  iron: nonNegativeNumberSchema.max(200).optional(), // mg
  magnesium: nonNegativeNumberSchema.max(2000).optional(), // mg
  zinc: nonNegativeNumberSchema.max(200).optional(), // mg
  potassium: nonNegativeNumberSchema.max(5000).optional(), // mg
  sodium: nonNegativeNumberSchema.max(2000).optional(), // mg
  phosphorus: nonNegativeNumberSchema.max(3000).optional(), // mg

  // Special nutrients (optional)
  omega3: nonNegativeNumberSchema.max(10000).optional(), // mg
});

/**
 * Supplement Update Schema
 */
export const updateSupplementSchema = createSupplementSchema.partial();

/**
 * Supplement Log Creation Schema
 */
export const createSupplementLogSchema = z.object({
  supplementId: nonEmptyStringSchema,
  takenAt: datetimeSchema.optional(),
  dosageAmount: positiveNumberSchema.optional(),
  notes: z.string().max(500).optional(),
  skipped: z.boolean().optional(),
});

/**
 * Bulk Supplement Log Creation Schema
 */
export const bulkCreateSupplementLogsSchema = z.object({
  logs: z.array(createSupplementLogSchema).min(1, 'At least one log is required'),
});

// ============================================================================
// USDA FOOD SEARCH SCHEMAS
// ============================================================================

/**
 * USDA Data Types
 */
export const usdaDataTypeSchema = z.enum([
  'Foundation',
  'SR Legacy',
  'Survey (FNDDS)',
  'Branded',
  'Experimental',
]);

/**
 * Food Search Query Parameters
 */
export const foodSearchQuerySchema = z.object({
  q: nonEmptyStringSchema.min(1, 'Search query is required').max(200),
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(25),
  dataType: z
    .string()
    .optional()
    .transform((val) => {
      if (!val) return undefined;
      return val.split(',').filter(Boolean);
    }),
  sortBy: z.enum(['dataType.keyword', 'description', 'fdcId', 'publishedDate']).optional(),
  sortOrder: z.enum(['asc', 'desc']).optional(),
  brandOwner: z.string().max(100).optional(),
});

/**
 * FDC ID parameter schema
 */
export const fdcIdParamSchema = z.object({
  fdcId: z.coerce.number().int().positive('Invalid FDC ID'),
});

/**
 * Multiple FDC IDs body schema
 */
export const bulkFdcIdsSchema = z.object({
  fdcIds: z.array(z.number().int().positive()).min(1).max(20),
  format: z.enum(['abridged', 'full']).optional(),
  nutrients: z.array(z.number().int().positive()).optional(),
});

/**
 * Classification hints for hybrid search
 */
export const classificationHintsSchema = z.object({
  coarseCategory: z.string().optional(),
  finegrainedSuggestions: z.array(z.string()).optional(),
  colorProfile: z.record(z.number()).optional(),
  cookingMethod: z.string().optional(),
  brandDetected: z.string().optional(),
  portionEstimate: z.number().positive().optional(),
});

/**
 * Classify and search request body
 */
export const classifyAndSearchSchema = z.object({
  image: z.string().min(1, 'Image data is required'),
  dimensions: z
    .object({
      width: z.number().positive(),
      height: z.number().positive(),
      depth: z.number().positive().optional(),
    })
    .optional(),
});

// ============================================================================
// FOOD FEEDBACK SCHEMAS
// ============================================================================

/**
 * Food feedback submission request body
 */
export const submitFoodFeedbackSchema = z.object({
  imageHash: nonEmptyStringSchema.min(8, 'Image hash is required'),
  classificationId: z.string().optional(),
  originalPrediction: nonEmptyStringSchema.min(1, 'Original prediction is required'),
  originalConfidence: z.number().min(0).max(1).default(0),
  originalCategory: z.string().optional(),
  selectedFdcId: z.number().int().positive('Valid FDC ID is required'),
  selectedFoodName: nonEmptyStringSchema.min(1, 'Selected food name is required'),
  wasCorrect: z.boolean(),
  classificationHints: z.record(z.unknown()).optional(),
  userDescription: z.string().max(500).optional(),
});

/**
 * Feedback list query parameters
 */
export const feedbackListQuerySchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  status: z.enum(['pending', 'approved', 'rejected', 'applied']).optional(),
});

/**
 * Update feedback status request body
 */
export const updateFeedbackStatusSchema = z.object({
  status: z.enum(['pending', 'approved', 'rejected', 'applied']),
});

/**
 * Food feedback ID parameter
 */
export const feedbackIdParamSchema = z.object({
  feedbackId: nonEmptyStringSchema,
});

/**
 * Legacy food feedback schema (for backwards compatibility)
 */
export const foodFeedbackSchema = z.object({
  classificationId: z.string().optional(),
  originalPrediction: z.string().optional(),
  selectedFdcId: z.number().int().positive('Valid FDC ID is required'),
  wasCorrect: z.boolean(),
  imageHash: z.string().optional(),
});

// ============================================================================
// PUSH NOTIFICATION SCHEMAS
// ============================================================================

/**
 * Device Platform enum
 */
export const devicePlatformSchema = z.enum(['IOS', 'ANDROID']);

/**
 * Notification Category enum
 */
export const notificationCategorySchema = z.enum([
  'MEAL_REMINDER',
  'GOAL_PROGRESS',
  'HEALTH_INSIGHT',
  'SUPPLEMENT_REMINDER',
  'STREAK_ALERT',
  'WEEKLY_SUMMARY',
  'MARKETING',
  'SYSTEM',
]);

/**
 * Notification Status enum
 */
export const notificationStatusSchema = z.enum([
  'PENDING',
  'SENT',
  'DELIVERED',
  'OPENED',
  'FAILED',
]);

/**
 * Time format schema (HH:mm)
 */
const timeFormatSchema = z
  .string()
  .regex(/^([01]\d|2[0-3]):([0-5]\d)$/, 'Time must be in HH:mm format');

/**
 * Register Device Schema
 */
export const registerDeviceSchema = z.object({
  token: nonEmptyStringSchema.min(10, 'Device token is required'),
  platform: devicePlatformSchema,
  expoPushToken: z.string().optional(),
  deviceModel: z.string().optional(),
  osVersion: z.string().optional(),
  appVersion: z.string().optional(),
});

/**
 * Unregister Device Schema
 */
export const unregisterDeviceSchema = z.object({
  token: nonEmptyStringSchema.min(10, 'Device token is required'),
});

/**
 * Meal Reminder Times Schema
 */
const mealReminderTimesSchema = z.object({
  breakfast: timeFormatSchema.optional().nullable(),
  lunch: timeFormatSchema.optional().nullable(),
  dinner: timeFormatSchema.optional().nullable(),
  snack: timeFormatSchema.optional().nullable(),
});

/**
 * Update Notification Preferences Schema
 */
export const updateNotificationPreferencesSchema = z.object({
  enabled: z.boolean().optional(),
  enabledCategories: z.array(notificationCategorySchema).optional(),
  quietHoursEnabled: z.boolean().optional(),
  quietHoursStart: timeFormatSchema.optional().nullable(),
  quietHoursEnd: timeFormatSchema.optional().nullable(),
  mealReminderTimes: mealReminderTimesSchema.optional(),
  settings: z.record(z.unknown()).optional(),
});

/**
 * Get Notification History Query Schema
 */
export const getNotificationHistoryQuerySchema = z.object({
  category: notificationCategorySchema.optional(),
  status: notificationStatusSchema.optional(),
  startDate: datetimeSchema.optional(),
  endDate: datetimeSchema.optional(),
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

/**
 * Track Notification Schema
 */
export const trackNotificationSchema = z.object({
  notificationLogId: nonEmptyStringSchema,
  action: z.enum(['delivered', 'opened']),
  actionTaken: z.string().optional(),
});

/**
 * Test Notification Schema (dev only)
 */
export const testNotificationSchema = z.object({
  title: z.string().optional().default('Test Notification'),
  body: z.string().optional().default('This is a test push notification'),
  category: notificationCategorySchema.optional().default('SYSTEM'),
});

// ============================================================================
// ONBOARDING SCHEMAS
// ============================================================================

/**
 * Biological Sex - matches Prisma BiologicalSex enum
 */
export const biologicalSexSchema = z.enum(['MALE', 'FEMALE', 'OTHER', 'PREFER_NOT_TO_SAY']);

/**
 * Primary Goal - matches Prisma PrimaryGoal enum
 */
export const primaryGoalSchema = z.enum([
  'WEIGHT_LOSS',
  'MUSCLE_GAIN',
  'MAINTENANCE',
  'GENERAL_HEALTH',
  'ATHLETIC_PERFORMANCE',
  'BETTER_SLEEP',
  'STRESS_REDUCTION',
]);

/**
 * Nicotine Use Level - matches Prisma NicotineUseLevel enum
 */
export const nicotineUseLevelSchema = z.enum(['NONE', 'OCCASIONAL', 'DAILY', 'HEAVY']);

/**
 * Alcohol Use Level - matches Prisma AlcoholUseLevel enum
 */
export const alcoholUseLevelSchema = z.enum(['NONE', 'OCCASIONAL', 'MODERATE', 'FREQUENT']);

/**
 * Activity Level (string-based for user model)
 */
export const activityLevelSchema = z.enum([
  'sedentary',
  'light',
  'moderate',
  'active',
  'veryActive',
]);

/**
 * Dietary Preferences
 */
export const dietaryPreferencesSchema = z.array(
  z.enum([
    'vegetarian',
    'vegan',
    'pescatarian',
    'keto',
    'paleo',
    'gluten_free',
    'dairy_free',
    'nut_free',
    'low_carb',
    'low_fat',
    'mediterranean',
    'halal',
    'kosher',
    'none',
  ])
);

// ============================================================================
// ONBOARDING STEP SCHEMAS
// ============================================================================

/**
 * Step 1: Profile Basics
 * Collects essential profile information for health calculations
 */
export const onboardingStep1Schema = z.object({
  name: nonEmptyStringSchema.max(100, 'Name cannot exceed 100 characters'),
  dateOfBirth: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be in YYYY-MM-DD format'),
  biologicalSex: biologicalSexSchema,
  height: z
    .number()
    .min(50, 'Height must be at least 50 cm')
    .max(300, 'Height cannot exceed 300 cm'),
  currentWeight: z
    .number()
    .min(20, 'Weight must be at least 20 kg')
    .max(500, 'Weight cannot exceed 500 kg'),
  activityLevel: activityLevelSchema,
});

/**
 * Step 2: Health Goals
 * Defines user's nutrition and fitness goals
 */
export const onboardingStep2Schema = z.object({
  primaryGoal: primaryGoalSchema,
  goalWeight: z
    .number()
    .min(20, 'Goal weight must be at least 20 kg')
    .max(500, 'Goal weight cannot exceed 500 kg')
    .optional(),
  dietaryPreferences: dietaryPreferencesSchema.optional().default([]),
  // Auto-calculated or custom macro targets
  customMacros: z
    .object({
      goalCalories: z.number().min(500).max(10000).optional(),
      goalProtein: z.number().min(0).max(500).optional(),
      goalCarbs: z.number().min(0).max(1000).optional(),
      goalFat: z.number().min(0).max(500).optional(),
    })
    .optional(),
});

/**
 * Step 3: Permissions
 * App permission requests (notifications, health data access)
 */
export const onboardingStep3Schema = z.object({
  notificationsEnabled: z.boolean(),
  notificationTypes: z
    .array(z.enum(['meal_reminders', 'insights', 'goals', 'weekly_summary']))
    .optional()
    .default([]),
  healthKitEnabled: z.boolean(),
  healthKitScopes: z
    .array(
      z.enum([
        'heartRate',
        'restingHeartRate',
        'hrv',
        'steps',
        'activeEnergy',
        'sleep',
        'weight',
        'bodyFat',
        'workouts',
        'vo2Max',
        'respiratoryRate',
      ])
    )
    .optional()
    .default([]),
  healthConnectEnabled: z.boolean().optional().default(false),
  healthConnectScopes: z.array(z.string()).optional().default([]),
});

/**
 * Chronic Condition Item Schema
 */
const chronicConditionSchema = z.object({
  type: z.enum([
    'diabetes_type1',
    'diabetes_type2',
    'prediabetic',
    'hypertension',
    'heart_disease',
    'thyroid_hypothyroid',
    'thyroid_hyperthyroid',
    'pcos',
    'ibs',
    'celiac',
    'crohns',
    'ulcerative_colitis',
    'asthma',
    'arthritis',
    'osteoporosis',
    'depression',
    'anxiety',
    'eating_disorder',
    'other',
  ]),
  customType: z.string().max(100).optional(), // For "other" type
  diagnosedYear: z.number().int().min(1900).max(new Date().getFullYear()).optional(),
  notes: z.string().max(500).optional(),
});

/**
 * Medication Item Schema
 */
const medicationSchema = z.object({
  name: nonEmptyStringSchema.max(100),
  dosage: z.string().max(50).optional(),
  frequency: z.enum(['as_needed', 'daily', 'twice_daily', 'weekly', 'monthly']).optional(),
  category: z
    .enum([
      'blood_pressure',
      'diabetes',
      'thyroid',
      'mental_health',
      'pain',
      'heart',
      'cholesterol',
      'hormones',
      'digestive',
      'other',
    ])
    .optional(),
});

/**
 * Supplement Item Schema (for onboarding - simpler than full supplement tracking)
 */
const onboardingSupplementSchema = z.object({
  name: z.enum([
    'vitamin_a',
    'vitamin_b_complex',
    'vitamin_b12',
    'vitamin_c',
    'vitamin_d',
    'vitamin_e',
    'vitamin_k',
    'iron',
    'magnesium',
    'zinc',
    'calcium',
    'potassium',
    'omega3_fish_oil',
    'probiotics',
    'creatine',
    'protein_powder',
    'collagen',
    'melatonin',
    'ashwagandha',
    'turmeric_curcumin',
    'coq10',
    'fiber',
    'multivitamin',
    'other',
  ]),
  customName: z.string().max(100).optional(), // For "other" type
  dosage: z.string().max(50).optional(),
  frequency: z.enum(['daily', 'twice_daily', 'weekly', 'as_needed']).optional(),
});

/**
 * Step 4: Health Background (Optional)
 * Collects health history for better insights
 */
export const onboardingStep4Schema = z.object({
  chronicConditions: z.array(chronicConditionSchema).optional().default([]),
  medications: z.array(medicationSchema).optional().default([]),
  supplements: z.array(onboardingSupplementSchema).optional().default([]),
  allergies: z
    .array(
      z.enum([
        'peanuts',
        'tree_nuts',
        'milk',
        'eggs',
        'wheat',
        'soy',
        'fish',
        'shellfish',
        'sesame',
        'sulfites',
        'other',
      ])
    )
    .optional()
    .default([]),
  allergyNotes: z.string().max(500).optional(), // For custom allergies
});

/**
 * Step 5: Lifestyle Factors (Optional)
 * Collects lifestyle information that affects health metrics
 */
export const onboardingStep5Schema = z.object({
  // Nicotine
  nicotineUse: nicotineUseLevelSchema.optional(),
  nicotineType: z.enum(['cigarettes', 'vape', 'chewing', 'patches', 'other']).optional(),

  // Alcohol
  alcoholUse: alcoholUseLevelSchema.optional(),

  // Caffeine (cups per day)
  caffeineDaily: z.number().int().min(0).max(20).optional(),

  // Sleep patterns
  typicalBedtime: z
    .string()
    .regex(/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Bedtime must be in HH:MM format')
    .optional(),
  typicalWakeTime: z
    .string()
    .regex(/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/, 'Wake time must be in HH:MM format')
    .optional(),
  sleepQuality: z.number().int().min(1).max(10).optional(),

  // Stress
  stressLevel: z.number().int().min(1).max(10).optional(),

  // Work schedule
  workSchedule: z.enum(['regular', 'shift', 'irregular', 'remote', 'not_working']).optional(),
});

/**
 * Step 6: Completion
 * No data required - just marks onboarding as complete
 */
export const onboardingStep6Schema = z.object({
  acknowledged: z.literal(true),
});

// ============================================================================
// ONBOARDING API SCHEMAS
// ============================================================================

/**
 * Start Onboarding Request
 */
export const startOnboardingSchema = z.object({
  version: z.string().optional().default('1.0'),
});

/**
 * Save Step Data Request
 */
export const saveStepSchema = z.object({
  stepNumber: z.number().int().min(1).max(6),
  data: z.union([
    onboardingStep1Schema,
    onboardingStep2Schema,
    onboardingStep3Schema,
    onboardingStep4Schema,
    onboardingStep5Schema,
    onboardingStep6Schema,
  ]),
});

/**
 * Skip Step Request
 */
export const skipStepSchema = z.object({
  stepNumber: z.number().int().min(1).max(6),
});

/**
 * Complete Onboarding Request
 */
export const completeOnboardingSchema = z.object({
  skipRemaining: z.boolean().optional().default(false),
});

// ============================================================================
// TYPE EXPORTS
// ============================================================================

export type OnboardingStep1Data = z.infer<typeof onboardingStep1Schema>;
export type OnboardingStep2Data = z.infer<typeof onboardingStep2Schema>;
export type OnboardingStep3Data = z.infer<typeof onboardingStep3Schema>;
export type OnboardingStep4Data = z.infer<typeof onboardingStep4Schema>;
export type OnboardingStep5Data = z.infer<typeof onboardingStep5Schema>;
export type OnboardingStep6Data = z.infer<typeof onboardingStep6Schema>;
export type BiologicalSex = z.infer<typeof biologicalSexSchema>;
export type PrimaryGoal = z.infer<typeof primaryGoalSchema>;
export type NicotineUseLevel = z.infer<typeof nicotineUseLevelSchema>;
export type AlcoholUseLevel = z.infer<typeof alcoholUseLevelSchema>;
export type ActivityLevel = z.infer<typeof activityLevelSchema>;
export type DietaryPreference = z.infer<typeof dietaryPreferencesSchema>[number];
