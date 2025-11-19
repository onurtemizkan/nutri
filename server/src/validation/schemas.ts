import { z } from 'zod';

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
 * Password (minimum 6 characters)
 */
export const passwordSchema = z.string().min(6, 'Password must be at least 6 characters');

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
