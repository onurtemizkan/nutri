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
 * Supplement Categories
 * Matches Prisma SupplementCategory enum
 */
export const supplementCategorySchema = z.enum([
  'AMINO_ACID',
  'VITAMIN',
  'MINERAL',
  'PERFORMANCE',
  'HERBAL',
  'PROTEIN',
  'FATTY_ACID',
  'PROBIOTIC',
  'OTHER',
]);

/**
 * Schedule Types
 * Matches Prisma ScheduleType enum
 */
export const scheduleTypeSchema = z.enum([
  'ONE_TIME',
  'DAILY',
  'DAILY_MULTIPLE',
  'WEEKLY',
  'INTERVAL',
]);

/**
 * Supplement Source
 * Matches Prisma SupplementSource enum
 */
export const supplementSourceSchema = z.enum(['SCHEDULED', 'MANUAL', 'QUICK_LOG']);

/**
 * Time string in HH:MM format
 */
export const timeStringSchema = z.string().regex(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/, {
  message: 'Time must be in HH:MM format',
});

/**
 * Day of week for weekly schedules
 */
export const dayOfWeekSchema = z.enum([
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
]);

/**
 * Weekly schedule configuration
 * { "monday": ["08:00", "20:00"], "wednesday": ["08:00"] }
 */
export const weeklyScheduleSchema = z.record(dayOfWeekSchema, z.array(timeStringSchema));

/**
 * Get Supplements Query Schema
 */
export const getSupplementsQuerySchema = z.object({
  category: supplementCategorySchema.optional(),
  search: z.string().optional(),
});

/**
 * Create User Supplement Schema (Configure schedule)
 */
export const createUserSupplementSchema = z
  .object({
    supplementId: nonEmptyStringSchema,
    dosage: nonEmptyStringSchema,
    unit: nonEmptyStringSchema,
    scheduleType: scheduleTypeSchema,
    scheduleTimes: z.array(timeStringSchema).optional(),
    weeklySchedule: weeklyScheduleSchema.optional(),
    intervalDays: z.number().int().min(1).max(365).optional(),
    startDate: datetimeSchema,
    endDate: datetimeSchema.optional(),
    notes: z.string().optional(),
  })
  .refine(
    (data) => {
      // DAILY_MULTIPLE requires scheduleTimes
      if (data.scheduleType === 'DAILY_MULTIPLE') {
        return data.scheduleTimes && data.scheduleTimes.length > 0;
      }
      return true;
    },
    {
      message: 'DAILY_MULTIPLE schedule requires at least one time in scheduleTimes',
      path: ['scheduleTimes'],
    }
  )
  .refine(
    (data) => {
      // WEEKLY requires weeklySchedule
      if (data.scheduleType === 'WEEKLY') {
        return (
          data.weeklySchedule && Object.keys(data.weeklySchedule).length > 0
        );
      }
      return true;
    },
    {
      message: 'WEEKLY schedule requires weeklySchedule with at least one day',
      path: ['weeklySchedule'],
    }
  )
  .refine(
    (data) => {
      // INTERVAL requires intervalDays
      if (data.scheduleType === 'INTERVAL') {
        return data.intervalDays && data.intervalDays >= 1;
      }
      return true;
    },
    {
      message: 'INTERVAL schedule requires intervalDays (minimum 1)',
      path: ['intervalDays'],
    }
  )
  .refine(
    (data) => {
      // If endDate is provided, it must be after startDate
      if (data.endDate) {
        return new Date(data.endDate) > new Date(data.startDate);
      }
      return true;
    },
    {
      message: 'endDate must be after startDate',
      path: ['endDate'],
    }
  );

/**
 * Update User Supplement Schema
 */
export const updateUserSupplementSchema = z
  .object({
    dosage: nonEmptyStringSchema.optional(),
    unit: nonEmptyStringSchema.optional(),
    scheduleType: scheduleTypeSchema.optional(),
    scheduleTimes: z.array(timeStringSchema).optional(),
    weeklySchedule: weeklyScheduleSchema.optional(),
    intervalDays: z.number().int().min(1).max(365).optional(),
    startDate: datetimeSchema.optional(),
    endDate: datetimeSchema.nullable().optional(),
    isActive: z.boolean().optional(),
    notes: z.string().nullable().optional(),
  })
  .refine(
    (data) => {
      // If scheduleType is changed to DAILY_MULTIPLE, scheduleTimes must be provided
      if (data.scheduleType === 'DAILY_MULTIPLE') {
        return data.scheduleTimes && data.scheduleTimes.length > 0;
      }
      return true;
    },
    {
      message: 'DAILY_MULTIPLE schedule requires at least one time in scheduleTimes',
      path: ['scheduleTimes'],
    }
  )
  .refine(
    (data) => {
      // If scheduleType is changed to WEEKLY, weeklySchedule must be provided
      if (data.scheduleType === 'WEEKLY') {
        return (
          data.weeklySchedule && Object.keys(data.weeklySchedule).length > 0
        );
      }
      return true;
    },
    {
      message: 'WEEKLY schedule requires weeklySchedule with at least one day',
      path: ['weeklySchedule'],
    }
  )
  .refine(
    (data) => {
      // If scheduleType is changed to INTERVAL, intervalDays must be provided
      if (data.scheduleType === 'INTERVAL') {
        return data.intervalDays && data.intervalDays >= 1;
      }
      return true;
    },
    {
      message: 'INTERVAL schedule requires intervalDays (minimum 1)',
      path: ['intervalDays'],
    }
  );

/**
 * Create Supplement Log Schema
 */
export const createSupplementLogSchema = z.object({
  userSupplementId: z.string().optional(),
  supplementId: nonEmptyStringSchema,
  dosage: nonEmptyStringSchema,
  unit: nonEmptyStringSchema,
  takenAt: datetimeSchema,
  scheduledFor: datetimeSchema.optional(),
  source: supplementSourceSchema,
  notes: z.string().optional(),
});

/**
 * Update Supplement Log Schema
 */
export const updateSupplementLogSchema = z.object({
  dosage: nonEmptyStringSchema.optional(),
  unit: nonEmptyStringSchema.optional(),
  takenAt: datetimeSchema.optional(),
  notes: z.string().nullable().optional(),
});

/**
 * Bulk Create Supplement Logs Schema
 */
export const bulkCreateSupplementLogsSchema = z.object({
  logs: z.array(createSupplementLogSchema).min(1, 'At least one log is required'),
});

/**
 * Get Supplement Logs Query Schema
 */
export const getSupplementLogsQuerySchema = z.object({
  startDate: datetimeSchema.optional(),
  endDate: datetimeSchema.optional(),
  supplementId: z.string().optional(),
  userSupplementId: z.string().optional(),
});

/**
 * Get Scheduled Supplements Query Schema
 */
export const getScheduledSupplementsQuerySchema = z.object({
  date: datetimeSchema.optional(),
});
