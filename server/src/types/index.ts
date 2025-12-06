import { Request } from 'express';

export interface AuthenticatedRequest extends Request {
  userId?: string;
}

export interface JWTPayload {
  userId: string;
}

export interface RegisterInput {
  email: string;
  password: string;
  name: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

// Enums for type safety
export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';
export type ActivityLevel =
  | 'sedentary'
  | 'light'
  | 'moderate'
  | 'active'
  | 'veryActive';

export interface CreateMealInput {
  name: string;
  mealType: MealType;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  servingSize?: string;
  notes?: string;
  consumedAt?: Date;
}

export interface UpdateMealInput {
  name?: string;
  mealType?: MealType;
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
  fiber?: number;
  sugar?: number;
  servingSize?: string;
  notes?: string;
  consumedAt?: Date;
}

export interface UpdateUserProfileInput {
  name?: string;
  goalCalories?: number;
  goalProtein?: number;
  goalCarbs?: number;
  goalFat?: number;
  currentWeight?: number;
  goalWeight?: number;
  height?: number;
  activityLevel?: ActivityLevel;
}

// ============================================================================
// ML ENGINE TYPES
// ============================================================================

// Health Metrics Types
export type HealthMetricType =
  // Cardiovascular
  | 'RESTING_HEART_RATE'
  | 'HEART_RATE_VARIABILITY_SDNN'
  | 'HEART_RATE_VARIABILITY_RMSSD'
  | 'BLOOD_PRESSURE_SYSTOLIC'
  | 'BLOOD_PRESSURE_DIASTOLIC'
  // Respiratory
  | 'RESPIRATORY_RATE'
  | 'OXYGEN_SATURATION'
  | 'VO2_MAX'
  // Sleep
  | 'SLEEP_DURATION'
  | 'DEEP_SLEEP_DURATION'
  | 'REM_SLEEP_DURATION'
  | 'SLEEP_EFFICIENCY'
  | 'SLEEP_SCORE'
  // Activity
  | 'STEPS'
  | 'ACTIVE_CALORIES'
  | 'TOTAL_CALORIES'
  | 'EXERCISE_MINUTES'
  | 'STANDING_HOURS'
  // Recovery & Strain
  | 'RECOVERY_SCORE'
  | 'STRAIN_SCORE'
  | 'READINESS_SCORE'
  // Body Composition
  | 'BODY_FAT_PERCENTAGE'
  | 'MUSCLE_MASS'
  | 'BONE_MASS'
  | 'WATER_PERCENTAGE'
  // Other
  | 'SKIN_TEMPERATURE'
  | 'BLOOD_GLUCOSE'
  | 'STRESS_LEVEL';

export type HealthMetricSource = 'apple_health' | 'fitbit' | 'garmin' | 'oura' | 'whoop' | 'manual';

export interface CreateHealthMetricInput {
  metricType: HealthMetricType;
  value: number;
  unit: string; // "bpm", "ms", "%", "steps", "kcal", etc.
  recordedAt: Date;
  source: HealthMetricSource;
  sourceId?: string;
  metadata?: Record<string, unknown>; // {quality: "high", confidence: 0.95, device: "Apple Watch"}
}

export interface GetHealthMetricsQuery {
  metricType?: HealthMetricType;
  startDate?: Date;
  endDate?: Date;
  source?: HealthMetricSource;
  limit?: number;
}

// Activity Types
export type ActivityType =
  // Cardio
  | 'RUNNING'
  | 'CYCLING'
  | 'SWIMMING'
  | 'WALKING'
  | 'HIKING'
  | 'ROWING'
  | 'ELLIPTICAL'
  // Strength
  | 'WEIGHT_TRAINING'
  | 'BODYWEIGHT'
  | 'CROSSFIT'
  | 'POWERLIFTING'
  // Sports
  | 'BASKETBALL'
  | 'SOCCER'
  | 'TENNIS'
  | 'GOLF'
  // Other
  | 'YOGA'
  | 'PILATES'
  | 'STRETCHING'
  | 'MARTIAL_ARTS'
  | 'DANCE'
  | 'OTHER';

export type ActivityIntensity = 'LOW' | 'MODERATE' | 'HIGH' | 'MAXIMUM';

export type ActivitySource = 'apple_health' | 'strava' | 'garmin' | 'manual';

export interface CreateActivityInput {
  activityType: ActivityType;
  intensity: ActivityIntensity;
  startedAt: Date;
  endedAt: Date;
  duration: number; // in minutes
  caloriesBurned?: number;
  averageHeartRate?: number;
  maxHeartRate?: number;
  distance?: number; // in meters
  steps?: number;
  source: ActivitySource;
  sourceId?: string;
  notes?: string;
}

export interface UpdateActivityInput {
  activityType?: ActivityType;
  intensity?: ActivityIntensity;
  startedAt?: Date;
  endedAt?: Date;
  duration?: number;
  caloriesBurned?: number;
  averageHeartRate?: number;
  maxHeartRate?: number;
  distance?: number;
  steps?: number;
  notes?: string;
}

export interface GetActivitiesQuery {
  activityType?: ActivityType;
  intensity?: ActivityIntensity;
  startDate?: Date;
  endDate?: Date;
  source?: ActivitySource;
  limit?: number;
}

// ============================================================================
// SUPPLEMENT TYPES
// ============================================================================

export type SupplementCategory =
  | 'AMINO_ACID'
  | 'VITAMIN'
  | 'MINERAL'
  | 'PERFORMANCE'
  | 'HERBAL'
  | 'PROTEIN'
  | 'FATTY_ACID'
  | 'PROBIOTIC'
  | 'OTHER';

export type ScheduleType =
  | 'ONE_TIME'
  | 'DAILY'
  | 'DAILY_MULTIPLE'
  | 'WEEKLY'
  | 'INTERVAL';

export type SupplementSource = 'SCHEDULED' | 'MANUAL' | 'QUICK_LOG';

export type DayOfWeek =
  | 'monday'
  | 'tuesday'
  | 'wednesday'
  | 'thursday'
  | 'friday'
  | 'saturday'
  | 'sunday';

export interface WeeklySchedule {
  [key: string]: string[]; // e.g., { "monday": ["08:00", "20:00"], "wednesday": ["08:00"] }
}

export interface CreateUserSupplementInput {
  supplementId: string;
  dosage: string;
  unit: string;
  scheduleType: ScheduleType;
  scheduleTimes?: string[]; // For DAILY_MULTIPLE: ["08:00", "20:00"]
  weeklySchedule?: WeeklySchedule; // For WEEKLY
  intervalDays?: number; // For INTERVAL
  startDate: Date;
  endDate?: Date;
  notes?: string;
}

export interface UpdateUserSupplementInput {
  dosage?: string;
  unit?: string;
  scheduleType?: ScheduleType;
  scheduleTimes?: string[];
  weeklySchedule?: WeeklySchedule;
  intervalDays?: number;
  startDate?: Date;
  endDate?: Date | null;
  isActive?: boolean;
  notes?: string | null;
}

export interface CreateSupplementLogInput {
  userSupplementId?: string;
  supplementId: string;
  dosage: string;
  unit: string;
  takenAt: Date;
  scheduledFor?: Date;
  source: SupplementSource;
  notes?: string;
}

export interface UpdateSupplementLogInput {
  dosage?: string;
  unit?: string;
  takenAt?: Date;
  notes?: string | null;
}

export interface GetSupplementsQuery {
  category?: SupplementCategory;
  search?: string;
}

export interface GetSupplementLogsQuery {
  startDate?: Date;
  endDate?: Date;
  supplementId?: string;
  userSupplementId?: string;
}
