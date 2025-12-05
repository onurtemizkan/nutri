/**
 * HealthKit Integration Types
 * Maps HealthKit data types to our backend HealthMetric schema
 */

/**
 * HealthKit metric type identifiers
 * These match the react-native-health API
 */
export type HealthKitIdentifier =
  // Cardiovascular
  | 'HeartRate'
  | 'RestingHeartRate'
  | 'HeartRateVariabilitySDNN'
  | 'WalkingHeartRateAverage'
  // Respiratory
  | 'RespiratoryRate'
  | 'OxygenSaturation'
  | 'Vo2Max'
  // Sleep
  | 'SleepAnalysis'
  // Activity
  | 'StepCount'
  | 'ActiveEnergyBurned'
  | 'BasalEnergyBurned'
  | 'DistanceWalkingRunning'
  | 'FlightsClimbed';

/**
 * Our backend HealthMetricType enum values
 * Must match server/prisma/schema.prisma HealthMetricType enum
 */
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

/**
 * Mapping from HealthKit identifiers to our backend metric types
 */
export const HEALTHKIT_TO_METRIC_TYPE: Record<string, HealthMetricType> = {
  RestingHeartRate: 'RESTING_HEART_RATE',
  HeartRateVariabilitySDNN: 'HEART_RATE_VARIABILITY_SDNN',
  RespiratoryRate: 'RESPIRATORY_RATE',
  OxygenSaturation: 'OXYGEN_SATURATION',
  Vo2Max: 'VO2_MAX',
  StepCount: 'STEPS',
  ActiveEnergyBurned: 'ACTIVE_CALORIES',
} as const;

/**
 * Units for each metric type
 */
export const METRIC_UNITS: Record<HealthMetricType, string> = {
  RESTING_HEART_RATE: 'bpm',
  HEART_RATE_VARIABILITY_SDNN: 'ms',
  HEART_RATE_VARIABILITY_RMSSD: 'ms',
  BLOOD_PRESSURE_SYSTOLIC: 'mmHg',
  BLOOD_PRESSURE_DIASTOLIC: 'mmHg',
  RESPIRATORY_RATE: 'breaths/min',
  OXYGEN_SATURATION: '%',
  VO2_MAX: 'mL/kg/min',
  SLEEP_DURATION: 'hours',
  DEEP_SLEEP_DURATION: 'hours',
  REM_SLEEP_DURATION: 'hours',
  SLEEP_EFFICIENCY: '%',
  SLEEP_SCORE: 'score',
  STEPS: 'steps',
  ACTIVE_CALORIES: 'kcal',
  TOTAL_CALORIES: 'kcal',
  EXERCISE_MINUTES: 'minutes',
  STANDING_HOURS: 'hours',
  RECOVERY_SCORE: 'score',
  STRAIN_SCORE: 'score',
  READINESS_SCORE: 'score',
  BODY_FAT_PERCENTAGE: '%',
  MUSCLE_MASS: 'kg',
  BONE_MASS: 'kg',
  WATER_PERCENTAGE: '%',
  SKIN_TEMPERATURE: 'C',
  BLOOD_GLUCOSE: 'mg/dL',
  STRESS_LEVEL: 'score',
};

/**
 * HealthKit sample result from react-native-health
 */
export interface HealthKitSample {
  value: number;
  startDate: string;
  endDate: string;
  sourceName?: string;
  sourceId?: string;
  id?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Sleep sample with category value
 */
export interface SleepSample {
  value: SleepCategory;
  startDate: string;
  endDate: string;
  sourceName?: string;
  sourceId?: string;
  id?: string;
}

/**
 * Sleep category values from HealthKit
 */
export type SleepCategory =
  | 'INBED'
  | 'ASLEEP'
  | 'AWAKE'
  | 'CORE' // Light sleep
  | 'DEEP'
  | 'REM';

/**
 * Processed health metric ready for API upload
 */
export interface ProcessedHealthMetric {
  metricType: HealthMetricType;
  value: number;
  unit: string;
  recordedAt: string; // ISO date string
  source: 'apple_health';
  sourceId?: string;
  metadata?: {
    device?: string;
    quality?: 'high' | 'medium' | 'low';
    sourceName?: string;
    [key: string]: unknown;
  };
}

/**
 * HealthKit sync options
 */
export interface HealthKitSyncOptions {
  startDate: Date;
  endDate: Date;
  /**
   * Whether to include detailed metadata
   */
  includeMetadata?: boolean;
}

/**
 * Sync result for a single metric category
 */
export interface SyncResult {
  success: boolean;
  metricsCount: number;
  errors: string[];
  lastSyncedDate?: Date;
}

/**
 * Overall sync status
 */
export interface SyncStatus {
  isAvailable: boolean;
  isAuthorized: boolean;
  lastSync?: {
    cardiovascular?: Date;
    respiratory?: Date;
    sleep?: Date;
    activity?: Date;
  };
  syncInProgress: boolean;
  error?: string;
}

/**
 * Permission status for HealthKit
 */
export interface HealthKitPermissions {
  read: {
    heartRate: boolean;
    restingHeartRate: boolean;
    heartRateVariability: boolean;
    respiratoryRate: boolean;
    oxygenSaturation: boolean;
    vo2Max: boolean;
    sleepAnalysis: boolean;
    stepCount: boolean;
    activeEnergy: boolean;
  };
  write: {
    // We primarily read, but may write nutrition data in future
  };
}

/**
 * Permission request result
 */
export interface PermissionRequestResult {
  success: boolean;
  granted: Partial<HealthKitPermissions['read']>;
  denied: string[];
  error?: string;
}

/**
 * HealthKit read permission types for react-native-health
 */
export const HEALTHKIT_READ_PERMISSIONS = [
  'HeartRate',
  'RestingHeartRate',
  'HeartRateVariabilitySDNN',
  'RespiratoryRate',
  'OxygenSaturation',
  'Vo2Max',
  'SleepAnalysis',
  'StepCount',
  'ActiveEnergyBurned',
  'BasalEnergyBurned',
  'DistanceWalkingRunning',
  'FlightsClimbed',
] as const;

/**
 * HealthKit write permission types for react-native-health
 */
export const HEALTHKIT_WRITE_PERMISSIONS: string[] = [
  // We don't write to HealthKit for now
];

/**
 * Sync timestamp storage keys
 */
export const SYNC_TIMESTAMP_KEYS = {
  CARDIOVASCULAR: 'healthkit_last_sync_cardiovascular',
  RESPIRATORY: 'healthkit_last_sync_respiratory',
  SLEEP: 'healthkit_last_sync_sleep',
  ACTIVITY: 'healthkit_last_sync_activity',
} as const;

/**
 * API request for bulk health metrics
 */
export interface BulkHealthMetricsRequest {
  metrics: ProcessedHealthMetric[];
}

/**
 * API response for bulk health metrics
 */
export interface BulkHealthMetricsResponse {
  created: number;
  updated: number;
  errors: {
    index: number;
    error: string;
  }[];
}
