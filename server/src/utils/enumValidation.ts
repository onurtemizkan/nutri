import { HealthMetricType, ActivityType, ActivityIntensity } from '@prisma/client';

/**
 * Type-safe enum validation utilities
 * Replaces unsafe `as any` type assertions with proper runtime validation
 */

const HEALTH_METRIC_SOURCE_VALUES = ['apple_health', 'fitbit', 'garmin', 'oura', 'whoop', 'manual'] as const;
export type HealthMetricSource = typeof HEALTH_METRIC_SOURCE_VALUES[number];

const ACTIVITY_SOURCE_VALUES = ['apple_health', 'strava', 'garmin', 'manual'] as const;
export type ActivitySource = typeof ACTIVITY_SOURCE_VALUES[number];

/**
 * Validates and parses a HealthMetricType from a string
 * @throws Error if value is not a valid HealthMetricType
 */
export function parseHealthMetricType(value: unknown): HealthMetricType {
  if (typeof value !== 'string') {
    throw new Error('Health metric type must be a string');
  }

  const upperValue = value.toUpperCase();
  const validValues = Object.values(HealthMetricType);

  if (!validValues.includes(upperValue as HealthMetricType)) {
    throw new Error(
      `Invalid health metric type: ${value}. Must be one of: ${validValues.join(', ')}`
    );
  }

  return upperValue as HealthMetricType;
}

/**
 * Validates and parses an ActivityType from a string
 * @throws Error if value is not a valid ActivityType
 */
export function parseActivityType(value: unknown): ActivityType {
  if (typeof value !== 'string') {
    throw new Error('Activity type must be a string');
  }

  const upperValue = value.toUpperCase();
  const validValues = Object.values(ActivityType);

  if (!validValues.includes(upperValue as ActivityType)) {
    throw new Error(
      `Invalid activity type: ${value}. Must be one of: ${validValues.join(', ')}`
    );
  }

  return upperValue as ActivityType;
}

/**
 * Validates and parses an ActivityIntensity from a string
 * @throws Error if value is not a valid ActivityIntensity
 */
export function parseActivityIntensity(value: unknown): ActivityIntensity {
  if (typeof value !== 'string') {
    throw new Error('Activity intensity must be a string');
  }

  const upperValue = value.toUpperCase();
  const validValues = Object.values(ActivityIntensity);

  if (!validValues.includes(upperValue as ActivityIntensity)) {
    throw new Error(
      `Invalid activity intensity: ${value}. Must be one of: ${validValues.join(', ')}`
    );
  }

  return upperValue as ActivityIntensity;
}

/**
 * Validates a HealthMetricSource from a string
 * @throws Error if value is not a valid source
 */
export function parseHealthMetricSource(value: unknown): HealthMetricSource {
  if (typeof value !== 'string') {
    throw new Error('Health metric source must be a string');
  }

  const lowerValue = value.toLowerCase();

  if (!HEALTH_METRIC_SOURCE_VALUES.includes(lowerValue as HealthMetricSource)) {
    throw new Error(
      `Invalid health metric source: ${value}. Must be one of: ${HEALTH_METRIC_SOURCE_VALUES.join(', ')}`
    );
  }

  return lowerValue as HealthMetricSource;
}

/**
 * Validates an ActivitySource from a string
 * @throws Error if value is not a valid source
 */
export function parseActivitySource(value: unknown): ActivitySource {
  if (typeof value !== 'string') {
    throw new Error('Activity source must be a string');
  }

  const lowerValue = value.toLowerCase();

  if (!ACTIVITY_SOURCE_VALUES.includes(lowerValue as ActivitySource)) {
    throw new Error(
      `Invalid activity source: ${value}. Must be one of: ${ACTIVITY_SOURCE_VALUES.join(', ')}`
    );
  }

  return lowerValue as ActivitySource;
}

/**
 * Safe optional enum parser - returns undefined if value is null/undefined
 */
export function parseOptionalHealthMetricType(value: unknown): HealthMetricType | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined;
  }
  return parseHealthMetricType(value);
}

export function parseOptionalActivityType(value: unknown): ActivityType | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined;
  }
  return parseActivityType(value);
}

export function parseOptionalActivityIntensity(value: unknown): ActivityIntensity | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined;
  }
  return parseActivityIntensity(value);
}

export function parseOptionalHealthMetricSource(value: unknown): HealthMetricSource | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined;
  }
  return parseHealthMetricSource(value);
}

export function parseOptionalActivitySource(value: unknown): ActivitySource | undefined {
  if (value === null || value === undefined || value === '') {
    return undefined;
  }
  return parseActivitySource(value);
}
