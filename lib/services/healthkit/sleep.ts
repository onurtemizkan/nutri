/**
 * Sleep Metrics Sync
 * Syncs sleep analysis data from HealthKit and calculates sleep metrics
 */

import { Platform } from 'react-native';
import {
  SleepSample,
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
  SleepCategory,
} from '@/lib/types/healthkit';
import { getHealthKit } from './permissions';

/**
 * Query options for react-native-health
 */
interface HealthKitQueryOptions {
  startDate: string;
  endDate: string;
  ascending?: boolean;
  limit?: number;
}

/**
 * Sleep session aggregated from multiple sleep samples
 */
interface SleepSession {
  startDate: Date;
  endDate: Date;
  totalDuration: number; // in hours
  deepSleepDuration: number; // in hours
  remSleepDuration: number; // in hours
  coreSleepDuration: number; // in hours (light sleep)
  awakeDuration: number; // in hours
  inBedDuration: number; // in hours
  sleepEfficiency: number; // percentage
  sourceName?: string;
}

/**
 * Fetch raw sleep samples from HealthKit
 */
export async function fetchSleepSamples(
  options: HealthKitSyncOptions
): Promise<SleepSample[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  const healthKit = await getHealthKit();
  if (!healthKit) {
    return [];
  }

  const queryOptions: HealthKitQueryOptions = {
    startDate: options.startDate.toISOString(),
    endDate: options.endDate.toISOString(),
    ascending: true,
  };

  return new Promise((resolve) => {
    healthKit.getSleepSamples(queryOptions, (error: string | null, results: unknown) => {
      if (error) {
        console.warn('Error fetching sleep samples:', error);
        resolve([]);
        return;
      }

      // Type the raw results from HealthKit
      type RawSleepSample = {
        value: string;
        startDate: string;
        endDate: string;
        sourceName?: string;
        sourceId?: string;
        id?: string;
      };

      // Map the results to our SleepSample type
      const rawResults = (results || []) as RawSleepSample[];
      const samples: SleepSample[] = rawResults.map((sample) => ({
        value: mapSleepValue(sample.value),
        startDate: sample.startDate,
        endDate: sample.endDate,
        sourceName: sample.sourceName,
        sourceId: sample.sourceId,
        id: sample.id,
      }));

      resolve(samples);
    });
  });
}

/**
 * Map HealthKit sleep value string to our SleepCategory
 */
function mapSleepValue(value: string): SleepCategory {
  // HealthKit returns values like 'INBED', 'ASLEEP', 'AWAKE', 'CORE', 'DEEP', 'REM'
  const upperValue = (value || '').toUpperCase();

  if (upperValue.includes('DEEP')) return 'DEEP';
  if (upperValue.includes('REM')) return 'REM';
  if (upperValue.includes('CORE')) return 'CORE';
  if (upperValue.includes('AWAKE')) return 'AWAKE';
  if (upperValue.includes('INBED')) return 'INBED';
  if (upperValue.includes('ASLEEP')) return 'ASLEEP';

  return 'ASLEEP'; // Default fallback
}

/**
 * Calculate duration in hours between two dates
 */
function calculateDurationHours(startDate: string, endDate: string): number {
  const start = new Date(startDate).getTime();
  const end = new Date(endDate).getTime();
  return (end - start) / (1000 * 60 * 60); // Convert ms to hours
}

/**
 * Group sleep samples into sessions (nights)
 * A new session starts if there's a gap of more than 4 hours between samples
 */
function groupIntoSessions(samples: SleepSample[]): SleepSession[] {
  if (samples.length === 0) {
    return [];
  }

  const sessions: SleepSession[] = [];
  let currentSession: SleepSample[] = [];
  const SESSION_GAP_HOURS = 4;

  for (const sample of samples) {
    if (currentSession.length === 0) {
      currentSession.push(sample);
      continue;
    }

    const lastSample = currentSession[currentSession.length - 1];
    const gap = calculateDurationHours(lastSample.endDate, sample.startDate);

    if (gap > SESSION_GAP_HOURS) {
      // Start a new session
      sessions.push(aggregateSession(currentSession));
      currentSession = [sample];
    } else {
      currentSession.push(sample);
    }
  }

  // Don't forget the last session
  if (currentSession.length > 0) {
    sessions.push(aggregateSession(currentSession));
  }

  return sessions;
}

/**
 * Aggregate sleep samples into a single session with metrics
 */
function aggregateSession(samples: SleepSample[]): SleepSession {
  let deepSleepDuration = 0;
  let remSleepDuration = 0;
  let coreSleepDuration = 0;
  let awakeDuration = 0;
  let inBedDuration = 0;
  let asleepDuration = 0;

  for (const sample of samples) {
    const duration = calculateDurationHours(sample.startDate, sample.endDate);

    switch (sample.value) {
      case 'DEEP':
        deepSleepDuration += duration;
        break;
      case 'REM':
        remSleepDuration += duration;
        break;
      case 'CORE':
        coreSleepDuration += duration;
        break;
      case 'AWAKE':
        awakeDuration += duration;
        break;
      case 'INBED':
        inBedDuration += duration;
        break;
      case 'ASLEEP':
        asleepDuration += duration;
        break;
    }
  }

  const firstSample = samples[0];
  const lastSample = samples[samples.length - 1];

  // Total sleep = deep + REM + core + unspecified asleep
  const totalSleep = deepSleepDuration + remSleepDuration + coreSleepDuration + asleepDuration;

  // Total time in bed = all samples
  const totalInBed = inBedDuration + totalSleep + awakeDuration;

  // Sleep efficiency = time asleep / time in bed
  const sleepEfficiency = totalInBed > 0 ? (totalSleep / totalInBed) * 100 : 0;

  return {
    startDate: new Date(firstSample.startDate),
    endDate: new Date(lastSample.endDate),
    totalDuration: totalSleep,
    deepSleepDuration,
    remSleepDuration,
    coreSleepDuration,
    awakeDuration,
    inBedDuration,
    sleepEfficiency,
    sourceName: firstSample.sourceName,
  };
}

/**
 * Convert sleep sessions to ProcessedHealthMetric array
 */
function sessionToHealthMetrics(session: SleepSession): ProcessedHealthMetric[] {
  const baseMetric = {
    source: 'apple_health' as const,
    recordedAt: session.endDate.toISOString(), // Use end of sleep as recorded time
    metadata: {
      sourceName: session.sourceName,
      device: session.sourceName || 'Apple Watch',
      quality: 'high' as const,
      sessionStart: session.startDate.toISOString(),
      sessionEnd: session.endDate.toISOString(),
    },
  };

  const metrics: ProcessedHealthMetric[] = [];

  // Total sleep duration
  if (session.totalDuration > 0) {
    metrics.push({
      ...baseMetric,
      metricType: 'SLEEP_DURATION',
      value: Math.round(session.totalDuration * 100) / 100, // Round to 2 decimals
      unit: METRIC_UNITS.SLEEP_DURATION,
    });
  }

  // Deep sleep duration
  if (session.deepSleepDuration > 0) {
    metrics.push({
      ...baseMetric,
      metricType: 'DEEP_SLEEP_DURATION',
      value: Math.round(session.deepSleepDuration * 100) / 100,
      unit: METRIC_UNITS.DEEP_SLEEP_DURATION,
    });
  }

  // REM sleep duration
  if (session.remSleepDuration > 0) {
    metrics.push({
      ...baseMetric,
      metricType: 'REM_SLEEP_DURATION',
      value: Math.round(session.remSleepDuration * 100) / 100,
      unit: METRIC_UNITS.REM_SLEEP_DURATION,
    });
  }

  // Sleep efficiency
  if (session.sleepEfficiency > 0) {
    metrics.push({
      ...baseMetric,
      metricType: 'SLEEP_EFFICIENCY',
      value: Math.round(session.sleepEfficiency * 10) / 10, // Round to 1 decimal
      unit: METRIC_UNITS.SLEEP_EFFICIENCY,
    });
  }

  return metrics;
}

/**
 * Sync all sleep metrics
 */
export async function syncSleepMetrics(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  const samples = await fetchSleepSamples(options);

  if (samples.length === 0) {
    return [];
  }

  // Group samples into sessions
  const sessions = groupIntoSessions(samples);

  // Convert sessions to health metrics
  const allMetrics: ProcessedHealthMetric[] = [];
  for (const session of sessions) {
    const metrics = sessionToHealthMetrics(session);
    allMetrics.push(...metrics);
  }

  // Sort by recordedAt (newest first)
  allMetrics.sort(
    (a, b) => new Date(b.recordedAt).getTime() - new Date(a.recordedAt).getTime()
  );

  return allMetrics;
}

/**
 * Get sleep summary for a specific date
 */
export async function getSleepSummaryForDate(date: Date): Promise<SleepSession | null> {
  // Query from 6 PM previous day to 12 PM current day to capture overnight sleep
  const startDate = new Date(date);
  startDate.setDate(startDate.getDate() - 1);
  startDate.setHours(18, 0, 0, 0);

  const endDate = new Date(date);
  endDate.setHours(12, 0, 0, 0);

  const samples = await fetchSleepSamples({ startDate, endDate });

  if (samples.length === 0) {
    return null;
  }

  const sessions = groupIntoSessions(samples);

  // Return the longest session (most likely the main night's sleep)
  if (sessions.length === 0) {
    return null;
  }

  return sessions.reduce((longest, current) =>
    current.totalDuration > longest.totalDuration ? current : longest
  );
}
