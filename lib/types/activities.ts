/**
 * Activity Types
 * TypeScript types for activities matching the backend API contracts
 */

/**
 * Activity Type enum - matches server/src/validation/schemas.ts activityTypeSchema
 * Organized by category: Cardio, Strength, Sports, Other
 */
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

/**
 * Activity Intensity levels - matches server/src/validation/schemas.ts activityIntensitySchema
 */
export type ActivityIntensity = 'LOW' | 'MODERATE' | 'HIGH' | 'MAXIMUM';

/**
 * Activity Source - matches server/src/validation/schemas.ts activitySourceSchema
 */
export type ActivitySource = 'apple_health' | 'strava' | 'garmin' | 'manual';

/**
 * Activity entity returned from API
 */
export interface Activity {
  id: string;
  userId: string;
  startedAt: string;
  endedAt: string;
  duration: number; // minutes
  activityType: ActivityType;
  intensity: ActivityIntensity;
  caloriesBurned?: number;
  averageHeartRate?: number;
  maxHeartRate?: number;
  distance?: number; // meters
  steps?: number;
  source: ActivitySource;
  sourceId?: string;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

/**
 * Input for creating a new activity
 */
export interface CreateActivityInput {
  activityType: ActivityType;
  intensity: ActivityIntensity;
  startedAt: string; // ISO 8601 datetime string
  endedAt: string; // ISO 8601 datetime string
  duration: number; // minutes
  caloriesBurned?: number;
  averageHeartRate?: number;
  maxHeartRate?: number;
  distance?: number; // meters
  steps?: number;
  source: ActivitySource;
  sourceId?: string;
  notes?: string;
}

/**
 * Input for updating an existing activity
 */
export interface UpdateActivityInput {
  activityType?: ActivityType;
  intensity?: ActivityIntensity;
  startedAt?: string;
  endedAt?: string;
  duration?: number;
  caloriesBurned?: number;
  averageHeartRate?: number;
  maxHeartRate?: number;
  distance?: number;
  steps?: number;
  notes?: string;
}

/**
 * Weekly summary for activities
 */
export interface WeeklySummary {
  totalMinutes: number;
  totalCalories: number;
  workoutCount: number;
  averageIntensity: string;
  byType: Record<ActivityType, { count: number; minutes: number; calories: number }>;
}

/**
 * Daily summary for activities
 */
export interface DailySummary {
  totalMinutes: number;
  totalCalories: number;
  workoutCount: number;
  activities: Activity[];
}

/**
 * Activity statistics by type
 */
export interface ActivityStats {
  count: number;
  totalMinutes: number;
  totalCalories: number;
  averageDuration: number;
  averageCalories: number;
  longestDuration: number;
  mostRecentDate: string;
}

/**
 * Activity category for grouping types
 */
export type ActivityCategory = 'cardio' | 'strength' | 'sports' | 'other' | 'all';

/**
 * Configuration for each activity type
 */
export interface ActivityTypeConfig {
  displayName: string;
  shortName: string;
  icon: string; // Ionicons name
  category: ActivityCategory;
  color: string;
  hasDistance: boolean;
  hasSteps: boolean;
  defaultCaloriesPerMinute: number;
}

/**
 * Activity type configuration constant
 */
export const ACTIVITY_TYPE_CONFIG: Record<ActivityType, ActivityTypeConfig> = {
  // Cardio
  RUNNING: {
    displayName: 'Running',
    shortName: 'Run',
    icon: 'walk-outline',
    category: 'cardio',
    color: '#EF4444',
    hasDistance: true,
    hasSteps: true,
    defaultCaloriesPerMinute: 10,
  },
  CYCLING: {
    displayName: 'Cycling',
    shortName: 'Bike',
    icon: 'bicycle-outline',
    category: 'cardio',
    color: '#F59E0B',
    hasDistance: true,
    hasSteps: false,
    defaultCaloriesPerMinute: 8,
  },
  SWIMMING: {
    displayName: 'Swimming',
    shortName: 'Swim',
    icon: 'water-outline',
    category: 'cardio',
    color: '#3B82F6',
    hasDistance: true,
    hasSteps: false,
    defaultCaloriesPerMinute: 9,
  },
  WALKING: {
    displayName: 'Walking',
    shortName: 'Walk',
    icon: 'footsteps-outline',
    category: 'cardio',
    color: '#10B981',
    hasDistance: true,
    hasSteps: true,
    defaultCaloriesPerMinute: 4,
  },
  HIKING: {
    displayName: 'Hiking',
    shortName: 'Hike',
    icon: 'trail-sign-outline',
    category: 'cardio',
    color: '#84CC16',
    hasDistance: true,
    hasSteps: true,
    defaultCaloriesPerMinute: 6,
  },
  ROWING: {
    displayName: 'Rowing',
    shortName: 'Row',
    icon: 'boat-outline',
    category: 'cardio',
    color: '#06B6D4',
    hasDistance: true,
    hasSteps: false,
    defaultCaloriesPerMinute: 8,
  },
  ELLIPTICAL: {
    displayName: 'Elliptical',
    shortName: 'Ellip',
    icon: 'fitness-outline',
    category: 'cardio',
    color: '#8B5CF6',
    hasDistance: false,
    hasSteps: true,
    defaultCaloriesPerMinute: 7,
  },
  // Strength
  WEIGHT_TRAINING: {
    displayName: 'Weight Training',
    shortName: 'Weights',
    icon: 'barbell-outline',
    category: 'strength',
    color: '#EC4899',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 5,
  },
  BODYWEIGHT: {
    displayName: 'Bodyweight',
    shortName: 'BW',
    icon: 'body-outline',
    category: 'strength',
    color: '#F97316',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 6,
  },
  CROSSFIT: {
    displayName: 'CrossFit',
    shortName: 'CF',
    icon: 'flash-outline',
    category: 'strength',
    color: '#DC2626',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 12,
  },
  POWERLIFTING: {
    displayName: 'Powerlifting',
    shortName: 'PL',
    icon: 'barbell-outline',
    category: 'strength',
    color: '#7C3AED',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 4,
  },
  // Sports
  BASKETBALL: {
    displayName: 'Basketball',
    shortName: 'BBall',
    icon: 'basketball-outline',
    category: 'sports',
    color: '#F97316',
    hasDistance: false,
    hasSteps: true,
    defaultCaloriesPerMinute: 8,
  },
  SOCCER: {
    displayName: 'Soccer',
    shortName: 'Soccer',
    icon: 'football-outline',
    category: 'sports',
    color: '#22C55E',
    hasDistance: true,
    hasSteps: true,
    defaultCaloriesPerMinute: 9,
  },
  TENNIS: {
    displayName: 'Tennis',
    shortName: 'Tennis',
    icon: 'tennisball-outline',
    category: 'sports',
    color: '#FBBF24',
    hasDistance: false,
    hasSteps: true,
    defaultCaloriesPerMinute: 7,
  },
  GOLF: {
    displayName: 'Golf',
    shortName: 'Golf',
    icon: 'golf-outline',
    category: 'sports',
    color: '#059669',
    hasDistance: true,
    hasSteps: true,
    defaultCaloriesPerMinute: 4,
  },
  // Other
  YOGA: {
    displayName: 'Yoga',
    shortName: 'Yoga',
    icon: 'leaf-outline',
    category: 'other',
    color: '#14B8A6',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 3,
  },
  PILATES: {
    displayName: 'Pilates',
    shortName: 'Pilates',
    icon: 'body-outline',
    category: 'other',
    color: '#A855F7',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 4,
  },
  STRETCHING: {
    displayName: 'Stretching',
    shortName: 'Stretch',
    icon: 'expand-outline',
    category: 'other',
    color: '#64748B',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 2,
  },
  MARTIAL_ARTS: {
    displayName: 'Martial Arts',
    shortName: 'MA',
    icon: 'hand-left-outline',
    category: 'other',
    color: '#0EA5E9',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 10,
  },
  DANCE: {
    displayName: 'Dance',
    shortName: 'Dance',
    icon: 'musical-notes-outline',
    category: 'other',
    color: '#D946EF',
    hasDistance: false,
    hasSteps: true,
    defaultCaloriesPerMinute: 6,
  },
  OTHER: {
    displayName: 'Other',
    shortName: 'Other',
    icon: 'ellipsis-horizontal-outline',
    category: 'other',
    color: '#6B7280',
    hasDistance: false,
    hasSteps: false,
    defaultCaloriesPerMinute: 5,
  },
};

/**
 * Intensity level configuration
 */
export interface IntensityConfig {
  displayName: string;
  shortName: string;
  color: string;
  icon: string;
  multiplier: number; // calorie multiplier
}

/**
 * Intensity configuration constant
 */
export const INTENSITY_CONFIG: Record<ActivityIntensity, IntensityConfig> = {
  LOW: {
    displayName: 'Low',
    shortName: 'Low',
    color: '#22C55E',
    icon: 'leaf-outline',
    multiplier: 0.8,
  },
  MODERATE: {
    displayName: 'Moderate',
    shortName: 'Mod',
    color: '#F59E0B',
    icon: 'trending-up-outline',
    multiplier: 1.0,
  },
  HIGH: {
    displayName: 'High',
    shortName: 'High',
    color: '#F97316',
    icon: 'flame-outline',
    multiplier: 1.3,
  },
  MAXIMUM: {
    displayName: 'Maximum',
    shortName: 'Max',
    color: '#EF4444',
    icon: 'flash-outline',
    multiplier: 1.5,
  },
};

/**
 * Source configuration
 */
export const SOURCE_CONFIG: Record<ActivitySource, { displayName: string; icon: string }> = {
  apple_health: { displayName: 'Apple Health', icon: 'logo-apple' },
  strava: { displayName: 'Strava', icon: 'bicycle-outline' },
  garmin: { displayName: 'Garmin', icon: 'watch-outline' },
  manual: { displayName: 'Manual Entry', icon: 'create-outline' },
};

/**
 * All activity types as an array for iteration
 */
export const ACTIVITY_TYPES: ActivityType[] = [
  // Cardio
  'RUNNING',
  'CYCLING',
  'SWIMMING',
  'WALKING',
  'HIKING',
  'ROWING',
  'ELLIPTICAL',
  // Strength
  'WEIGHT_TRAINING',
  'BODYWEIGHT',
  'CROSSFIT',
  'POWERLIFTING',
  // Sports
  'BASKETBALL',
  'SOCCER',
  'TENNIS',
  'GOLF',
  // Other
  'YOGA',
  'PILATES',
  'STRETCHING',
  'MARTIAL_ARTS',
  'DANCE',
  'OTHER',
];

/**
 * Intensity levels as an array for iteration
 */
export const INTENSITY_LEVELS: ActivityIntensity[] = ['LOW', 'MODERATE', 'HIGH', 'MAXIMUM'];

/**
 * Activity sources as an array for iteration
 */
export const ACTIVITY_SOURCES: ActivitySource[] = ['apple_health', 'strava', 'garmin', 'manual'];

/**
 * Get activities grouped by category
 */
export function getActivitiesByCategory(): Record<ActivityCategory, ActivityType[]> {
  const grouped: Record<ActivityCategory, ActivityType[]> = {
    cardio: [],
    strength: [],
    sports: [],
    other: [],
    all: [...ACTIVITY_TYPES],
  };

  for (const activityType of ACTIVITY_TYPES) {
    const config = ACTIVITY_TYPE_CONFIG[activityType];
    if (config.category !== 'all') {
      grouped[config.category].push(activityType);
    }
  }

  return grouped;
}

/**
 * Category display names
 */
export const CATEGORY_DISPLAY_NAMES: Record<ActivityCategory, string> = {
  cardio: 'Cardio',
  strength: 'Strength',
  sports: 'Sports',
  other: 'Other',
  all: 'All Activities',
};

/**
 * Format duration in minutes to a readable string
 */
export function formatDuration(minutes: number): string {
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  if (mins === 0) {
    return `${hours}h`;
  }
  return `${hours}h ${mins}m`;
}

/**
 * Format distance in meters to a readable string
 */
export function formatDistance(meters: number): string {
  if (meters < 1000) {
    return `${Math.round(meters)}m`;
  }
  const km = meters / 1000;
  return `${km.toFixed(2)} km`;
}

/**
 * Estimate calories burned based on activity type, intensity, and duration
 */
export function estimateCalories(
  activityType: ActivityType,
  intensity: ActivityIntensity,
  durationMinutes: number
): number {
  const typeConfig = ACTIVITY_TYPE_CONFIG[activityType];
  const intensityConfig = INTENSITY_CONFIG[intensity];
  return Math.round(
    typeConfig.defaultCaloriesPerMinute * intensityConfig.multiplier * durationMinutes
  );
}
