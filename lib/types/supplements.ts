/**
 * Supplement Types for Mobile App
 *
 * Mirrors the backend API types for type-safe client-server communication.
 */

// =============================================================================
// ENUMS
// =============================================================================

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

// =============================================================================
// MASTER SUPPLEMENT (from database)
// =============================================================================

export interface Supplement {
  id: string;
  name: string;
  category: SupplementCategory;
  description?: string;
  defaultDosage?: string;
  defaultUnit?: string;
  metadata?: SupplementMetadata;
  createdAt: string;
  updatedAt: string;
}

export interface SupplementMetadata {
  benefits?: string[];
  timing?: string;
  warnings?: string[];
  interactions?: string[];
}

// =============================================================================
// USER SUPPLEMENT (Schedule)
// =============================================================================

export interface WeeklySchedule {
  [key: string]: string[]; // e.g., { "monday": ["08:00", "20:00"], "wednesday": ["08:00"] }
}

export interface UserSupplement {
  id: string;
  userId: string;
  supplementId: string;
  supplement: Supplement;
  dosage: string;
  unit: string;
  scheduleType: ScheduleType;
  scheduleTimes?: string[]; // For DAILY_MULTIPLE: ["08:00", "14:00", "20:00"]
  weeklySchedule?: WeeklySchedule; // For WEEKLY
  intervalDays?: number; // For INTERVAL
  startDate: string;
  endDate?: string;
  isActive: boolean;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateUserSupplementInput {
  supplementId: string;
  dosage: string;
  unit: string;
  scheduleType: ScheduleType;
  scheduleTimes?: string[];
  weeklySchedule?: WeeklySchedule;
  intervalDays?: number;
  startDate: string;
  endDate?: string;
  notes?: string;
}

export interface UpdateUserSupplementInput {
  dosage?: string;
  unit?: string;
  scheduleType?: ScheduleType;
  scheduleTimes?: string[];
  weeklySchedule?: WeeklySchedule;
  intervalDays?: number;
  startDate?: string;
  endDate?: string | null;
  isActive?: boolean;
  notes?: string | null;
}

// =============================================================================
// SUPPLEMENT LOG (Intake Tracking)
// =============================================================================

export interface SupplementLog {
  id: string;
  userId: string;
  userSupplementId?: string;
  userSupplement?: UserSupplement;
  supplementId: string;
  supplement: Supplement;
  dosage: string;
  unit: string;
  takenAt: string;
  scheduledFor?: string;
  source: SupplementSource;
  notes?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateSupplementLogInput {
  userSupplementId?: string;
  supplementId: string;
  dosage: string;
  unit: string;
  takenAt: string;
  scheduledFor?: string;
  source: SupplementSource;
  notes?: string;
}

export interface UpdateSupplementLogInput {
  dosage?: string;
  unit?: string;
  takenAt?: string;
  notes?: string | null;
}

// =============================================================================
// SCHEDULED SUPPLEMENTS (for today's view)
// =============================================================================

export interface ScheduledSupplement {
  userSupplement: UserSupplement;
  scheduledTimes: string[];
  takenCount: number;
  taken: boolean;
  logs: SupplementLog[];
}

// =============================================================================
// SUMMARIES AND ANALYTICS
// =============================================================================

export interface SupplementDailySummary {
  date: string;
  scheduledCount: number;
  takenCount: number;
  adherencePercentage: number;
  scheduled: ScheduledSupplement[];
  logs: SupplementLog[];
}

export interface SupplementWeeklySummary {
  days: {
    date: string;
    scheduledCount: number;
    takenCount: number;
    adherencePercentage: number;
  }[];
  totalScheduled: number;
  totalTaken: number;
  averageAdherence: number;
}

export interface SupplementStats {
  supplementId: string;
  days: number;
  totalLogs: number;
  averagePerDay: number;
  lastTaken: string;
  logs: SupplementLog[];
}

// =============================================================================
// QUERY PARAMETERS
// =============================================================================

export interface GetSupplementsQuery {
  category?: SupplementCategory;
  search?: string;
}

export interface GetSupplementLogsQuery {
  startDate?: string;
  endDate?: string;
  supplementId?: string;
  userSupplementId?: string;
}

export interface GetScheduledSupplementsQuery {
  date?: string;
}

// =============================================================================
// HELPER CONSTANTS
// =============================================================================

export const SUPPLEMENT_CATEGORIES: { value: SupplementCategory; label: string }[] = [
  { value: 'AMINO_ACID', label: 'Amino Acids' },
  { value: 'VITAMIN', label: 'Vitamins' },
  { value: 'MINERAL', label: 'Minerals' },
  { value: 'PERFORMANCE', label: 'Performance' },
  { value: 'HERBAL', label: 'Herbal' },
  { value: 'PROTEIN', label: 'Protein' },
  { value: 'FATTY_ACID', label: 'Fatty Acids' },
  { value: 'PROBIOTIC', label: 'Probiotics' },
  { value: 'OTHER', label: 'Other' },
];

export const SCHEDULE_TYPES: { value: ScheduleType; label: string; description: string }[] = [
  { value: 'ONE_TIME', label: 'One Time', description: 'Log once, no schedule' },
  { value: 'DAILY', label: 'Daily', description: 'Once per day' },
  { value: 'DAILY_MULTIPLE', label: 'Multiple Times Daily', description: 'Multiple times per day at specific times' },
  { value: 'WEEKLY', label: 'Weekly', description: 'Specific days of the week' },
  { value: 'INTERVAL', label: 'Every N Days', description: 'Every few days (e.g., every 3 days)' },
];

export const DAYS_OF_WEEK: { value: DayOfWeek; label: string; short: string }[] = [
  { value: 'monday', label: 'Monday', short: 'Mon' },
  { value: 'tuesday', label: 'Tuesday', short: 'Tue' },
  { value: 'wednesday', label: 'Wednesday', short: 'Wed' },
  { value: 'thursday', label: 'Thursday', short: 'Thu' },
  { value: 'friday', label: 'Friday', short: 'Fri' },
  { value: 'saturday', label: 'Saturday', short: 'Sat' },
  { value: 'sunday', label: 'Sunday', short: 'Sun' },
];

export const COMMON_UNITS = [
  'mg',
  'g',
  'mcg',
  'IU',
  'ml',
  'capsule',
  'capsules',
  'tablet',
  'tablets',
  'scoop',
  'scoops',
  'serving',
  'servings',
];
