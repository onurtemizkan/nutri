// Shared types between client and server
export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';
export type ActivityLevel = 'sedentary' | 'light' | 'moderate' | 'active' | 'veryActive';

export type SubscriptionTier = 'FREE' | 'PRO_TRIAL' | 'PRO';
export type BillingCycle = 'MONTHLY' | 'ANNUAL';

export interface User {
  id: string;
  email: string;
  name: string;
  profilePicture?: string | null;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight?: number;
  goalWeight?: number;
  height?: number;
  activityLevel: ActivityLevel;
  createdAt: string;
  // Subscription
  subscriptionTier: SubscriptionTier;
  subscriptionBillingCycle?: BillingCycle;
  subscriptionStartDate?: string;
  subscriptionEndDate?: string;
  subscriptionPrice?: number;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export interface Meal {
  id: string;
  userId: string;
  name: string;
  mealType: MealType;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;

  // Fat breakdown (optional)
  saturatedFat?: number; // grams
  transFat?: number; // grams
  cholesterol?: number; // mg

  // Minerals (optional) - values in mg unless noted
  sodium?: number; // mg
  potassium?: number; // mg
  calcium?: number; // mg
  iron?: number; // mg
  magnesium?: number; // mg
  zinc?: number; // mg
  phosphorus?: number; // mg

  // Vitamins (optional) - values in appropriate units
  vitaminA?: number; // mcg RAE
  vitaminC?: number; // mg
  vitaminD?: number; // mcg
  vitaminE?: number; // mg
  vitaminK?: number; // mcg
  vitaminB6?: number; // mg
  vitaminB12?: number; // mcg
  folate?: number; // mcg DFE
  thiamin?: number; // mg (B1)
  riboflavin?: number; // mg (B2)
  niacin?: number; // mg (B3)

  servingSize?: string;
  notes?: string;
  imageUrl?: string;
  consumedAt: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateMealInput {
  name: string;
  mealType: MealType;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;

  // Fat breakdown (optional)
  saturatedFat?: number;
  transFat?: number;
  cholesterol?: number;

  // Minerals (optional)
  sodium?: number;
  potassium?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  phosphorus?: number;

  // Vitamins (optional)
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
  vitaminE?: number;
  vitaminK?: number;
  vitaminB6?: number;
  vitaminB12?: number;
  folate?: number;
  thiamin?: number;
  riboflavin?: number;
  niacin?: number;

  servingSize?: string;
  notes?: string;
  consumedAt?: string;
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

  // Fat breakdown (optional)
  saturatedFat?: number;
  transFat?: number;
  cholesterol?: number;

  // Minerals (optional)
  sodium?: number;
  potassium?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  phosphorus?: number;

  // Vitamins (optional)
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
  vitaminE?: number;
  vitaminK?: number;
  vitaminB6?: number;
  vitaminB12?: number;
  folate?: number;
  thiamin?: number;
  riboflavin?: number;
  niacin?: number;

  servingSize?: string;
  notes?: string;
  consumedAt?: string;
}

export interface DailySummary {
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  totalFiber: number;
  totalSugar: number;
  mealCount: number;
  goals: {
    goalCalories: number;
    goalProtein: number;
    goalCarbs: number;
    goalFat: number;
  };
  meals: Meal[];
}

export interface WeeklySummary {
  date: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  mealCount: number;
}

// ============================================================================
// SUPPLEMENT TYPES
// ============================================================================

export type SupplementFrequency =
  | 'DAILY'
  | 'TWICE_DAILY'
  | 'THREE_TIMES_DAILY'
  | 'WEEKLY'
  | 'EVERY_OTHER_DAY'
  | 'AS_NEEDED';

export type SupplementTimeOfDay =
  | 'MORNING'
  | 'AFTERNOON'
  | 'EVENING'
  | 'BEFORE_BED'
  | 'WITH_BREAKFAST'
  | 'WITH_LUNCH'
  | 'WITH_DINNER'
  | 'EMPTY_STOMACH';

export interface Supplement {
  id: string;
  userId: string;
  name: string;
  brand?: string;
  dosageAmount: number;
  dosageUnit: string;
  frequency: SupplementFrequency;
  timesPerDay: number;
  timeOfDay: SupplementTimeOfDay[];
  withFood: boolean;
  isActive: boolean;
  startDate: string;
  endDate?: string | null;
  notes?: string;
  color?: string;

  // Micronutrient content (estimated or from barcode)
  // Vitamins
  vitaminA?: number; // mcg RAE
  vitaminC?: number; // mg
  vitaminD?: number; // mcg
  vitaminE?: number; // mg
  vitaminK?: number; // mcg
  vitaminB6?: number; // mg
  vitaminB12?: number; // mcg
  folate?: number; // mcg DFE
  thiamin?: number; // mg (B1)
  riboflavin?: number; // mg (B2)
  niacin?: number; // mg (B3)

  // Minerals
  calcium?: number; // mg
  iron?: number; // mg
  magnesium?: number; // mg
  zinc?: number; // mg
  potassium?: number; // mg
  sodium?: number; // mg
  phosphorus?: number; // mg

  // Special
  omega3?: number; // mg (total EPA + DHA)

  createdAt: string;
  updatedAt: string;
}

export interface SupplementLog {
  id: string;
  userId: string;
  supplementId: string;
  supplement?: Supplement;
  takenAt: string;
  dosageAmount?: number;
  notes?: string;
  skipped: boolean;
  createdAt: string;
}

export interface CreateSupplementInput {
  name: string;
  brand?: string;
  dosageAmount: number;
  dosageUnit: string;
  frequency?: SupplementFrequency;
  timesPerDay?: number;
  timeOfDay?: SupplementTimeOfDay[];
  withFood?: boolean;
  isActive?: boolean;
  startDate?: string;
  endDate?: string | null;
  notes?: string;
  color?: string;

  // Micronutrient content (optional)
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
  vitaminE?: number;
  vitaminK?: number;
  vitaminB6?: number;
  vitaminB12?: number;
  folate?: number;
  thiamin?: number;
  riboflavin?: number;
  niacin?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  potassium?: number;
  sodium?: number;
  phosphorus?: number;
  omega3?: number;
}

export interface UpdateSupplementInput {
  name?: string;
  brand?: string;
  dosageAmount?: number;
  dosageUnit?: string;
  frequency?: SupplementFrequency;
  timesPerDay?: number;
  timeOfDay?: SupplementTimeOfDay[];
  withFood?: boolean;
  isActive?: boolean;
  startDate?: string;
  endDate?: string | null;
  notes?: string;
  color?: string;

  // Micronutrient content (optional)
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
  vitaminE?: number;
  vitaminK?: number;
  vitaminB6?: number;
  vitaminB12?: number;
  folate?: number;
  thiamin?: number;
  riboflavin?: number;
  niacin?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  potassium?: number;
  sodium?: number;
  phosphorus?: number;
  omega3?: number;
}

export interface CreateSupplementLogInput {
  supplementId: string;
  takenAt?: string;
  dosageAmount?: number;
  notes?: string;
  skipped?: boolean;
}

export interface SupplementStatus {
  supplement: Supplement;
  takenCount: number;
  skippedCount: number;
  targetCount: number;
  isComplete: boolean;
  logs: SupplementLog[];
}

export interface TodaySupplementStatus {
  date: string;
  totalSupplements: number;
  completedSupplements: number;
  completionRate: number;
  supplements: SupplementStatus[];
}

// Re-export notification types
export * from './notifications';

// ============================================================================
// WATER INTAKE TYPES
// ============================================================================

export interface WaterIntake {
  id: string;
  userId: string;
  amount: number; // in ml
  recordedAt: string;
  createdAt: string;
}

export interface CreateWaterIntakeInput {
  amount: number;
  recordedAt?: string;
}

export interface UpdateWaterIntakeInput {
  amount?: number;
  recordedAt?: string;
}

export interface QuickAddWaterInput {
  preset: 'glass' | 'bottle' | 'cup' | 'custom';
  customAmount?: number;
}

export interface WaterDailySummary {
  date: string;
  totalAmount: number;
  goalAmount: number;
  percentageComplete: number;
  intakeCount: number;
  intakes: {
    id: string;
    amount: number;
    recordedAt: string;
  }[];
}

export interface WaterWeeklySummary {
  startDate: string;
  endDate: string;
  dailyAverage: number;
  totalAmount: number;
  goalAmount: number;
  daysMetGoal: number;
  dailySummaries: {
    date: string;
    totalAmount: number;
    goalAmount: number;
    percentageComplete: number;
  }[];
}

export interface WaterGoal {
  goalWater: number;
}

// Water preset amounts in ml
export const WATER_PRESETS = {
  glass: 250, // Standard glass
  bottle: 500, // Standard bottle
  cup: 200, // Coffee cup
} as const;
