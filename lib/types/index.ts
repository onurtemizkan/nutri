// Shared types between client and server
export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';
export type ActivityLevel =
  | 'sedentary'
  | 'light'
  | 'moderate'
  | 'active'
  | 'veryActive';

export interface User {
  id: string;
  email: string;
  name: string;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight?: number;
  goalWeight?: number;
  height?: number;
  activityLevel: ActivityLevel;
  createdAt: string;
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
