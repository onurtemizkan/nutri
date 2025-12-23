# Task ID: 43

**Title:** Weight Tracking Mobile UI with Progress Visualization and Goal Tracking

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement complete weight tracking functionality including mobile UI screens, backend API endpoints, weight trend visualization, goal progress tracking, and body composition integration. The WeightRecord Prisma model exists but has no API or UI. User model has currentWeight and goalWeight fields that need integration.

**Details:**

## Current State Analysis

### Existing Infrastructure:
- **Prisma Model** (`server/prisma/schema.prisma` lines 132-142):
  ```prisma
  model WeightRecord {
    id        String   @id @default(cuid())
    userId    String
    user      User     @relation(fields: [userId], references: [id], onDelete: Cascade)
    weight    Float    // in kg
    recordedAt DateTime @default(now())
    createdAt DateTime @default(now())
    @@index([userId, recordedAt])
  }
  ```
- **User Model Fields**:
  - `currentWeight Float?` - Current weight
  - `goalWeight Float?` - Target weight
  - `height Float?` - For BMI calculation
  - `weightRecords WeightRecord[]` - Relation
- **NO backend API endpoints** for weight tracking
- **NO mobile UI screens** for weight tracking

### Required Schema Updates

Add to WeightRecord model:
```prisma
model WeightRecord {
  // ... existing fields ...
  
  // Optional body composition data (from smart scales)
  bodyFatPercentage  Float?
  muscleMass         Float?  // kg
  boneMass           Float?  // kg
  waterPercentage    Float?
  visceralFat        Float?  // level 1-60
  
  // Source tracking
  source             String  @default("manual") // "manual", "apple_health", "withings", "fitbit"
  sourceId           String?
  
  // Notes
  notes              String?
  
  @@unique([userId, recordedAt, source]) // Prevent duplicates from same source
}
```

Add to User model:
```prisma
// Weight preferences
weightUnit          String   @default("kg")  // "kg" or "lb"
weightGoalType      WeightGoalType @default(MAINTAIN)
weightGoalPace      Float?   // kg per week (0.25, 0.5, 0.75, 1.0)
weightGoalStartDate DateTime?
weightGoalStartWeight Float?
```

Add enum:
```prisma
enum WeightGoalType {
  LOSE
  GAIN
  MAINTAIN
}
```

## Backend API Implementation

### 1. Weight Service (`server/src/services/weightService.ts`)

```typescript
interface WeightRecordInput {
  weight: number;           // kg
  recordedAt?: Date;
  bodyFatPercentage?: number;
  muscleMass?: number;
  boneMass?: number;
  waterPercentage?: number;
  visceralFat?: number;
  source?: string;
  notes?: string;
}

interface WeightTrend {
  date: string;
  weight: number;
  movingAverage7d: number;
  movingAverage30d: number;
}

interface WeightProgress {
  currentWeight: number;
  goalWeight: number;
  startWeight: number;
  weightLost: number;        // or gained if positive
  percentComplete: number;
  projectedGoalDate: Date | null;
  weeklyChange: number;
  monthlyChange: number;
  trend: 'losing' | 'gaining' | 'maintaining';
}

class WeightService {
  // CRUD operations
  async createWeightRecord(userId: string, data: WeightRecordInput): Promise<WeightRecord>;
  async getWeightRecords(userId: string, startDate: Date, endDate: Date): Promise<WeightRecord[]>;
  async getLatestWeight(userId: string): Promise<WeightRecord | null>;
  async updateWeightRecord(userId: string, recordId: string, data: Partial<WeightRecordInput>): Promise<WeightRecord>;
  async deleteWeightRecord(userId: string, recordId: string): Promise<void>;
  
  // Analytics
  async getWeightTrends(userId: string, period: 'month' | '3months' | '6months' | 'year' | 'all'): Promise<WeightTrend[]>;
  async getWeightProgress(userId: string): Promise<WeightProgress>;
  async getWeightStats(userId: string): Promise<WeightStats>;
  
  // Goal management
  async setWeightGoal(userId: string, goalWeight: number, goalType: WeightGoalType, pace?: number): Promise<User>;
  async updateCurrentWeight(userId: string, weight: number): Promise<User>;
  
  // Body composition trends (if data available)
  async getBodyCompositionTrends(userId: string, period: string): Promise<BodyCompositionTrend[]>;
}
```

### 2. Weight Controller (`server/src/controllers/weightController.ts`)

Endpoints:
- `POST /api/weight` - Log weight entry
- `GET /api/weight/latest` - Get most recent weight
- `GET /api/weight?startDate=&endDate=` - Get weight records
- `GET /api/weight/:id` - Get specific record
- `PUT /api/weight/:id` - Update record
- `DELETE /api/weight/:id` - Delete record
- `GET /api/weight/trends?period=` - Get weight trends with moving averages
- `GET /api/weight/progress` - Get goal progress stats
- `GET /api/weight/stats` - Get statistics (min, max, avg, change)
- `PUT /api/user/weight-goal` - Set/update weight goal
- `GET /api/weight/body-composition` - Body composition trends

### 3. Validation Schemas (`server/src/validation/schemas.ts`)

```typescript
export const createWeightRecordSchema = z.object({
  weight: z.number().min(20).max(500), // 20kg to 500kg
  recordedAt: z.string().datetime().optional(),
  bodyFatPercentage: z.number().min(1).max(70).optional(),
  muscleMass: z.number().min(10).max(150).optional(),
  boneMass: z.number().min(1).max(10).optional(),
  waterPercentage: z.number().min(20).max(80).optional(),
  visceralFat: z.number().min(1).max(60).optional(),
  notes: z.string().max(500).optional(),
});

export const weightGoalSchema = z.object({
  goalWeight: z.number().min(30).max(300),
  goalType: z.enum(['LOSE', 'GAIN', 'MAINTAIN']),
  pace: z.number().min(0.1).max(1.5).optional(), // kg per week
});

export const weightTrendsQuerySchema = z.object({
  period: z.enum(['month', '3months', '6months', 'year', 'all']),
});
```

## Mobile App Implementation

### 1. API Client (`lib/api/weight.ts`)

```typescript
export const weightApi = {
  logWeight: (data: WeightRecordInput) => 
    apiClient.post('/weight', data),
  
  getLatest: () => 
    apiClient.get('/weight/latest'),
  
  getRecords: (startDate: string, endDate: string) => 
    apiClient.get('/weight', { params: { startDate, endDate } }),
  
  getRecord: (id: string) => 
    apiClient.get(`/weight/${id}`),
  
  updateRecord: (id: string, data: Partial<WeightRecordInput>) => 
    apiClient.put(`/weight/${id}`, data),
  
  deleteRecord: (id: string) => 
    apiClient.delete(`/weight/${id}`),
  
  getTrends: (period: string) => 
    apiClient.get('/weight/trends', { params: { period } }),
  
  getProgress: () => 
    apiClient.get('/weight/progress'),
  
  setGoal: (goalWeight: number, goalType: string, pace?: number) => 
    apiClient.put('/user/weight-goal', { goalWeight, goalType, pace }),
  
  getBodyComposition: (period: string) => 
    apiClient.get('/weight/body-composition', { params: { period } }),
};
```

### 2. Weight Tracking Screen (`app/weight/index.tsx`)

Main screen layout:

**Header Section:**
- Current weight (large, prominent)
- Change from last entry (+/- X.X kg)
- Last weighed date

**Progress Card (if goal set):**
- Visual progress bar: Start → Current → Goal
- "X kg to go" or "Goal reached!"
- Projected completion date
- Weekly rate indicator

**Quick Log Button:**
- Floating action button or prominent CTA
- Opens weight entry modal

**Trend Chart:**
- Line chart with:
  - Raw data points
  - 7-day moving average line (smoothed trend)
  - 30-day moving average line (long-term trend)
  - Goal line (horizontal target)
- Period selector: 1M, 3M, 6M, 1Y, All
- Pinch to zoom, pan to scroll
- Tap point to see details

**History Section:**
- List of recent entries
- Each shows: date, weight, change from previous
- Body composition data if available
- Swipe to delete

**Stats Card:**
- Lowest weight (with date)
- Highest weight (with date)
- Average weight (selected period)
- Total change (selected period)

### 3. Weight Entry Modal (`lib/components/WeightEntryModal.tsx`)

- Large numeric input for weight
- Unit toggle (kg/lb) with instant conversion
- Date/time picker (defaults to now)
- Optional body composition fields (expandable):
  - Body fat %
  - Muscle mass
  - Water %
- Notes field
- Save button with validation

**Smart Features:**
- Pre-fill with last weight for quick increment/decrement
- +/- 0.1kg buttons for fine adjustment
- Haptic feedback on changes

### 4. Weight Goal Setup (`app/weight/goal.tsx`)

Step-by-step wizard:

**Step 1: Current Status**
- Display current weight
- Enter if not set

**Step 2: Goal Type**
- Lose weight (with icon)
- Gain weight (with icon)
- Maintain weight (with icon)

**Step 3: Target Weight**
- Slider or numeric input
- Shows BMI at target (if height set)
- Healthy range indicator

**Step 4: Timeline (for lose/gain)**
- Pace selector: Slow (0.25kg/wk), Moderate (0.5kg/wk), Fast (0.75kg/wk), Aggressive (1kg/wk)
- Projected completion date shown
- Calorie adjustment recommendation

**Step 5: Confirmation**
- Summary of goal
- Daily calorie target (calculated)
- "Set Goal" button

### 5. Body Composition View (`app/weight/composition.tsx`)

For users with smart scale data:
- Body fat % trend chart
- Muscle mass trend chart
- Water % trend chart
- Visceral fat level
- Comparison cards: Current vs 30 days ago

### 6. Weight Widget (`lib/components/WeightWidget.tsx`)

Dashboard widget:
- Current weight
- Change arrow (up/down)
- Mini sparkline of last 7 entries
- "Log Weight" quick action

## Unit Conversion System

```typescript
// lib/utils/weightConversions.ts
export const WEIGHT_CONVERSIONS = {
  kg_to_lb: 2.20462,
  lb_to_kg: 0.453592,
};

export const convertWeight = (value: number, from: 'kg' | 'lb', to: 'kg' | 'lb'): number => {
  if (from === to) return value;
  return from === 'kg' ? value * WEIGHT_CONVERSIONS.kg_to_lb : value * WEIGHT_CONVERSIONS.lb_to_kg;
};

// Always store in kg, display in user's preference
export const formatWeight = (kg: number, unit: 'kg' | 'lb'): string => {
  const value = unit === 'kg' ? kg : convertWeight(kg, 'kg', 'lb');
  return `${value.toFixed(1)} ${unit}`;
};
```

## BMI Calculation

```typescript
// lib/utils/bmi.ts
export const calculateBMI = (weightKg: number, heightCm: number): number => {
  const heightM = heightCm / 100;
  return weightKg / (heightM * heightM);
};

export const getBMICategory = (bmi: number): string => {
  if (bmi < 18.5) return 'Underweight';
  if (bmi < 25) return 'Normal';
  if (bmi < 30) return 'Overweight';
  return 'Obese';
};

export const getHealthyWeightRange = (heightCm: number): { min: number; max: number } => {
  const heightM = heightCm / 100;
  return {
    min: 18.5 * heightM * heightM,
    max: 24.9 * heightM * heightM,
  };
};
```

## Integration Points

### 1. Dashboard (`app/(tabs)/index.tsx`):
```tsx
<WeightWidget 
  currentWeight={user.currentWeight}
  goalWeight={user.goalWeight}
  onLogPress={() => setShowWeightModal(true)}
  onPress={() => router.push('/weight')}
/>
```

### 2. Profile Settings:
- Weight goal link in profile
- Unit preference (kg/lb)
- Connect smart scale option

### 3. HealthKit Integration (Task 5):
- Read weight from Apple Health (HKQuantityTypeIdentifierBodyMass)
- Read body fat % (HKQuantityTypeIdentifierBodyFatPercentage)
- Read lean body mass (HKQuantityTypeIdentifierLeanBodyMass)
- Write logged weights to HealthKit
- Deduplicate entries by source

### 4. ML Insights (Task 8):
- Correlate weight trends with:
  - Calorie intake patterns
  - Macronutrient ratios
  - Exercise frequency
  - Sleep quality
- Generate insights: "You lose more weight in weeks with 4+ workouts"

### 5. Weekly Reports (planned):
- Include weight change in weekly summary
- Progress toward goal percentage

## Chart Implementation

Use react-native-chart-kit (already installed) or Victory Native:

```typescript
// Weight trend chart data preparation
const prepareChartData = (records: WeightRecord[], showMovingAverage: boolean) => {
  const sortedRecords = records.sort((a, b) => 
    new Date(a.recordedAt).getTime() - new Date(b.recordedAt).getTime()
  );
  
  const labels = sortedRecords.map(r => format(new Date(r.recordedAt), 'MMM d'));
  const data = sortedRecords.map(r => r.weight);
  
  // Calculate 7-day moving average
  const movingAvg7d = data.map((_, i) => {
    const start = Math.max(0, i - 6);
    const slice = data.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
  
  return { labels, datasets: [{ data }, { data: movingAvg7d }] };
};
```

## Design Specifications

### Colors (extend `lib/theme/colors.ts`):
```typescript
weight: {
  losing: '#10B981',      // Green - losing weight (toward goal)
  gaining: '#EF4444',     // Red - gaining when trying to lose
  maintaining: '#6B7280', // Gray - stable
  goalLine: '#8B5CF6',    // Purple - goal target line
  movingAverage: '#F59E0B', // Amber - trend line
}
```

### Animations:
- Weight change animates from old → new value
- Chart draws progressively on load
- Progress bar fills with spring animation
- Entry deletion slides out

## Error Handling

- Network error: Cache locally, sync when online
- Invalid weight: Show inline validation (e.g., "Weight must be between 20-500 kg")
- Duplicate entry for date: Ask to update or add new
- Smart scale sync errors: Show status, offer manual entry

**Test Strategy:**

## Testing Strategy

### Backend Unit Tests (`server/src/__tests__/weight.test.ts`)

1. **Service Tests**:
   - createWeightRecord: Creates entry and updates user.currentWeight
   - createWeightRecord: Validates weight bounds
   - createWeightRecord: Handles body composition data
   - getWeightTrends: Calculates moving averages correctly
   - getWeightProgress: Returns correct progress percentage
   - getWeightProgress: Handles edge cases (no goal, goal reached)
   - setWeightGoal: Updates user goal fields
   - deleteWeightRecord: Only owner can delete

2. **Controller Tests**:
   - POST /api/weight: 201 with valid data
   - POST /api/weight: 400 with invalid weight
   - GET /api/weight/trends: Returns correct period data
   - GET /api/weight/progress: Calculates progress correctly
   - PUT /api/user/weight-goal: Updates goal

3. **Integration Tests**:
   - Log multiple weights → Verify trends calculated correctly
   - Set goal → Log weights → Verify progress updates

### Mobile Tests

1. **Component Tests**:
   - WeightWidget: Displays current weight and change
   - WeightEntryModal: Validates input, converts units
   - WeightChart: Renders with various data sizes

2. **Screen Tests** (`__tests__/screens/WeightScreen.test.tsx`):
   - Loads latest weight on mount
   - Chart renders with trend data
   - Entry modal opens and saves
   - Unit toggle converts display
   - Goal progress shows correctly

3. **Utility Tests**:
   - Unit conversions: kg ↔ lb
   - BMI calculation
   - Moving average calculation

### E2E Tests (Maestro)

1. **Weight Logging Flow**:
   - Open weight screen
   - Tap log weight
   - Enter weight
   - Verify history updates
   - Verify chart updates

2. **Goal Setting Flow**:
   - Navigate to goal setup
   - Complete wizard
   - Verify goal appears on main screen

### Manual Testing

- [ ] Weight displays in correct unit
- [ ] Unit toggle converts correctly
- [ ] Chart is scrollable/zoomable
- [ ] Moving average line is smooth
- [ ] Goal line shows at correct position
- [ ] Progress percentage is accurate
- [ ] Body composition fields save correctly
- [ ] HealthKit sync works (if connected)

## Subtasks

### 43.1. Implement backend weight service, controller, validation schemas, and API routes

**Status:** pending  
**Dependencies:** None  

Create WeightService class with CRUD operations, trend analysis, goal progress calculations, and BMI utilities. Implement WeightController with error handling and Zod validation. Add weight-specific validation schemas and register routes in Express app.

**Details:**

FILES TO CREATE:
- server/src/services/weightService.ts: Class-based service with methods:
  - createWeightRecord(userId, data): Creates record and syncs User.currentWeight
  - getWeightRecords(userId, startDate, endDate): Fetch records with date range
  - getLatestWeight(userId): Get most recent entry
  - updateWeightRecord(userId, recordId, data): Update existing record
  - deleteWeightRecord(userId, recordId): Delete record
  - getWeightTrends(userId, period): Calculate 7-day and 30-day moving averages
  - getWeightProgress(userId): Calculate progress toward goalWeight (current vs goal vs start, percent complete, weekly/monthly rate of change, projected date)
  - getWeightStats(userId): Min, max, average, total change
  - setWeightGoal(userId, goalWeight): Update User.goalWeight
  - calculateBMI(weightKg, heightCm): BMI calculation utility
  - getHealthyWeightRange(heightCm): Return {min, max} for BMI 18.5-24.9

- server/src/controllers/weightController.ts: Controller methods using withErrorHandling:
  - createWeightRecord: POST /weight (validates with createWeightRecordSchema)
  - getLatestWeight: GET /weight/latest
  - getWeightRecords: GET /weight?startDate=&endDate=
  - getWeightRecord: GET /weight/:id (use ErrorHandlers.withNotFound)
  - updateWeightRecord: PUT /weight/:id
  - deleteWeightRecord: DELETE /weight/:id (use ErrorHandlers.withNotFound)
  - getWeightTrends: GET /weight/trends?period= (validates period enum)
  - getWeightProgress: GET /weight/progress
  - getWeightStats: GET /weight/stats
  - setWeightGoal: PUT /user/weight-goal

- server/src/validation/schemas.ts: Add schemas:
  - createWeightRecordSchema: weight (20-500kg), recordedAt (optional datetime), notes (optional, max 500 chars)
  - updateWeightRecordSchema: Partial of create schema
  - weightGoalSchema: goalWeight (30-300kg)
  - weightTrendsQuerySchema: period enum ('week' | 'month' | '3months' | '6months' | 'year' | 'all')

- server/src/routes/weightRoutes.ts: Express router with authenticate middleware, register all endpoints

- server/src/index.ts: Register weight routes as app.use('/api/weight', weightRoutes)

FOLLOW EXISTING PATTERNS:
- Service: Export singleton instance (export const weightService = new WeightService())
- Controller: Class with methods using withErrorHandling, requireAuth
- All methods must have zero 'any' types, use proper Prisma types
- Use constants from config/constants.ts (HTTP_STATUS, ERROR_MESSAGES)
- Use dateHelpers for date parsing
- Service methods throw Error, controller catches with withErrorHandling
- Update User.currentWeight in transaction when creating WeightRecord

BMI CALCULATION:
- Formula: weight(kg) / (height(m)^2)
- Categories: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, >=30 obese
- Healthy range: 18.5 * height(m)^2 to 24.9 * height(m)^2

TREND ANALYSIS:
- 7-day moving average: Average of last 7 data points
- 30-day moving average: Average of last 30 data points
- Weekly change: (current - weight_7_days_ago) / 1
- Monthly change: (current - weight_30_days_ago) / 1
- Trend direction: 'losing' if weekly change < -0.1, 'gaining' if > +0.1, else 'maintaining'

GOAL PROGRESS:
- Start weight: User.currentWeight when goal was set (or earliest WeightRecord if not tracked)
- Current weight: Latest WeightRecord.weight
- Goal weight: User.goalWeight
- Weight lost/gained: startWeight - currentWeight (negative = gained)
- Percent complete: ((startWeight - currentWeight) / (startWeight - goalWeight)) * 100
- Projected goal date: Based on weekly rate of change, calculate weeks remaining

TESTING REQUIREMENTS:
- Unit tests in server/src/__tests__/weight.test.ts
- Test service methods: CRUD, trends, progress, BMI
- Test validation: invalid weights, date formats
- Test User.currentWeight sync on create
- Test goal progress calculation edge cases (no goal, already at goal)
- Follow test patterns from __tests__/meal.test.ts

### 43.2. Create mobile API client with TypeScript types for weight tracking

**Status:** pending  
**Dependencies:** 43.1  

Add weight-related TypeScript interfaces to lib/types/index.ts and create weightApi client in lib/api/weight.ts following the object-based pattern used in mealsApi.

**Details:**

FILES TO CREATE/UPDATE:

- lib/types/index.ts: Add interfaces:
```typescript
export interface WeightRecord {
  id: string;
  userId: string;
  weight: number;        // kg
  recordedAt: string;    // ISO datetime
  notes?: string;
  createdAt: string;
}

export interface CreateWeightRecordInput {
  weight: number;        // kg
  recordedAt?: string;   // ISO datetime, defaults to now
  notes?: string;
}

export interface WeightTrendDataPoint {
  date: string;              // ISO date
  weight: number;
  movingAverage7d: number;   // 7-day moving average
  movingAverage30d: number;  // 30-day moving average
}

export interface WeightProgress {
  currentWeight: number;
  goalWeight: number;
  startWeight: number;       // When goal was set
  weightChange: number;      // Lost (negative) or gained (positive)
  percentComplete: number;   // 0-100+
  projectedGoalDate: string | null; // ISO date
  weeklyChange: number;      // kg/week
  monthlyChange: number;     // kg/month
  trend: 'losing' | 'gaining' | 'maintaining';
}

export interface WeightStats {
  min: number;
  max: number;
  average: number;
  totalChange: number;       // From first to latest record
  recordCount: number;
}

export interface WeightGoalInput {
  goalWeight: number;        // kg
}

export interface BMIInfo {
  bmi: number;
  category: 'underweight' | 'normal' | 'overweight' | 'obese';
  healthyWeightRange: { min: number; max: number }; // kg
}
```

- lib/api/weight.ts: Create API client following mealsApi pattern:
```typescript
import api from './client';
import {
  WeightRecord,
  CreateWeightRecordInput,
  WeightTrendDataPoint,
  WeightProgress,
  WeightStats,
  WeightGoalInput,
  BMIInfo,
} from '../types';

export const weightApi = {
  async createWeightRecord(data: CreateWeightRecordInput): Promise<WeightRecord> {
    const response = await api.post<WeightRecord>('/weight', data);
    return response.data;
  },

  async getLatestWeight(): Promise<WeightRecord | null> {
    const response = await api.get<WeightRecord | null>('/weight/latest');
    return response.data;
  },

  async getWeightRecords(startDate: string, endDate: string): Promise<WeightRecord[]> {
    const response = await api.get<WeightRecord[]>('/weight', {
      params: { startDate, endDate },
    });
    return response.data;
  },

  async getWeightRecord(id: string): Promise<WeightRecord> {
    const response = await api.get<WeightRecord>(`/weight/${id}`);
    return response.data;
  },

  async updateWeightRecord(
    id: string,
    data: Partial<CreateWeightRecordInput>
  ): Promise<WeightRecord> {
    const response = await api.put<WeightRecord>(`/weight/${id}`, data);
    return response.data;
  },

  async deleteWeightRecord(id: string): Promise<void> {
    await api.delete(`/weight/${id}`);
  },

  async getWeightTrends(
    period: 'week' | 'month' | '3months' | '6months' | 'year' | 'all'
  ): Promise<WeightTrendDataPoint[]> {
    const response = await api.get<WeightTrendDataPoint[]>('/weight/trends', {
      params: { period },
    });
    return response.data;
  },

  async getWeightProgress(): Promise<WeightProgress | null> {
    const response = await api.get<WeightProgress | null>('/weight/progress');
    return response.data;
  },

  async getWeightStats(): Promise<WeightStats> {
    const response = await api.get<WeightStats>('/weight/stats');
    return response.data;
  },

  async setWeightGoal(data: WeightGoalInput): Promise<void> {
    await api.put('/user/weight-goal', data);
  },

  async calculateBMI(weightKg: number, heightCm: number): Promise<BMIInfo> {
    const response = await api.get<BMIInfo>('/weight/bmi', {
      params: { weightKg, heightCm },
    });
    return response.data;
  },
};
```

FOLLOW EXISTING PATTERNS:
- Use api client from './client' (has JWT interceptor)
- All dates as ISO strings for API communication
- Proper TypeScript typing for all methods
- Error handling via getErrorMessage in consuming components
- Export as const object (not class)

UNIT CONVERSION NOTES:
- Backend stores weights in kg
- Mobile can display in kg or lbs based on user preference
- Conversion factors: 1 kg = 2.20462 lbs, 1 lb = 0.453592 kg
- Always send kg to API, convert for display only

TESTING:
- Type-check with TypeScript compiler
- Manual API integration tests with backend running
- Error handling tests (network failures, 401, 404, validation errors)

### 43.3. Build weight tracking main screen with trend chart and history list

**Status:** pending  
**Dependencies:** 43.2  

Create app/weight/index.tsx as the main weight tracking screen with header summary, progress card, trend line chart with moving averages, period selector, and swipeable history list following the pattern from health/[metricType].tsx.

**Details:**

FILE TO CREATE:

- app/weight/index.tsx: Main weight tracking screen

COMPONENT STRUCTURE:

```typescript
import { useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { LineChart } from 'react-native-chart-kit';
import { weightApi } from '@/lib/api/weight';
import { WeightRecord, WeightTrendDataPoint, WeightProgress, WeightStats } from '@/lib/types';
import { SwipeableWeightCard } from '@/lib/components/SwipeableWeightCard'; // Create in subtask 4
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

type DateRange = 'week' | 'month' | '3months' | '6months' | 'year' | 'all';

export default function WeightTrackingScreen() {
  const router = useRouter();
  const { isTablet, getSpacing, width: screenWidth } = useResponsive();
  const responsiveSpacing = getSpacing();

  const [trendData, setTrendData] = useState<WeightTrendDataPoint[]>([]);
  const [progress, setProgress] = useState<WeightProgress | null>(null);
  const [stats, setStats] = useState<WeightStats | null>(null);
  const [recentRecords, setRecentRecords] = useState<WeightRecord[]>([]);
  const [dateRange, setDateRange] = useState<DateRange>('month');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [visibleEntriesCount, setVisibleEntriesCount] = useState(10);

  const loadData = useCallback(async () => { /* Fetch trends, progress, stats, recent records */ });
  const onRefresh = useCallback(async () => { /* Pull to refresh */ });
  const handleDeleteRecord = async (id: string) => { /* Delete and reload */ };
  const handleLoadMore = () => setVisibleEntriesCount(prev => prev + 10);

  // ... implementation
}
```

SCREEN SECTIONS:

1. **HEADER SECTION** (sticky at top):
   - LinearGradient background (gradients.primary)
   - Title: "Weight Tracking"
   - Current weight (large, bold): "72.5 kg"
   - Change from last entry: "+0.3 kg" with up/down arrow icon (green if losing toward goal, red if gaining)
   - Last weighed: "Today" or "X days ago"
   - Settings icon (navigate to /weight/settings for goal setup)

2. **PROGRESS CARD** (if User.goalWeight is set):
   - Card with shadow
   - Progress bar visualization:
     - Three markers: Start (left) → Current (middle) → Goal (right)
     - Filled bar showing progress
   - "X kg to go" or "Goal reached! 🎉"
   - Stats row:
     - Weekly change: "-0.5 kg/wk"
     - Percent complete: "60% complete"
   - Projected completion: "Estimated: Mar 15, 2025" (if on track)
   - Tap to navigate to /weight/goal for goal setup/editing

3. **TREND CHART CARD**:
   - Period selector buttons (week, month, 3months, 6months, year, all)
   - LineChart from react-native-chart-kit:
     - Dataset 1: Raw weight data points (color: colors.primary)
     - Dataset 2: 7-day moving average (color: colors.amber, dashed)
     - Dataset 3: 30-day moving average (color: colors.purple, dashed)
     - Goal line (horizontal, color: colors.success, dashed)
   - Chart config: responsive width (screenWidth - padding), height 220
   - Y-axis: Weight in kg (user's unit preference)
   - X-axis: Date labels (formatted)
   - Tap data point to show details tooltip
   - Empty state: "No weight data yet. Add your first entry!"

4. **STATS CARD**:
   - Grid layout (2 columns):
     - Lowest: "68.0 kg" (with date)
     - Highest: "75.2 kg" (with date)
     - Average: "71.5 kg"
     - Total change: "-3.2 kg"
   - Styled with borders, icons, subtle backgrounds

5. **HISTORY SECTION**:
   - Title: "Recent Entries"
   - List of SwipeableWeightCard components (create in subtask 4)
   - Each card shows: date, weight, change from previous, notes
   - Swipe left to delete (with confirmation)
   - Tap to edit (navigate to /weight/edit/[id])
   - Load more button at bottom if > 10 entries
   - Empty state: "No entries yet"

6. **FLOATING ACTION BUTTON (FAB)**:
   - Fixed bottom-right position
   - Icon: Ionicons "add" or "scale"
   - Opens weight entry modal/screen: router.push('/weight/add')
   - Styled with shadow, primary color gradient

CHART DATA PREPARATION:
```typescript
const prepareChartData = (data: WeightTrendDataPoint[], goalWeight?: number) => {
  if (data.length === 0) return null;
  
  const labels = data.map(d => format(new Date(d.date), 'MMM d'));
  const weights = data.map(d => d.weight);
  const avg7d = data.map(d => d.movingAverage7d);
  const avg30d = data.map(d => d.movingAverage30d);
  const goalLine = goalWeight ? data.map(() => goalWeight) : undefined;
  
  return {
    labels,
    datasets: [
      { data: weights, color: () => colors.primary, strokeWidth: 2 },
      { data: avg7d, color: () => colors.amber, strokeWidth: 1.5, withDots: false },
      { data: avg30d, color: () => colors.purple, strokeWidth: 1.5, withDots: false },
      ...(goalLine ? [{ data: goalLine, color: () => colors.success, strokeWidth: 1, withDots: false }] : []),
    ],
  };
};
```

RESPONSIVE DESIGN:
- Use useResponsive hook for spacing, chart width
- Tablet: 2-column layout for stats, wider chart
- Phone: Single column, scrollable
- Chart width calculation: screenWidth - (padding * 2) - (card padding * 2)
- Max width on tablet: FORM_MAX_WIDTH

ERROR HANDLING:
- Network errors: Show error message, retry button
- Empty data: Friendly empty states with CTA to add first entry
- Loading states: ActivityIndicator with shimmer effect

ANIMATIONS:
- Pull-to-refresh with RefreshControl
- Chart draws progressively on mount (built-in to react-native-chart-kit)
- Card swipe delete animation
- FAB scale animation on press

ACCESSIBILITY:
- All touchables have accessibilityLabel
- Chart has accessibilityLabel describing trend
- Screen reader announces current weight and progress

FOLLOW EXISTING PATTERNS:
- Similar to app/health/[metricType].tsx structure
- Use SafeAreaView from react-native-safe-area-context
- Use colors, gradients, shadows from theme/colors.ts
- Error handling with getErrorMessage utility
- showAlert for confirmations and errors

REGISTRATION:
- Add to app/_layout.tsx with headerShown: false (custom header in component)
- Add navigation from dashboard widget or profile screen

### 43.4. Create weight entry modal, swipeable card component, and goal setup screen

**Status:** pending  
**Dependencies:** 43.2  

Build reusable UI components: weight entry modal with numeric input and validation, swipeable weight record card for history lists, and goal setup wizard screen with step-by-step flow for setting weight goals.

**Details:**

FILES TO CREATE:

1. **lib/components/WeightEntryModal.tsx** - Modal for adding/editing weight

Component structure:
```typescript
import { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Modal, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { weightApi } from '@/lib/api/weight';
import { CreateWeightRecordInput, WeightRecord } from '@/lib/types';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';

interface WeightEntryModalProps {
  visible: boolean;
  onClose: () => void;
  onSaved: () => void;
  initialData?: WeightRecord; // For editing existing record
  lastWeight?: number;         // Pre-fill for quick adjustments
}

export function WeightEntryModal({ visible, onClose, onSaved, initialData, lastWeight }: WeightEntryModalProps) {
  const [weight, setWeight] = useState(initialData?.weight.toString() || lastWeight?.toString() || '');
  const [recordedAt, setRecordedAt] = useState(initialData ? new Date(initialData.recordedAt) : new Date());
  const [notes, setNotes] = useState(initialData?.notes || '');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  const handleSave = async () => {
    // Validation: weight must be 20-500 kg
    const weightNum = parseFloat(weight);
    if (isNaN(weightNum) || weightNum < 20 || weightNum > 500) {
      showAlert('Invalid Weight', 'Weight must be between 20 and 500 kg');
      return;
    }
    
    setIsSaving(true);
    try {
      const data: CreateWeightRecordInput = {
        weight: weightNum,
        recordedAt: recordedAt.toISOString(),
        notes: notes.trim() || undefined,
      };
      
      if (initialData) {
        await weightApi.updateWeightRecord(initialData.id, data);
      } else {
        await weightApi.createWeightRecord(data);
      }
      
      onSaved();
      onClose();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to save weight'));
    } finally {
      setIsSaving(false);
    }
  };
  
  return (
    <Modal visible={visible} animationType="slide" transparent>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.overlay}>
        <View style={styles.modalContent}>
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>{initialData ? 'Edit Weight' : 'Log Weight'}</Text>
            <TouchableOpacity onPress={onClose}>
              <Ionicons name="close" size={28} color={colors.text} />
            </TouchableOpacity>
          </View>
          
          {/* Weight Input - Large, centered */}
          <View style={styles.weightInputContainer}>
            <TextInput
              style={styles.weightInput}
              value={weight}
              onChangeText={setWeight}
              keyboardType="decimal-pad"
              placeholder="0.0"
              autoFocus
            />
            <Text style={styles.unitLabel}>kg</Text>
          </View>
          
          {/* Quick Adjustment Buttons */}
          <View style={styles.quickButtons}>
            <TouchableOpacity onPress={() => setWeight((prev) => (parseFloat(prev || '0') - 0.1).toFixed(1))}>
              <Text style={styles.quickButton}>-0.1</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setWeight((prev) => (parseFloat(prev || '0') + 0.1).toFixed(1))}>
              <Text style={styles.quickButton}>+0.1</Text>
            </TouchableOpacity>
          </View>
          
          {/* Date/Time Picker */}
          <TouchableOpacity style={styles.dateButton} onPress={() => setShowDatePicker(true)}>
            <Ionicons name="calendar-outline" size={20} color={colors.textSecondary} />
            <Text style={styles.dateText}>{format(recordedAt, 'MMM d, yyyy h:mm a')}</Text>
          </TouchableOpacity>
          {showDatePicker && (
            <DateTimePicker
              value={recordedAt}
              mode="datetime"
              onChange={(event, date) => {
                setShowDatePicker(false);
                if (date) setRecordedAt(date);
              }}
            />
          )}
          
          {/* Notes Input */}
          <TextInput
            style={styles.notesInput}
            value={notes}
            onChangeText={setNotes}
            placeholder="Notes (optional)"
            multiline
            maxLength={500}
          />
          
          {/* Save Button */}
          <TouchableOpacity
            style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
            onPress={handleSave}
            disabled={isSaving}
          >
            <Text style={styles.saveButtonText}>{isSaving ? 'Saving...' : 'Save'}</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}
```

2. **lib/components/SwipeableWeightCard.tsx** - Swipeable card for weight record in history list

Component structure:
```typescript
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Swipeable from 'react-native-gesture-handler/Swipeable';
import { WeightRecord } from '@/lib/types';
import { format } from 'date-fns';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';

interface SwipeableWeightCardProps {
  record: WeightRecord;
  previousWeight?: number;     // For showing change
  onPress: () => void;         // Tap to edit
  onDelete: () => void;        // Swipe to delete
}

export function SwipeableWeightCard({ record, previousWeight, onPress, onDelete }: SwipeableWeightCardProps) {
  const change = previousWeight ? record.weight - previousWeight : null;
  const changeColor = change ? (change > 0 ? colors.error : colors.success) : colors.textSecondary;
  const changeIcon = change ? (change > 0 ? 'arrow-up' : 'arrow-down') : null;
  
  const renderRightActions = (progress: Animated.AnimatedInterpolation<number>, dragX: Animated.AnimatedInterpolation<number>) => (
    <TouchableOpacity style={styles.deleteAction} onPress={onDelete}>
      <Ionicons name="trash-outline" size={24} color="white" />
      <Text style={styles.deleteText}>Delete</Text>
    </TouchableOpacity>
  );
  
  return (
    <Swipeable renderRightActions={renderRightActions}>
      <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.7}>
        <View style={styles.leftSection}>
          <Text style={styles.date}>{format(new Date(record.recordedAt), 'MMM d, yyyy')}</Text>
          <Text style={styles.time}>{format(new Date(record.recordedAt), 'h:mm a')}</Text>
          {record.notes && <Text style={styles.notes} numberOfLines={1}>{record.notes}</Text>}
        </View>
        
        <View style={styles.rightSection}>
          <Text style={styles.weight}>{record.weight.toFixed(1)} kg</Text>
          {change !== null && (
            <View style={styles.changeContainer}>
              {changeIcon && <Ionicons name={changeIcon} size={14} color={changeColor} />}
              <Text style={[styles.change, { color: changeColor }]}>
                {change > 0 ? '+' : ''}{change.toFixed(1)} kg
              </Text>
            </View>
          )}
        </View>
      </TouchableOpacity>
    </Swipeable>
  );
}
```

3. **app/weight/goal.tsx** - Goal setup wizard screen

Screen structure:
```typescript
import { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { weightApi } from '@/lib/api/weight';
import { useAuth } from '@/lib/context/AuthContext';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';

type GoalType = 'lose' | 'gain' | 'maintain';

export default function WeightGoalScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const [step, setStep] = useState(1);
  const [goalWeight, setGoalWeight] = useState(user?.goalWeight?.toString() || '');
  const [goalType, setGoalType] = useState<GoalType>('lose');
  const [isSaving, setIsSaving] = useState(false);
  
  const currentWeight = user?.currentWeight || 0;
  const height = user?.height || 0;
  const bmi = height > 0 ? currentWeight / Math.pow(height / 100, 2) : 0;
  const targetBMI = height > 0 && goalWeight ? parseFloat(goalWeight) / Math.pow(height / 100, 2) : 0;
  
  const handleSaveGoal = async () => {
    const goalWeightNum = parseFloat(goalWeight);
    if (isNaN(goalWeightNum) || goalWeightNum < 30 || goalWeightNum > 300) {
      showAlert('Invalid Goal', 'Goal weight must be between 30 and 300 kg');
      return;
    }
    
    setIsSaving(true);
    try {
      await weightApi.setWeightGoal({ goalWeight: goalWeightNum });
      showAlert('Success', 'Weight goal set successfully!');
      router.back();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to set goal'));
    } finally {
      setIsSaving(false);
    }
  };
  
  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <LinearGradient colors={gradients.primary} style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Set Weight Goal</Text>
      </LinearGradient>
      
      <ScrollView style={styles.content}>
        {step === 1 && (
          <View>
            <Text style={styles.stepTitle}>Current Weight</Text>
            <Text style={styles.currentWeight}>{currentWeight.toFixed(1)} kg</Text>
            {bmi > 0 && <Text style={styles.bmiText}>BMI: {bmi.toFixed(1)}</Text>}
            <TouchableOpacity style={styles.nextButton} onPress={() => setStep(2)}>
              <Text style={styles.nextButtonText}>Next</Text>
            </TouchableOpacity>
          </View>
        )}
        
        {step === 2 && (
          <View>
            <Text style={styles.stepTitle}>Goal Type</Text>
            <View style={styles.goalTypeButtons}>
              <TouchableOpacity
                style={[styles.goalTypeButton, goalType === 'lose' && styles.goalTypeButtonActive]}
                onPress={() => setGoalType('lose')}
              >
                <Ionicons name="arrow-down" size={32} color={goalType === 'lose' ? colors.primary : colors.textSecondary} />
                <Text style={styles.goalTypeLabel}>Lose Weight</Text>
              </TouchableOpacity>
              {/* Similar buttons for 'gain' and 'maintain' */}
            </View>
            <TouchableOpacity style={styles.nextButton} onPress={() => setStep(3)}>
              <Text style={styles.nextButtonText}>Next</Text>
            </TouchableOpacity>
          </View>
        )}
        
        {step === 3 && (
          <View>
            <Text style={styles.stepTitle}>Target Weight</Text>
            <TextInput
              style={styles.goalInput}
              value={goalWeight}
              onChangeText={setGoalWeight}
              keyboardType="decimal-pad"
              placeholder="Goal weight (kg)"
            />
            {targetBMI > 0 && <Text style={styles.bmiText}>Target BMI: {targetBMI.toFixed(1)}</Text>}
            <TouchableOpacity
              style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
              onPress={handleSaveGoal}
              disabled={isSaving}
            >
              <Text style={styles.saveButtonText}>{isSaving ? 'Saving...' : 'Set Goal'}</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}
```

COMMON PATTERNS:
- Use theme colors, spacing, typography
- Error handling with getErrorMessage
- showAlert for confirmations and errors
- Haptic feedback on button presses (Haptics.impactAsync)
- Accessibility labels on all interactive elements

REGISTRATION:
- Add app/weight/goal.tsx to app/_layout.tsx with headerShown: false
- Export components from lib/components/index.ts (if exists)

### 43.5. Integrate weight tracking into dashboard and profile with widget and navigation

**Status:** pending  
**Dependencies:** 43.3, 43.4  

Add weight tracking widget to dashboard (app/(tabs)/index.tsx) showing current weight and quick-log action, add navigation to weight screens from profile, and integrate with existing user settings.

**Details:**

FILES TO UPDATE:

1. **lib/components/WeightWidget.tsx** - Dashboard widget component

Create new component:
```typescript
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { Sparklines, SparklinesLine } from 'react-sparklines'; // Or custom sparkline implementation
import { WeightRecord } from '@/lib/types';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';

interface WeightWidgetProps {
  currentWeight?: number;      // From user profile
  goalWeight?: number;         // From user profile
  recentRecords: WeightRecord[]; // Last 7-14 entries for sparkline
  onLogPress: () => void;      // Open weight entry modal
  onPress: () => void;         // Navigate to full weight screen
  isLoading?: boolean;
}

export function WeightWidget({ currentWeight, goalWeight, recentRecords, onLogPress, onPress, isLoading }: WeightWidgetProps) {
  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator color={colors.primary} />
      </View>
    );
  }
  
  if (!currentWeight && recentRecords.length === 0) {
    // Empty state - no weight logged yet
    return (
      <TouchableOpacity style={styles.emptyCard} onPress={onLogPress}>
        <Ionicons name="scale-outline" size={48} color={colors.primary} />
        <Text style={styles.emptyTitle}>Track Your Weight</Text>
        <Text style={styles.emptySubtitle}>Tap to log your first entry</Text>
      </TouchableOpacity>
    );
  }
  
  const latestWeight = recentRecords[0]?.weight || currentWeight || 0;
  const previousWeight = recentRecords[1]?.weight;
  const change = previousWeight ? latestWeight - previousWeight : null;
  const changeColor = change ? (change > 0 ? colors.error : colors.success) : colors.textSecondary;
  const changeIcon = change ? (change > 0 ? 'arrow-up' : 'arrow-down') : null;
  
  // Prepare sparkline data (last 7 entries)
  const sparklineData = recentRecords.slice(0, 7).reverse().map(r => r.weight);
  
  return (
    <TouchableOpacity style={styles.card} onPress={onPress} activeOpacity={0.8}>
      <LinearGradient colors={gradients.weight} style={styles.gradient}>
        <View style={styles.header}>
          <View style={styles.titleRow}>
            <Ionicons name="scale" size={24} color="white" />
            <Text style={styles.title}>Weight</Text>
          </View>
          <TouchableOpacity onPress={onLogPress} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
            <Ionicons name="add-circle" size={28} color="white" />
          </TouchableOpacity>
        </View>
        
        <View style={styles.content}>
          <View style={styles.leftSection}>
            <Text style={styles.currentWeight}>{latestWeight.toFixed(1)} kg</Text>
            {change !== null && (
              <View style={styles.changeContainer}>
                {changeIcon && <Ionicons name={changeIcon} size={16} color="white" />}
                <Text style={styles.changeText}>
                  {change > 0 ? '+' : ''}{change.toFixed(1)} kg
                </Text>
              </View>
            )}
            {goalWeight && (
              <Text style={styles.goalText}>Goal: {goalWeight.toFixed(1)} kg</Text>
            )}
          </View>
          
          <View style={styles.rightSection}>
            {/* Mini sparkline chart */}
            {sparklineData.length > 1 && (
              <View style={styles.sparkline}>
                {/* Implement simple sparkline with SVG or use library */}
                <Text style={styles.sparklineLabel}>Last 7 days</Text>
              </View>
            )}
          </View>
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );
}
```

2. **app/(tabs)/index.tsx** - Update dashboard to include weight widget

Add to imports:
```typescript
import { WeightWidget } from '@/lib/components/WeightWidget';
import { weightApi } from '@/lib/api/weight';
import { WeightRecord } from '@/lib/types';
```

Add state:
```typescript
const [recentWeightRecords, setRecentWeightRecords] = useState<WeightRecord[]>([]);
const [showWeightModal, setShowWeightModal] = useState(false);
```

Update loadSummary to fetch weight data:
```typescript
const loadSummary = useCallback(async () => {
  if (!user) return;
  
  try {
    const [mealsData, supplementsData, trendsData, weightRecords] = await Promise.all([
      mealsApi.getDailySummary(),
      supplementsApi.getTodayStatus().catch(() => null),
      healthMetricsApi.getDashboardData(DASHBOARD_METRICS, 7).catch(() => null),
      weightApi.getWeightRecords(
        new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(), // Last 14 days
        new Date().toISOString()
      ).catch(() => []),
    ]);
    
    setSummary(mealsData);
    setSupplementStatus(supplementsData);
    setHealthTrends(trendsData);
    setRecentWeightRecords(weightRecords.slice(0, 7)); // Last 7 for sparkline
  } catch (error) {
    console.error('Failed to load summary:', error);
  } finally {
    setIsLoading(false);
  }
}, [user]);
```

Add widget to render:
```typescript
{/* After supplements section, before health trends */}
<WeightWidget
  currentWeight={user?.currentWeight}
  goalWeight={user?.goalWeight}
  recentRecords={recentWeightRecords}
  onLogPress={() => setShowWeightModal(true)}
  onPress={() => router.push('/weight')}
  isLoading={isLoading}
/>

{/* Weight entry modal */}
<WeightEntryModal
  visible={showWeightModal}
  onClose={() => setShowWeightModal(false)}
  onSaved={() => {
    setShowWeightModal(false);
    loadSummary(); // Reload data
  }}
  lastWeight={recentWeightRecords[0]?.weight}
/>
```

3. **app/(tabs)/profile.tsx** - Add navigation to weight tracking and goal setup

Add to profile menu items:
```typescript
<TouchableOpacity
  style={styles.menuItem}
  onPress={() => router.push('/weight')}
>
  <Ionicons name="scale" size={24} color={colors.primary} />
  <View style={styles.menuItemContent}>
    <Text style={styles.menuItemTitle}>Weight Tracking</Text>
    {user?.currentWeight && (
      <Text style={styles.menuItemSubtitle}>
        Current: {user.currentWeight.toFixed(1)} kg
        {user.goalWeight && ` • Goal: ${user.goalWeight.toFixed(1)} kg`}
      </Text>
    )}
  </View>
  <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
</TouchableOpacity>

<TouchableOpacity
  style={styles.menuItem}
  onPress={() => router.push('/weight/goal')}
>
  <Ionicons name="flag" size={24} color={colors.primary} />
  <View style={styles.menuItemContent}>
    <Text style={styles.menuItemTitle}>Weight Goal</Text>
    <Text style={styles.menuItemSubtitle}>
      {user?.goalWeight ? `Set your weight goal` : 'No goal set'}
    </Text>
  </View>
  <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
</TouchableOpacity>
```

4. **app/_layout.tsx** - Register weight screens

Add to stack screens:
```typescript
<Stack.Screen name="weight/index" options={{ headerShown: false }} />
<Stack.Screen name="weight/goal" options={{ headerShown: false }} />
<Stack.Screen name="weight/add" options={{ headerShown: false }} />
<Stack.Screen name="weight/edit/[id]" options={{ headerShown: false }} />
```

5. **lib/theme/colors.ts** - Add weight-specific colors

Add to gradients:
```typescript
weight: ['#8B5CF6', '#6366F1'], // Purple to indigo
```

INTEGRATION POINTS:

1. **Dashboard Widget**:
   - Position after supplement tracker, before health trends
   - Show current weight, change from last entry
   - Mini sparkline of last 7 entries
   - Quick-log button (+ icon)
   - Tap to navigate to full weight screen

2. **Profile Menu**:
   - "Weight Tracking" item → /weight
   - "Weight Goal" item → /weight/goal
   - Show current and goal weights in subtitle

3. **User Context Integration**:
   - Weight widget uses user.currentWeight, user.goalWeight
   - Backend syncs WeightRecord → User.currentWeight on create
   - Goal setup updates user.goalWeight

4. **Empty States**:
   - Dashboard widget shows "Track Your Weight" CTA if no data
   - Weight screen shows onboarding message if no entries

ACCESSIBILITY:
- All touchables have accessibilityLabel and accessibilityRole
- Widget announces current weight and change
- Navigation items have descriptive labels

ERROR HANDLING:
- Graceful degradation if weight API fails (don't crash dashboard)
- Loading states with ActivityIndicator
- Error states with retry buttons

TESTING:
- Test dashboard loads weight data
- Test widget navigation to full screen
- Test quick-log modal from widget
- Test profile navigation to weight screens
- Test empty states (no weight data)
- Test loading states
- Test error states (API failures)
