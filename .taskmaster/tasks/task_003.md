# Task ID: 3

**Title:** Build Health Metrics Mobile UI Screens

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Create mobile screens for viewing and manually entering health metrics (RHR, HRV, sleep, recovery). Backend API is complete at /api/health-metrics.

**Details:**

1. Create new screens in `app/` directory:
   - `app/health/index.tsx` - Health metrics dashboard/list
   - `app/health/[id].tsx` - Detail view for specific metric
   - `app/health/add.tsx` - Manual entry form

2. Health Dashboard (`app/health/index.tsx`):
   - Display today's key metrics in cards (RHR, HRV, Sleep, Recovery)
   - Time range selector: Today, Week, Month
   - Pull-to-refresh functionality
   - Navigate to detail view on tap

3. Metric Detail View (`app/health/[id].tsx`):
   - Line chart showing metric over time (use react-native-chart-kit or Victory Native)
   - Statistics: avg, min, max, trend arrow
   - Data source indicator (Apple Health, Fitbit, Manual)
   - Date range filter

4. Manual Entry Form (`app/health/add.tsx`):
   - Metric type picker (dropdown with all HealthMetricType enum values)
   - Value input with unit display (bpm, ms, hours, %)
   - Date/time picker (defaults to now)
   - Source set to 'MANUAL'
   - Validation: min/max ranges per metric type

5. Create API client in `lib/api/health-metrics.ts`:
```typescript
export const healthMetricsApi = {
  getAll: (params: { startDate?: string; endDate?: string; metricType?: string }) => 
    apiClient.get('/health-metrics', { params }),
  getById: (id: string) => apiClient.get(`/health-metrics/${id}`),
  create: (data: CreateHealthMetricInput) => apiClient.post('/health-metrics', data),
  getDailySummary: (date: string) => apiClient.get(`/health-metrics/daily/${date}`),
}
```

6. Add navigation:
   - Add 'Health' tab to bottom navigation in `app/(tabs)/_layout.tsx`
   - Use health heart icon from @expo/vector-icons
<info added on 2025-12-05T01:32:13.931Z>
Now I have a comprehensive understanding of the codebase's styling patterns, typography, colors, and testing conventions. Let me provide the update text:

7. Styling and UX Consistency Requirements:

All Health Metrics screens must follow the established design system in `lib/theme/colors.ts`:
- Use `colors.background.primary` (#0F1419) as main background
- Use `colors.background.tertiary` (#1E2330) for cards and surfaces
- Use `colors.primary.main` (#8B5CF6) for interactive elements
- Apply `gradients.primary` (purple-pink) for CTAs and active states
- Text: `colors.text.primary` for headings, `colors.text.tertiary` for labels
- Use `spacing` constants (xs:4, sm:8, md:16, lg:24, xl:32)
- Apply `borderRadius` constants (sm:8, md:12, lg:16)
- Use `typography.fontSize` and `typography.fontWeight` for consistent text styling
- Import theme tokens: `import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors'`

Match existing UX patterns from `app/(tabs)/index.tsx` and `app/add-meal.tsx`:
- Cards with `borderWidth: 1, borderColor: colors.border.secondary`
- Section titles: `fontSize: typography.fontSize['2xl']` or `lg`, `fontWeight: bold/semibold`
- Form inputs: Height 48px, `backgroundColor: colors.background.tertiary`, `borderRadius: borderRadius.md`
- Pull-to-refresh using `RefreshControl` with `tintColor={colors.primary.main}`
- Loading states with `ActivityIndicator` using `colors.primary.main`
- FAB pattern: 56x56 with LinearGradient and `shadows.xl`
- SafeAreaView container with ScrollView using `showsVerticalScrollIndicator={false}`

8. Comprehensive Test Requirements:

Create mobile component tests in `__tests__/` directory using react-native-testing-library:

Test file structure:
- `__tests__/screens/health/HealthDashboard.test.tsx`
- `__tests__/screens/health/HealthMetricDetail.test.tsx`
- `__tests__/screens/health/AddHealthMetric.test.tsx`
- `__tests__/api/health-metrics.test.ts`

Required test coverage per screen:

HealthDashboard tests:
- Renders loading state with ActivityIndicator
- Renders metric cards for RHR, HRV, Sleep, Recovery when data exists
- Renders empty state when no health data available
- Time range selector changes displayed data (Today/Week/Month)
- Pull-to-refresh triggers API reload
- Tapping metric card navigates to detail view
- Error state rendering when API fails

HealthMetricDetail tests:
- Renders line chart with historical data
- Displays statistics (avg, min, max, trend)
- Shows correct data source indicator (Apple Health, Fitbit, Manual)
- Date range filter updates chart data
- Handles empty data gracefully
- Loading and error states

AddHealthMetric (Manual Entry) tests:
- Metric type picker renders all HealthMetricType enum values
- Value input accepts numeric input with correct units per type
- Date/time picker defaults to current time
- Source automatically set to MANUAL
- Validation errors for out-of-range values (per metric type)
- Required field validation
- Successful submission calls API and navigates back
- Cancel button discards changes and navigates back
- Form disabled during submission

API client tests (`lib/api/health-metrics.ts`):
- Follow existing patterns from `__tests__/unit/api/food-analysis.test.ts`
- Mock axios using `jest.mock('axios')`
- Test getAll with various query params (startDate, endDate, metricType)
- Test getById returns single metric
- Test create sends correct payload and returns created metric
- Test getDailySummary formats date correctly
- Test error handling with proper error messages

Use test patterns from existing tests:
- Arrange, Act, Assert pattern
- Mock dependencies with `jest.mock()`
- Use `waitFor` for async assertions
- Test user interactions with `fireEvent`
</info added on 2025-12-05T01:32:13.931Z>

**Test Strategy:**

1. Component tests for each screen using react-native-testing-library
2. Test form validation for manual entry
3. Test API integration with mock server
4. Visual regression tests for chart rendering
5. Test pull-to-refresh behavior
6. Test empty state when no health data exists

## Subtasks

### 3.1. Create TypeScript types for health metrics in lib/types/health-metrics.ts

**Status:** done  
**Dependencies:** None  

Define TypeScript interfaces and types for health metrics to be used across the mobile app, matching the backend API contracts.

**Details:**

Create `lib/types/health-metrics.ts` with the following types:

1. **HealthMetricType enum** - Match the 28 types from `server/src/validation/schemas.ts` healthMetricTypeSchema: RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN, HEART_RATE_VARIABILITY_RMSSD, BLOOD_PRESSURE_SYSTOLIC, BLOOD_PRESSURE_DIASTOLIC, RESPIRATORY_RATE, OXYGEN_SATURATION, VO2_MAX, SLEEP_DURATION, DEEP_SLEEP_DURATION, REM_SLEEP_DURATION, SLEEP_EFFICIENCY, SLEEP_SCORE, STEPS, ACTIVE_CALORIES, TOTAL_CALORIES, EXERCISE_MINUTES, STANDING_HOURS, RECOVERY_SCORE, STRAIN_SCORE, READINESS_SCORE, BODY_FAT_PERCENTAGE, MUSCLE_MASS, BONE_MASS, WATER_PERCENTAGE, SKIN_TEMPERATURE, BLOOD_GLUCOSE, STRESS_LEVEL

2. **HealthMetricSource type** - Match `server/src/validation/schemas.ts` healthMetricSourceSchema: 'apple_health' | 'fitbit' | 'garmin' | 'oura' | 'whoop' | 'manual'

3. **HealthMetric interface** - Based on backend response: id, userId, metricType, value, unit, recordedAt, source, sourceId?, metadata?, createdAt, updatedAt

4. **CreateHealthMetricInput interface** - Match `server/src/validation/schemas.ts` createHealthMetricSchema: metricType, value, unit, recordedAt (ISO string), source, sourceId?, metadata?

5. **HealthMetricStats interface** - For stats endpoint: average, min, max, count, trend ('up' | 'down' | 'stable'), percentChange

6. **TimeSeriesDataPoint interface** - For charts: date (string), value, source?

7. **METRIC_CONFIG constant** - Unit and display info per metric type: { unit: string, displayName: string, minValue?: number, maxValue?: number, icon?: string }. Example: RESTING_HEART_RATE: { unit: 'bpm', displayName: 'Resting Heart Rate', minValue: 30, maxValue: 220 }

### 3.2. Create health metrics API client in lib/api/health-metrics.ts

**Status:** done  
**Dependencies:** 3.1  

Implement API client functions to communicate with the backend health metrics endpoints, following the existing mealsApi pattern in lib/api/meals.ts.

**Details:**

Create `lib/api/health-metrics.ts` following the pattern from `lib/api/meals.ts`:

```typescript
import api from './client';
import { HealthMetric, CreateHealthMetricInput, HealthMetricStats, TimeSeriesDataPoint, HealthMetricType, HealthMetricSource } from '../types/health-metrics';

export interface GetHealthMetricsParams {
  metricType?: HealthMetricType;
  startDate?: string;
  endDate?: string;
  source?: HealthMetricSource;
  limit?: number;
}

export const healthMetricsApi = {
  // POST /health-metrics - Create single metric
  async create(data: CreateHealthMetricInput): Promise<HealthMetric>,

  // GET /health-metrics - Get all with optional filters
  async getAll(params?: GetHealthMetricsParams): Promise<HealthMetric[]>,

  // GET /health-metrics/:id - Get by ID
  async getById(id: string): Promise<HealthMetric>,

  // GET /health-metrics/latest/:metricType - Get latest value for a metric type
  async getLatest(metricType: HealthMetricType): Promise<HealthMetric | null>,

  // GET /health-metrics/timeseries/:metricType - Get time series data for charts
  async getTimeSeries(metricType: HealthMetricType, startDate?: string, endDate?: string): Promise<TimeSeriesDataPoint[]>,

  // GET /health-metrics/stats/:metricType - Get statistics (avg, min, max, trend)
  async getStats(metricType: HealthMetricType, days?: number): Promise<HealthMetricStats>,

  // GET /health-metrics/average/daily/:metricType - Get daily average
  async getDailyAverage(metricType: HealthMetricType, date?: string): Promise<{ average: number; count: number }>,

  // GET /health-metrics/average/weekly/:metricType - Get weekly average
  async getWeeklyAverage(metricType: HealthMetricType): Promise<{ average: number; count: number }>,

  // DELETE /health-metrics/:id - Delete metric
  async delete(id: string): Promise<void>,
};
```

Use the existing `api` client from `./client` which handles JWT auth token injection. Match the routes from `server/src/routes/healthMetricRoutes.ts`.

### 3.3. Add Health tab to bottom navigation in app/(tabs)/_layout.tsx

**Status:** done  
**Dependencies:** None  

Add a new 'Health' tab to the existing tab navigation using a heart icon, maintaining consistency with the current tab styling.

**Details:**

Modify `app/(tabs)/_layout.tsx` to add the Health tab:

1. Add the 'heart.fill' SF Symbol to the MAPPING in `components/ui/IconSymbol.tsx`:
```typescript
'heart.fill': 'favorite',  // MaterialIcons mapping
'person.fill': 'person',   // Already exists
```

2. Add new Tabs.Screen in `app/(tabs)/_layout.tsx` after the index tab and before profile:
```typescript
<Tabs.Screen
  name="health"
  options={{
    title: 'Health',
    tabBarIcon: ({ color }) => <IconSymbol size={28} name="heart.fill" color={color} />,
  }}
/>
```

3. Ensure the tab follows existing styling from tabBarStyle with:
- `tabBarActiveTintColor: colors.primary.main` (purple #8B5CF6)
- `tabBarInactiveTintColor: colors.text.disabled` (gray #6B7280)
- Same height and padding as other tabs

4. Export IconSymbolName type must include 'heart.fill' for TypeScript safety.

### 3.4. Create Health Dashboard screen at app/(tabs)/health.tsx

**Status:** done  
**Dependencies:** 3.1, 3.2, 3.3  

Build the main Health Dashboard screen displaying today's key metrics (RHR, HRV, Sleep, Recovery) in cards with time range selector and pull-to-refresh.

**Details:**

Create `app/(tabs)/health.tsx` following patterns from `app/(tabs)/index.tsx`:

**Imports:**
- React hooks: useState, useEffect, useCallback
- Components: View, Text, StyleSheet, ScrollView, TouchableOpacity, RefreshControl, ActivityIndicator
- expo-router: useRouter
- SafeAreaView from react-native-safe-area-context
- LinearGradient from expo-linear-gradient
- Theme: colors, gradients, shadows, spacing, borderRadius, typography from '@/lib/theme/colors'
- API: healthMetricsApi from '@/lib/api/health-metrics'
- Types: HealthMetric, HealthMetricStats, METRIC_CONFIG from '@/lib/types/health-metrics'

**State:**
- metrics: Record<HealthMetricType, { latest: HealthMetric | null; stats: HealthMetricStats | null }>
- timeRange: 'today' | 'week' | 'month' (default 'today')
- isLoading: boolean, refreshing: boolean

**Layout Structure:**
1. Header with title "Health" and date (matches index.tsx greeting style: fontSize: typography.fontSize['3xl'], fontWeight: bold)
2. Time Range Selector (horizontal buttons like meal type selector in add-meal.tsx)
3. Metric Cards Grid (2x2) for: RESTING_HEART_RATE, HEART_RATE_VARIABILITY_SDNN, SLEEP_DURATION, RECOVERY_SCORE

**Each Metric Card (TouchableOpacity):**
- backgroundColor: colors.background.tertiary
- borderWidth: 1, borderColor: colors.border.secondary
- borderRadius: borderRadius.lg
- padding: spacing.md
- Icon (use Ionicons: heart-outline, pulse-outline, moon-outline, fitness-outline)
- Label (colors.text.tertiary, fontSize: typography.fontSize.sm)
- Value (colors.text.primary, fontSize: typography.fontSize['2xl'], fontWeight: bold)
- Unit (colors.text.tertiary)
- Trend indicator arrow (green up, red down, gray stable)
- onPress: router.push(`/health/${metricType}`)

**Pull-to-refresh:** Use RefreshControl with tintColor={colors.primary.main}

**Loading State:** ActivityIndicator centered with colors.primary.main

**Empty State:** "No health data yet. Add your first metric!" with button to /health/add

### 3.5. Create Metric Detail screen at app/health/[metricType].tsx

**Status:** done  
**Dependencies:** 3.1, 3.2  

Build the detail view for a specific health metric showing historical line chart, statistics (avg, min, max, trend), date range filter, and data source indicator.

**Details:**

Create `app/health/[metricType].tsx` as a dynamic route screen:

**Imports:**
- useLocalSearchParams from expo-router to get metricType param
- LineChart from react-native-chart-kit (install if needed) or VictoryLine from victory-native
- All theme imports from lib/theme/colors
- healthMetricsApi and types

**State:**
- timeSeries: TimeSeriesDataPoint[] for chart data
- stats: HealthMetricStats | null
- dateRange: '7d' | '30d' | '90d' (default '30d')
- isLoading: boolean

**Layout:**
1. **Header** with back button (TouchableOpacity with Ionicons chevron-back) and metric display name from METRIC_CONFIG

2. **Date Range Selector** - Horizontal buttons matching add-meal.tsx mealTypeContainer style:
   - 7 Days, 30 Days, 90 Days
   - Active: LinearGradient with gradients.primary
   - Inactive: backgroundColor: colors.background.tertiary

3. **Line Chart** (full width, height ~200):
   - backgroundColor: colors.background.tertiary
   - Line color: colors.primary.main (#8B5CF6)
   - Grid lines: colors.border.secondary
   - Labels: colors.text.tertiary
   - Data points from timeSeries API response

4. **Statistics Card** (similar to macrosContainer in index.tsx):
   - Three columns: Average, Minimum, Maximum
   - Each shows value with unit
   - fontSize: typography.fontSize.xl for values
   - backgroundColor: colors.background.tertiary
   - borderRadius: borderRadius.md

5. **Trend Section:**
   - Arrow icon (trending-up/down/minus from Ionicons)
   - Percentage change text
   - Color: status.success (green) for up, status.error (red) for down

6. **Data Source Badge:**
   - Icon per source (Apple Health, Fitbit, Manual, etc.)
   - Text showing source name
   - colors.text.tertiary styling

**Chart Config (react-native-chart-kit):**
```typescript
chartConfig: {
  backgroundColor: colors.background.tertiary,
  backgroundGradientFrom: colors.background.tertiary,
  backgroundGradientTo: colors.background.tertiary,
  color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`, // primary.main
  labelColor: (opacity = 1) => `rgba(156, 163, 175, ${opacity})`, // text.tertiary
  strokeWidth: 2,
}
```

### 3.6. Create Manual Entry form at app/health/add.tsx

**Status:** done  
**Dependencies:** 3.1, 3.2  

Build the form for manually entering health metrics with metric type picker, value input with dynamic units, date/time picker, and validation for min/max ranges per metric type.

**Details:**

Create `app/health/add.tsx` following the pattern from `app/add-meal.tsx`:

**Imports:**
- useState, useEffect from react
- All RN components: View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, KeyboardAvoidingView, Platform, ActivityIndicator
- useRouter from expo-router
- SafeAreaView, LinearGradient, Ionicons
- DateTimePicker from @react-native-community/datetimepicker (or expo-date-time-picker)
- Picker from @react-native-picker/picker
- healthMetricsApi, CreateHealthMetricInput, HealthMetricType, METRIC_CONFIG
- colors, gradients, spacing, borderRadius, typography from theme
- showAlert from '@/lib/utils/alert'
- getErrorMessage from '@/lib/utils/errorHandling'

**State:**
- metricType: HealthMetricType (default 'RESTING_HEART_RATE')
- value: string (for TextInput)
- recordedAt: Date (default new Date())
- isLoading: boolean
- showDatePicker: boolean
- errors: { value?: string }

**Layout (match add-meal.tsx structure):**
1. **Header** - Same as add-meal: Cancel (left), "Add Health Metric" (center), Save (right)
   - Cancel: text style, color: colors.text.secondary
   - Save: text style, color: colors.primary.main, disabled when isLoading

2. **Metric Type Picker Section:**
   - Label: "Metric Type" (styles.sectionTitle)
   - Dropdown picker with all 28 HealthMetricType values
   - Group by category: Cardiovascular, Sleep, Activity, Recovery, Body Composition
   - Display friendly names from METRIC_CONFIG.displayName

3. **Value Input Section:**
   - Label: "Value ({unit})" - dynamically show unit from METRIC_CONFIG[metricType].unit
   - TextInput with keyboardType="decimal-pad"
   - Height 48px, backgroundColor: colors.background.tertiary
   - Validation message below if out of range (colors.status.error)

4. **Date/Time Picker Section:**
   - Label: "Recorded At"
   - TouchableOpacity showing formatted date/time
   - Opens DateTimePicker modal
   - Default to current time

5. **Source Badge** (non-editable):
   - Shows "Manual" with checkmark icon
   - Subtle styling: colors.special.highlight background

**Validation:**
- Value required and numeric
- Check against METRIC_CONFIG[metricType].minValue and maxValue
- Show inline error: "Value must be between {min} and {max} {unit}"

**handleSave:**
```typescript
const data: CreateHealthMetricInput = {
  metricType,
  value: parseFloat(value),
  unit: METRIC_CONFIG[metricType].unit,
  recordedAt: recordedAt.toISOString(),
  source: 'manual',
};
await healthMetricsApi.create(data);
showAlert('Success', 'Health metric added!', [{ text: 'OK', onPress: () => router.back() }]);
```

### 3.7. Write unit tests for health-metrics API client

**Status:** done  
**Dependencies:** 3.2  

Create comprehensive unit tests for lib/api/health-metrics.ts following the existing test patterns from __tests__/unit/api/food-analysis.test.ts.

**Details:**

Create `__tests__/unit/api/health-metrics.test.ts`:

**Setup:**
```typescript
import axios from 'axios';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import { HealthMetric, HealthMetricType, HealthMetricStats, TimeSeriesDataPoint } from '@/lib/types/health-metrics';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;
```

**Test Fixtures:**
```typescript
const mockHealthMetric: HealthMetric = {
  id: 'metric-1',
  userId: 'user-1',
  metricType: 'RESTING_HEART_RATE',
  value: 62,
  unit: 'bpm',
  recordedAt: '2024-01-15T08:00:00Z',
  source: 'manual',
  createdAt: '2024-01-15T08:00:00Z',
  updatedAt: '2024-01-15T08:00:00Z',
};

const mockStats: HealthMetricStats = {
  average: 65,
  min: 58,
  max: 72,
  count: 30,
  trend: 'down',
  percentChange: -3.5,
};
```

**Test Cases:**

1. **create():**
   - Should successfully create a health metric
   - Should send correct payload format to POST /health-metrics
   - Should handle validation errors (400)
   - Should handle auth errors (401)

2. **getAll():**
   - Should return array of metrics
   - Should pass query params (metricType, startDate, endDate, source, limit)
   - Should handle empty results
   - Should handle network errors

3. **getById():**
   - Should return single metric by ID
   - Should handle 404 not found

4. **getLatest():**
   - Should return latest metric for type
   - Should return null when no metrics exist

5. **getTimeSeries():**
   - Should return array of data points for chart
   - Should pass date range params
   - Should handle empty data gracefully

6. **getStats():**
   - Should return stats with average, min, max, trend
   - Should pass days param
   - Should handle 404 when no data

7. **getDailyAverage() and getWeeklyAverage():**
   - Should return average and count
   - Should handle date param for daily

8. **delete():**
   - Should successfully delete metric
   - Should handle 404 errors

**Pattern:** Use Arrange, Act, Assert. Mock `api.get`, `api.post`, `api.delete` from the client module.

### 3.8. Write component tests for Health screens

**Status:** done  
**Dependencies:** 3.4, 3.5, 3.6  

Create comprehensive component tests for HealthDashboard, HealthMetricDetail, and AddHealthMetric screens using react-native-testing-library.

**Details:**

Create test files in `__tests__/screens/health/` directory:

**1. __tests__/screens/health/HealthDashboard.test.tsx:**
```typescript
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import HealthDashboard from '@/app/(tabs)/health';
import { healthMetricsApi } from '@/lib/api/health-metrics';

jest.mock('@/lib/api/health-metrics');
jest.mock('expo-router', () => ({ useRouter: () => ({ push: jest.fn() }) }));
jest.mock('@/lib/context/AuthContext', () => ({ useAuth: () => ({ user: { id: '1', name: 'Test' } }) }));
```

Test cases:
- Renders loading state with ActivityIndicator initially
- Renders metric cards when data loads successfully
- Renders empty state when no health data available
- Time range selector updates displayed data (Today/Week/Month)
- Pull-to-refresh triggers API reload (test RefreshControl onRefresh)
- Tapping metric card calls router.push with correct metric type
- Error state renders when API fails

**2. __tests__/screens/health/HealthMetricDetail.test.tsx:**

Test cases:
- Renders line chart with historical data points
- Displays statistics (avg, min, max, trend arrow, percentage)
- Shows correct data source indicator icon/text
- Date range filter buttons update chart data (7d/30d/90d)
- Handles empty data gracefully with message
- Loading state shows ActivityIndicator
- Error state with retry button

**3. __tests__/screens/health/AddHealthMetric.test.tsx:**

Test cases:
- Metric type picker renders all HealthMetricType enum values
- Value input accepts numeric input and shows correct unit dynamically
- Date/time picker defaults to current time and opens modal
- Source automatically displays "Manual"
- Shows validation error for out-of-range values (test per metric type bounds)
- Shows required field validation when value empty
- Successful submission calls API and navigates back
- Cancel button navigates back without saving
- Form inputs disabled during submission (isLoading state)
- Shows loading indicator on Save button during submission

**Common Mocks:**
- Mock @react-native-community/datetimepicker
- Mock @react-native-picker/picker
- Mock react-native-chart-kit or victory-native
- Mock expo-linear-gradient
