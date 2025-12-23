# Task ID: 45

**Title:** Goal Progress Dashboard & Achievement Tracking

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement comprehensive goal progress visualization dashboard showing daily/weekly/monthly progress towards nutrition, weight, and health goals with visual indicators, trend analysis, and motivational feedback.

**Details:**

No details provided.

**Test Strategy:**

No test strategy provided.

## Subtasks

### 45.1. Create backend goal progress calculation service with historical tracking

**Status:** pending  
**Dependencies:** None  

Implement backend service methods to calculate daily, weekly, and monthly goal progress for calories and macros with trend analysis and percentage completion calculations.

**Details:**

Create new methods in `server/src/services/mealService.ts` to calculate goal progress metrics:

1. `getGoalProgress(userId, date, period)` - Calculate progress for specified period (daily/weekly/monthly)
2. `getGoalTrends(userId, days)` - Get historical goal achievement data for trend charts
3. Calculate percentage completion for calories, protein, carbs, fat vs user goals
4. Return streak data (consecutive days meeting goals)
5. Calculate weekly/monthly averages and compare to goals
6. Add achievement status (under/on-track/over)

Return structure:
```typescript
{
  period: 'daily' | 'weekly' | 'monthly',
  calories: { consumed: number, goal: number, percentage: number, status: string },
  protein: { consumed: number, goal: number, percentage: number, status: string },
  carbs: { consumed: number, goal: number, percentage: number, status: string },
  fat: { consumed: number, goal: number, percentage: number, status: string },
  streak: { days: number, type: 'all_goals' | 'calories_only' },
  trend: Array<{ date: string, caloriesPercentage: number, proteinPercentage: number }>
}
```

Follow existing pattern in `mealService.ts:118-153` for getDailySummary implementation.

### 45.2. Add goal progress API endpoints with date range filtering

**Status:** pending  
**Dependencies:** 45.1  

Create REST API endpoints for retrieving goal progress data with support for daily, weekly, and monthly views, including trend history endpoints.

**Details:**

Add new routes in `server/src/routes/mealRoutes.ts` following existing pattern at line 12:

```typescript
// GET /api/meals/goals/progress?period=daily&date=2024-01-15
router.get('/goals/progress', (req, res) => mealController.getGoalProgress(req, res));

// GET /api/meals/goals/trends?days=7
router.get('/goals/trends', (req, res) => mealController.getGoalTrends(req, res));

// GET /api/meals/goals/streak
router.get('/goals/streak', (req, res) => mealController.getGoalStreak(req, res));
```

Create controller methods in `server/src/controllers/mealController.ts`:
- Validate query parameters (period: daily/weekly/monthly, date, days: 7/30/90)
- Call corresponding service methods
- Return formatted response with HTTP 200
- Handle errors with proper HTTP status codes

Add Zod validation schemas in `server/src/validation/schemas.ts` for query parameters.

### 45.3. Create reusable CircularProgress and GoalCard components

**Status:** pending  
**Dependencies:** None  

Build reusable React Native components for displaying circular progress rings and goal cards with animations, following existing design patterns in app/(tabs)/index.tsx.

**Details:**

Create `lib/components/CircularProgress.tsx`:
- Circular progress ring using react-native-svg (already installed)
- Props: size, strokeWidth, progress (0-100), color, backgroundColor, label, value
- Animated progress with react-native-reanimated
- Support gradient colors like existing calorieRing (lines 195-211 in index.tsx)
- Responsive sizing using useResponsive hook

Create `lib/components/GoalCard.tsx`:
- Card displaying goal progress with CircularProgress component
- Props: title, current, goal, unit, progressColor, icon
- Show percentage and "On Track" / "Under" / "Over" status
- Match design of existing macroCard (lines 522-556 in index.tsx)
- Include LinearGradient for progress bars
- Support onPress for navigation to details

Follow existing patterns:
- Use colors, gradients, shadows from `lib/theme/colors.ts`
- Use spacing, borderRadius, typography from theme
- Add responsive design support
- Include testID props for E2E testing

### 45.4. Integrate goal progress cards into dashboard home screen

**Status:** pending  
**Dependencies:** 45.2, 45.3  

Add goal progress visualization section to app/(tabs)/index.tsx dashboard, displaying daily progress for calories and macros with trend indicators.

**Details:**

Modify `app/(tabs)/index.tsx` to add Goal Progress section:

1. Create API client method in `lib/api/meals.ts`:
```typescript
getGoalProgress: async (period: 'daily' | 'weekly' | 'monthly', date?: Date) => {
  const response = await client.get('/meals/goals/progress', { params: { period, date } });
  return response.data;
}
```

2. Add state and data fetching in index.tsx (around line 44):
```typescript
const [goalProgress, setGoalProgress] = useState<GoalProgress | null>(null);
// Fetch in loadSummary() alongside existing calls (line 84)
```

3. Insert Goal Progress section after Macros Container (after line 281):
- Section header: "Daily Goals" with View All button
- 2x2 grid of GoalCard components (Calories, Protein, Carbs, Fat)
- Each card shows CircularProgress with current/goal values
- Use existing isTablet responsive patterns (lines 227, 296)
- Match styles from trendsSection (lines 559-633)

4. Add navigation to detailed goals screen (next subtask) on card press

5. Refresh on pull-to-refresh and screen focus (existing pattern lines 109-115)

### 45.5. Build detailed goals screen with weekly/monthly trends and achievement history

**Status:** pending  
**Dependencies:** 45.2, 45.3  

Create comprehensive goals tracking screen at app/goals.tsx with line charts showing progress trends, achievement streaks, and historical performance using react-native-chart-kit.

**Details:**

Create new screen `app/goals.tsx`:

1. Setup screen structure:
- Custom header with title "Goal Progress" (headerShown: false pattern)
- Period selector: Daily / Weekly / Monthly tabs
- ScrollView with sections

2. Sections:
- **Summary Cards**: 4 GoalCard components in 2x2 grid (Calories, Protein, Carbs, Fat) with current period progress
- **Trend Chart**: LineChart from react-native-chart-kit showing 7-day/30-day/90-day progress trends for all macros
- **Achievement Streak**: Card showing consecutive days meeting goals with flame icon
- **Weekly Breakdown**: Bar chart showing daily performance for the week
- **Goal History**: List of past achievements with dates and percentages

3. Chart configuration using react-native-chart-kit:
```typescript
import { LineChart, BarChart } from 'react-native-chart-kit';
// Use existing color scheme from lib/theme/colors.ts
// Follow responsive patterns from index.tsx
```

4. Data fetching:
- Use getGoalProgress and getGoalTrends API calls
- Update on period change (Daily/Weekly/Monthly)
- Pull-to-refresh support

5. Register route in `app/_layout.tsx` with headerShown: false

6. Add navigation from dashboard "View All" button and GoalCard presses
