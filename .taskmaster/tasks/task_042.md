# Task ID: 42

**Title:** Water Tracking Mobile UI with Daily Goals and Reminders

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement complete water tracking functionality including mobile UI screens, backend API endpoints, daily goal management, quick-add interactions, and hydration reminders integration. The WaterIntake Prisma model already exists but has no API or UI implementation.

**Details:**

## Current State Analysis

### Existing Infrastructure:
- **Prisma Model** (`server/prisma/schema.prisma` lines 120-130):
  ```prisma
  model WaterIntake {
    id        String   @id @default(cuid())
    userId    String
    user      User     @relation(fields: [userId], references: [id], onDelete: Cascade)
    amount    Float    // in ml
    recordedAt DateTime @default(now())
    createdAt DateTime @default(now())
    @@index([userId, recordedAt])
  }
  ```
- **User Model**: Has `waterIntakes WaterIntake[]` relation
- **NO backend API endpoints** for water tracking
- **NO mobile UI screens** for water tracking
- **NO water goal field** in User model (need to add)

### Required Schema Updates

Add to User model in `server/prisma/schema.prisma`:
```prisma
// Water tracking
goalWaterMl       Int      @default(2500)  // Daily water goal in ml (default 2.5L)
waterReminderEnabled Boolean @default(false)
waterReminderInterval Int?   // Minutes between reminders (e.g., 60, 90, 120)
```

## Backend API Implementation

### 1. Water Service (`server/src/services/waterService.ts`)

```typescript
interface WaterIntakeInput {
  amount: number;      // ml
  recordedAt?: Date;   // defaults to now
}

interface DailyWaterSummary {
  date: string;
  totalMl: number;
  goalMl: number;
  percentComplete: number;
  entries: WaterIntake[];
}

class WaterService {
  // Create water intake entry
  async createWaterIntake(userId: string, data: WaterIntakeInput): Promise<WaterIntake>;
  
  // Get daily summary with goal progress
  async getDailySummary(userId: string, date: string): Promise<DailyWaterSummary>;
  
  // Get entries for date range
  async getWaterIntakes(userId: string, startDate: Date, endDate: Date): Promise<WaterIntake[]>;
  
  // Delete entry
  async deleteWaterIntake(userId: string, entryId: string): Promise<void>;
  
  // Update user's daily water goal
  async updateWaterGoal(userId: string, goalMl: number): Promise<User>;
  
  // Get weekly/monthly trends
  async getWaterTrends(userId: string, period: 'week' | 'month'): Promise<WaterTrend[]>;
}
```

### 2. Water Controller (`server/src/controllers/waterController.ts`)

Endpoints:
- `POST /api/water` - Log water intake
- `GET /api/water/today` - Get today's summary with goal progress
- `GET /api/water/daily/:date` - Get specific day's summary
- `GET /api/water?startDate=&endDate=` - Get entries for range
- `DELETE /api/water/:id` - Delete entry
- `PUT /api/user/water-goal` - Update water goal
- `GET /api/water/trends?period=week|month` - Get trends

### 3. Validation Schemas (`server/src/validation/schemas.ts`)

```typescript
export const createWaterIntakeSchema = z.object({
  amount: z.number().min(1).max(5000), // 1ml to 5L per entry
  recordedAt: z.string().datetime().optional(),
});

export const updateWaterGoalSchema = z.object({
  goalMl: z.number().min(500).max(10000), // 0.5L to 10L
});

export const getWaterTrendsSchema = z.object({
  period: z.enum(['week', 'month']),
});
```

### 4. Routes (`server/src/routes/waterRoutes.ts`)

- Apply `requireAuth` middleware to all routes
- Apply rate limiting (100 req/15min)
- Register in `server/src/index.ts`

## Mobile App Implementation

### 1. API Client (`lib/api/water.ts`)

```typescript
export const waterApi = {
  logWater: (amount: number, recordedAt?: string) => 
    apiClient.post('/water', { amount, recordedAt }),
  
  getTodaySummary: () => 
    apiClient.get('/water/today'),
  
  getDailySummary: (date: string) => 
    apiClient.get(`/water/daily/${date}`),
  
  getWaterIntakes: (startDate: string, endDate: string) => 
    apiClient.get('/water', { params: { startDate, endDate } }),
  
  deleteEntry: (id: string) => 
    apiClient.delete(`/water/${id}`),
  
  updateGoal: (goalMl: number) => 
    apiClient.put('/user/water-goal', { goalMl }),
  
  getTrends: (period: 'week' | 'month') => 
    apiClient.get('/water/trends', { params: { period } }),
};
```

### 2. Water Tracking Screen (`app/water.tsx`)

Main screen with:
- **Circular Progress Ring**: Visual representation of daily progress
  - Shows current ml / goal ml
  - Percentage complete
  - Animated fill as water is logged
  - Color changes: blue → light blue → green when goal reached
  
- **Quick Add Buttons Grid**:
  - 150ml (small glass)
  - 250ml (glass/cup)
  - 500ml (bottle)
  - 750ml (large bottle)
  - Custom amount button (opens modal)
  
- **Today's Log Section**:
  - Scrollable list of today's entries
  - Each entry shows: time, amount, delete button
  - Swipe to delete (use existing SwipeableCard pattern)
  
- **Weekly Overview Chart**:
  - Bar chart showing last 7 days
  - Goal line overlay
  - Tap day to see details

### 3. Water Goal Settings (`app/water-settings.tsx`)

- Daily goal input with slider (500ml - 5000ml)
- Preset goals: 2L, 2.5L, 3L, 3.5L, 4L
- Goal calculator based on:
  - Body weight (0.033L per kg)
  - Activity level adjustment
  - Climate/season adjustment
- Reminder settings:
  - Enable/disable toggle
  - Interval picker (30min, 1hr, 1.5hr, 2hr)
  - Quiet hours (sleep time exclusion)
  - Integration with push notifications (Task 40)

### 4. Water Widget Component (`lib/components/WaterWidget.tsx`)

Compact widget for dashboard (`app/(tabs)/index.tsx`):
- Small circular progress indicator
- Current ml / goal
- Quick +250ml button
- Tap to open full water screen

### 5. Custom Amount Modal (`lib/components/WaterAmountModal.tsx`)

- Number input with +/- buttons
- Common amounts as chips: 100, 200, 300, 400, 500ml
- Unit toggle: ml / oz / cups
- Keyboard-friendly numeric input

## UI/UX Design Specifications

### Design Tokens (from `lib/theme/colors.ts`):
```typescript
water: {
  background: colors.background.tertiary,
  progressEmpty: '#1E3A5F',      // Dark blue
  progressFill: '#3B82F6',        // Bright blue
  progressComplete: '#10B981',    // Green when goal met
  quickAddButton: colors.background.secondary,
  quickAddButtonPressed: colors.primary.main,
}
```

### Animations:
- Progress ring fills with spring animation on log
- Ripple effect on quick add buttons
- Celebration animation when goal reached (confetti or pulse)
- Entry items slide in with stagger

### Accessibility:
- VoiceOver: "Water tracking, 1500 milliliters of 2500 milliliter goal, 60 percent complete"
- Quick add buttons have clear labels
- Color contrast for all states

## Integration Points

### 1. Dashboard Widget:
Add to `app/(tabs)/index.tsx` between existing cards:
```tsx
<WaterWidget onPress={() => router.push('/water')} />
```

### 2. Navigation:
- Add route to `app/_layout.tsx` with `headerShown: false`
- Add "Water" row in profile settings linking to water-settings
- Optional: Add Water as a tab (but may clutter - recommend modal access)

### 3. Push Notification Integration (Task 40):
- Hydration reminders as WATER_REMINDER category
- Smart reminders based on:
  - Time since last log
  - Current progress vs goal
  - User's typical drinking patterns

### 4. HealthKit Integration (Task 5):
- Write water data to HealthKit (HKQuantityTypeIdentifierDietaryWater)
- Read water from HealthKit if user logs elsewhere
- Deduplicate entries

### 5. ML Insights (Task 8):
- Correlate hydration with:
  - Energy levels
  - Sleep quality
  - Exercise performance
- Generate insights: "You drink 30% more water on workout days"

## Unit Conversions

Support user preference for units:
```typescript
const CONVERSIONS = {
  ml_to_oz: 0.033814,
  oz_to_ml: 29.5735,
  ml_to_cups: 0.00422675,
  cups_to_ml: 236.588,
};

// Store in ml, display in user's preferred unit
// Add `waterUnit` field to User model: 'ml' | 'oz' | 'cups'
```

## Error Handling

Follow existing patterns from `lib/utils/errorHandling.ts`:
- Network errors: Show retry button
- Validation errors: Inline field errors
- Server errors: Alert with generic message
- Optimistic updates: Log immediately, rollback on failure

## Performance Considerations

- Cache today's summary in memory
- Debounce frequent logging (if user taps rapidly)
- Lazy load weekly chart data
- Prefetch trends when entering screen

**Test Strategy:**

## Testing Strategy

### Backend Unit Tests (`server/src/__tests__/water.test.ts`)

1. **Service Tests**:
   - createWaterIntake: Valid input creates entry
   - createWaterIntake: Invalid amount (negative, too large) throws
   - getDailySummary: Returns correct total and percentage
   - getDailySummary: Handles no entries (0%)
   - getDailySummary: Respects timezone for "today"
   - deleteWaterIntake: Only owner can delete
   - updateWaterGoal: Validates min/max bounds
   - getWaterTrends: Aggregates correctly for week/month

2. **Controller Tests**:
   - POST /api/water: 201 on success
   - POST /api/water: 400 on invalid amount
   - POST /api/water: 401 without auth
   - GET /api/water/today: Returns summary
   - DELETE /api/water/:id: 404 for non-existent
   - DELETE /api/water/:id: 403 for other user's entry

3. **Integration Tests**:
   - Full flow: Log multiple entries → Get summary → Verify total
   - Goal update → Summary reflects new goal percentage

### Mobile Tests

1. **Component Tests** (`__tests__/components/WaterWidget.test.tsx`):
   - Renders current progress correctly
   - Quick add button calls API
   - Tap navigates to full screen

2. **Screen Tests** (`__tests__/screens/WaterScreen.test.tsx`):
   - Loads today's summary on mount
   - Quick add buttons log correct amounts
   - Custom amount modal validates input
   - Delete entry removes from list
   - Progress ring animates on log
   - Goal reached shows celebration

3. **API Client Tests** (`__tests__/api/water.test.ts`):
   - Mock axios, verify correct endpoints called
   - Error handling for network failures

### E2E Tests (Maestro)

1. **Water Logging Flow** (`e2e/tests/water/log_water.yaml`):
   - Open water screen from dashboard
   - Tap 250ml quick add
   - Verify progress updates
   - Add custom amount
   - Verify total increases

2. **Water Goal Flow** (`e2e/tests/water/water_goal.yaml`):
   - Open water settings
   - Change goal to 3000ml
   - Verify home screen reflects new goal

### Manual Testing Checklist

- [ ] Progress ring fills smoothly
- [ ] Quick add haptic feedback works
- [ ] Goal completion celebration triggers
- [ ] Entries appear in correct order (newest first)
- [ ] Swipe to delete works
- [ ] Custom amount keyboard is numeric
- [ ] Unit conversion displays correctly
- [ ] Offline logging queues correctly
- [ ] HealthKit sync (if implemented)

## Subtasks

### 42.1. Update Prisma schema and create water service backend layer

**Status:** pending  
**Dependencies:** None  

Add water goal fields to User model, create WaterService class following mealService.ts pattern with CRUD operations and daily/weekly summary methods

**Details:**

1. Update `server/prisma/schema.prisma` User model:
   - Add `goalWaterMl Int @default(2500)` (daily water goal in ml, default 2.5L)
   - Add `waterReminderEnabled Boolean @default(false)`
   - Add `waterReminderInterval Int?` (minutes between reminders)

2. Run database migration:
   - `cd server && npm run db:generate`
   - `cd server && npm run db:push`

3. Create `server/src/services/waterService.ts` following mealService.ts pattern:
   - Class-based singleton export: `export class WaterService` + `export const waterService = new WaterService()`
   - Methods:
     - `createWaterIntake(userId: string, data: { amount: number, recordedAt?: Date })` - Create entry
     - `getWaterIntakes(userId: string, date?: Date)` - Get entries for day using getDayBoundaries
     - `deleteWaterIntake(userId: string, entryId: string)` - Delete entry with ownership check
     - `getDailySummary(userId: string, date?: Date)` - Return { totalMl, goalMl, percentComplete, entries }
     - `getWeeklySummary(userId: string)` - Last 7 days with averages using getDaysAgo(WEEK_IN_DAYS)
     - `updateWaterGoal(userId: string, goalMl: number)` - Update user.goalWaterMl
   - Use prisma for database access
   - Use getDayBoundaries, getDaysAgo from utils/dateHelpers
   - Use USER_GOALS_SELECT_FIELDS pattern for user data
   - Follow exact structure of mealService.ts

### 42.2. Create water controller with Zod validation and API routes

**Status:** pending  
**Dependencies:** 42.1  

Implement WaterController class following mealController.ts pattern with withErrorHandling wrapper, create Zod schemas, and set up Express routes

**Details:**

1. Add Zod schemas to `server/src/validation/schemas.ts`:
   ```typescript
   export const createWaterIntakeSchema = z.object({
     amount: z.number().min(1).max(5000),
     recordedAt: datetimeSchema.optional(),
   });
   export const updateWaterGoalSchema = z.object({
     goalMl: z.number().min(500).max(10000),
   });
   ```

2. Create `server/src/controllers/waterController.ts` following mealController.ts pattern:
   - Import waterService, requireAuth, withErrorHandling, ErrorHandlers
   - Class-based with singleton export
   - Methods:
     - `createWaterIntake` - withErrorHandling, requireAuth, validate with createWaterIntakeSchema
     - `getWaterIntakes` - Get entries for date (query param)
     - `deleteWaterIntake` - ErrorHandlers.withNotFound for 404 handling
     - `getDailySummary` - Optional date query param
     - `getWeeklySummary` - No params
     - `updateWaterGoal` - Validate with updateWaterGoalSchema
   - Use HTTP_STATUS constants (OK=200, CREATED=201)
   - Use parseOptionalDate for date query params

3. Create `server/src/routes/waterRoutes.ts` following mealRoutes.ts:
   - Apply authenticate middleware to all routes
   - Routes:
     - `POST /` - createWaterIntake
     - `GET /` - getWaterIntakes
     - `GET /summary/daily` - getDailySummary
     - `GET /summary/weekly` - getWeeklySummary
     - `DELETE /:id` - deleteWaterIntake
     - `PUT /goal` - updateWaterGoal (user goal, not entry-specific)
   - Export default router

4. Register routes in `server/src/index.ts`:
   - Import waterRoutes
   - Add `app.use('/api/water', waterRoutes)`

### 42.3. Create mobile water API client and TypeScript types

**Status:** pending  
**Dependencies:** 42.2  

Implement waterApi client following mealsApi pattern with typed responses, add WaterIntake and summary types to lib/types

**Details:**

1. Add TypeScript types to `lib/types/index.ts`:
   ```typescript
   export interface WaterIntake {
     id: string;
     userId: string;
     amount: number;  // ml
     recordedAt: string;
     createdAt: string;
   }
   export interface DailyWaterSummary {
     totalMl: number;
     goalMl: number;
     percentComplete: number;
     entries: WaterIntake[];
   }
   export interface WeeklyWaterSummary {
     date: string;
     totalMl: number;
     goalMl: number;
   }
   ```

2. Create `lib/api/water.ts` following mealsApi pattern:
   - Import api from './client'
   - Export waterApi object with methods:
     - `createWaterIntake(amount: number, recordedAt?: string): Promise<WaterIntake>`
     - `getWaterIntakes(date?: Date): Promise<WaterIntake[]>`
     - `deleteWaterIntake(id: string): Promise<void>`
     - `getDailySummary(date?: Date): Promise<DailyWaterSummary>`
     - `getWeeklySummary(): Promise<WeeklyWaterSummary[]>`
     - `updateWaterGoal(goalMl: number): Promise<User>`
   - Use api.post, api.get, api.delete, api.put
   - Convert Date to ISO string for params
   - Return typed responses: `api.get<WaterIntake[]>(...)`
   - Follow exact structure of mealsApi.ts

### 42.4. Build main water tracking screen with progress ring and quick-add UI

**Status:** pending  
**Dependencies:** 42.3  

Create app/water.tsx screen with circular progress visualization, quick-add buttons for common amounts, today's log list with swipeable cards, and weekly chart

**Details:**

1. Create `app/water.tsx` following app/activity/index.tsx screen pattern:
   - Import waterApi, useAuth, useRouter, colors, spacing, typography
   - State: dailySummary, isLoading, refreshing
   - useFocusEffect to load data on screen focus
   - SafeAreaView with ScrollView and RefreshControl

2. Circular Progress Ring component:
   - SVG-based ring (react-native-svg) showing totalMl/goalMl
   - Center text: "1500 / 2500 ml" and "60%"
   - Color gradient: colors.secondary.main → colors.status.success as progress increases
   - Animated using Animated API (spring animation on update)
   - Celebration animation when goal reached (confetti or scale pulse)

3. Quick Add Buttons Grid (4x2 layout):
   - Common amounts: 150ml, 250ml, 500ml, 750ml
   - Each button:
     - TouchableOpacity with icon (water drop) and amount label
     - onPress: Call waterApi.createWaterIntake(amount), reload summary
     - Ripple effect on press
     - backgroundColor: colors.background.elevated
     - activeOpacity: 0.7
   - Custom amount button: Opens modal (handled in subtask 5)

4. Today's Log Section:
   - ScrollView with swipeable cards (use SwipeableWaterCard pattern like SwipeableMealCard)
   - Each entry shows:
     - Time (formatTime from utils/formatters)
     - Amount ("250 ml")
     - Delete button on swipe
   - Empty state: "No water logged today. Tap a quick-add button to start!"
   - List reverse chronological (latest first)

5. Weekly Overview Chart:
   - Bar chart component (react-native-chart-kit or custom with react-native-svg)
   - Load getWeeklySummary data
   - Show last 7 days with goal line overlay
   - Tap day to navigate to that day's detail
   - X-axis: Day names (M, T, W, T, F, S, S)
   - Y-axis: ml (0 to max of goal+500)

6. Floating Action Button (FAB):
   - Position: absolute, bottom: 20, right: 20
   - Opens water-settings screen
   - Icon: settings gear
   - backgroundColor: colors.primary.main

7. Register screen in `app/_layout.tsx`:
   - Add 'water' route with headerShown: false
   - Add 'water-settings' route with headerShown: false

### 42.5. Create water goal settings screen and dashboard widget integration

**Status:** pending  
**Dependencies:** 42.4  

Build water-settings.tsx for goal management and reminders, create WaterWidget component for dashboard, integrate into app/(tabs)/index.tsx

**Details:**

1. Create `app/water-settings.tsx`:
   - Daily goal input:
     - Slider (500ml - 5000ml) with live preview
     - Preset buttons: 2000ml, 2500ml, 3000ml, 3500ml, 4000ml
     - Save button calls waterApi.updateWaterGoal
   - Goal calculator (optional):
     - Input: body weight (kg)
     - Formula: weight * 33ml
     - Activity level multiplier: sedentary (1.0), active (1.2), very active (1.5)
     - Climate adjustment: +500ml for hot/dry
   - Reminder settings:
     - Toggle: Enable reminders
     - Interval picker: 30min, 1hr, 1.5hr, 2hr
     - Quiet hours: Start time, End time (future: integrate with Task 40 push notifications)
   - Save button updates User model fields
   - Follow existing settings screen patterns

2. Create `lib/components/WaterWidget.tsx` for dashboard:
   - Compact widget (height: ~100px)
   - Circular mini progress ring (size: 60px)
   - Text: "1500 / 2500 ml"
   - Quick add button: "+250ml" (most common amount)
   - onPress (on widget): Navigate to /water screen using router.push('/water')
   - onPress (on quick add): Call waterApi.createWaterIntake(250), update widget
   - Style: colors.background.tertiary card with borderRadius.md
   - Follow existing widget patterns from dashboard

3. Integrate WaterWidget into `app/(tabs)/index.tsx`:
   - Add import: `import { WaterWidget } from '@/lib/components/WaterWidget'`
   - Insert widget after meals summary section, before health metrics
   - Load water summary in parallel with meals/supplements (add to Promise.all)
   - Pass summary data to WaterWidget as prop
   - Widget should blend seamlessly with existing dashboard cards

4. Add navigation link in profile settings:
   - In `app/(tabs)/profile.tsx`
   - Add row: "Water Goal & Reminders" → links to /water-settings
   - Icon: water drop
   - Follow existing settings row pattern

5. Accessibility:
   - All buttons have accessibilityLabel
   - Progress ring announces "Water tracking, 1500 milliliters of 2500 milliliter goal, 60 percent complete"
   - Quick add buttons: "Add 250 milliliters"
   - VoiceOver support for slider and inputs
