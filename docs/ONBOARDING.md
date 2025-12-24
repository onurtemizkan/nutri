# Nutri Onboarding Flow

## Overview

The onboarding flow guides new users through setting up their profile, health goals, permissions, and lifestyle information. The flow consists of 6 steps, with some steps being optional (skippable).

## Architecture

### Backend (Server)

- **Database Models**: Prisma schema with `UserOnboarding`, `UserPermissions`, `UserHealthBackground`, and `UserLifestyle` tables
- **API Endpoints**: RESTful API at `/api/onboarding/*`
- **Validation**: Zod schemas for all step data

### Frontend (Mobile App)

- **Screens**: Expo Router file-based routing in `/app/onboarding/`
- **State Management**: React Context (`OnboardingContext`) with AsyncStorage for persistence
- **Configuration**: Centralized in `/lib/onboarding/config.ts`

## Flow Steps

### Step 1: Welcome
- **Screen**: `app/onboarding/index.tsx`
- **Purpose**: Introduction to the app
- **Data**: None (just navigation)
- **Skippable**: No

### Step 2: Profile Basics
- **Screen**: `app/onboarding/profile.tsx`
- **Purpose**: Collect basic user information
- **Data**:
  - Name (required)
  - Date of birth (required)
  - Biological sex (required): MALE, FEMALE, OTHER
  - Height in cm (required, 50-300)
  - Current weight in kg (required, 20-500)
  - Activity level (required): sedentary, light, moderate, active, veryActive
- **Skippable**: No (required for calorie calculations)

### Step 3: Health Goals
- **Screen**: `app/onboarding/goals.tsx`
- **Purpose**: Set nutrition and health goals
- **Data**:
  - Primary goal (required): WEIGHT_LOSS, WEIGHT_GAIN, MUSCLE_GAIN, MAINTENANCE, GENERAL_HEALTH, ATHLETIC_PERFORMANCE
  - Goal weight (optional)
  - Dietary preferences (optional): vegetarian, vegan, gluten_free, dairy_free, etc.
  - Custom macros (optional): goalCalories, goalProtein, goalCarbs, goalFat
- **Skippable**: No (required for personalization)

### Step 4: Permissions
- **Screen**: `app/onboarding/permissions.tsx`
- **Purpose**: Request app permissions
- **Data**:
  - Notifications enabled (optional)
  - Notification types: meal_reminders, insights, goals, weekly_summary
  - HealthKit enabled (iOS only, optional)
  - HealthKit scopes: heartRate, steps, sleep, activeEnergy, weight, etc.
- **Skippable**: No (user can skip individual toggles but must proceed)

### Step 5: Health Background
- **Screen**: `app/onboarding/health-background.tsx`
- **Purpose**: Collect health history for personalization
- **Data**:
  - Chronic conditions (optional)
  - Medications (optional)
  - Supplements (optional)
  - Allergies (optional)
- **Skippable**: Yes

### Step 6: Lifestyle
- **Screen**: `app/onboarding/lifestyle.tsx`
- **Purpose**: Collect lifestyle information
- **Data**:
  - Nicotine use (optional): NONE, OCCASIONAL, DAILY, HEAVY
  - Alcohol use (optional): NONE, OCCASIONAL, MODERATE, FREQUENT
  - Caffeine daily (optional, cups 0-20)
  - Sleep schedule (optional): bedtime, wake time
  - Sleep quality (optional, 1-10)
  - Stress level (optional, 1-10)
  - Work schedule (optional): regular, shift, irregular, remote, not_working
- **Skippable**: Yes

### Step 7: Completion
- **Screen**: `app/onboarding/complete.tsx`
- **Purpose**: Confirm completion and navigate to main app
- **Features**:
  - Shows setup summary
  - Lists skipped steps
  - Allows returning to skipped steps
  - Navigates to home screen

## API Endpoints

### Authentication
All endpoints require JWT authentication via `Authorization: Bearer <token>` header.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/onboarding/start` | Start or resume onboarding |
| GET | `/api/onboarding/status` | Get current onboarding status |
| GET | `/api/onboarding/data` | Get all onboarding data |
| PUT | `/api/onboarding/step/1` | Save Step 1 (Profile) |
| PUT | `/api/onboarding/step/2` | Save Step 2 (Goals) |
| PUT | `/api/onboarding/step/3` | Save Step 3 (Permissions) |
| PUT | `/api/onboarding/step/4` | Save Step 4 (Health Background) |
| PUT | `/api/onboarding/step/5` | Save Step 5 (Lifestyle) |
| POST | `/api/onboarding/skip/:step` | Skip a step (4, 5 only) |
| POST | `/api/onboarding/complete` | Complete onboarding |

### Response Format

**Status Response:**
```json
{
  "id": "cuid",
  "userId": "cuid",
  "currentStep": 1,
  "totalSteps": 6,
  "isComplete": false,
  "completedAt": null,
  "progress": 0,
  "skippedSteps": [],
  "stepsCompleted": []
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "details": {} // Zod validation errors if applicable
}
```

## State Management

### OnboardingContext

The `OnboardingContext` provides:
- `status`: Current onboarding status from backend
- `isLoading`: Loading state
- `refreshStatus()`: Refresh status from backend
- `saveStep(step, data)`: Save step data to backend
- `skipStep(step)`: Skip a skippable step
- `completeOnboarding()`: Mark onboarding as complete
- `getDraftForStep(step)`: Get locally saved draft
- `updateDraft(step, data)`: Update local draft

### Local Storage

Draft data is persisted to AsyncStorage under `nutri:onboarding:drafts` to:
- Allow users to leave and return to onboarding
- Preserve form data during app restarts
- Support offline-first data entry

## Testing

### Backend Tests

Run backend tests:
```bash
cd server
npm test -- src/__tests__/onboarding.test.ts
```

Tests cover:
- Starting onboarding
- Getting status
- Saving each step with validation
- Skipping optional steps
- Completing onboarding
- Error handling

### E2E Tests (Maestro)

Run E2E tests:
```bash
maestro test .maestro/flows/
```

Test flows:
- `onboarding_complete_flow.yaml`: Full happy path
- `onboarding_skip_optional_steps.yaml`: Skipping Health Background and Lifestyle
- `onboarding_navigation.yaml`: Back button and data persistence
- `onboarding_validation.yaml`: Form validation

## Configuration

### Step Configuration

Edit `/lib/onboarding/config.ts` to modify:
- Step definitions
- Skippable steps
- Required fields
- Option values (goals, dietary preferences, etc.)

### Validation Schema

Edit `/server/src/validation/schemas.ts` for:
- Field validation rules
- Value ranges
- Enum values

## Extending the Flow

### Adding a New Step

1. Create screen file in `/app/onboarding/`
2. Add step configuration in `/lib/onboarding/config.ts`
3. Add Zod schema in `/server/src/validation/schemas.ts`
4. Add service method in `/server/src/services/onboardingService.ts`
5. Add controller method in `/server/src/controllers/onboardingController.ts`
6. Add Prisma model if storing new data
7. Update `TOTAL_ONBOARDING_STEPS` constant
8. Add tests

### Adding a New Field to Existing Step

1. Update Zod schema
2. Update Prisma schema if needed
3. Update service method
4. Update mobile screen
5. Update config options if applicable
6. Add tests

## Troubleshooting

### Common Issues

**"Onboarding not started" error**
- Ensure you call `/api/onboarding/start` before accessing status or saving steps

**Step not advancing**
- Check validation errors in response
- Ensure all required fields are provided

**Data not persisting**
- Check network connectivity
- Verify AsyncStorage permissions
- Check backend logs for errors

**HealthKit/Notifications not working**
- Only available on physical devices
- Check iOS/Android permissions
- Verify correct scope configuration

## Security Considerations

- All endpoints require authentication
- Input sanitization applied to all fields
- Rate limiting on all API endpoints
- Sensitive health data encrypted at rest
- HIPAA-compliant data handling

## Performance

- Draft data stored locally to minimize API calls
- Optimistic UI updates
- Lazy loading of step screens
- Efficient Prisma queries with proper indexes
