# Nutri E2E Tests

End-to-end tests for the Nutri mobile app using [Maestro](https://maestro.mobile.dev/).

## Overview

This comprehensive test suite covers all major user flows in the Nutri app with predefined test data and multiple test user profiles:

### Test Coverage

| Area | Tests | Description |
|------|-------|-------------|
| **Authentication** | 20+ | Sign in, sign up, forgot password, reset password, validation |
| **Home/Dashboard** | 10+ | View summary, refresh, navigation, empty states |
| **Meal Tracking** | 15+ | Add meals (all types), edit, delete, validation |
| **Health Metrics** | 10+ | View metrics, add manual entries, sync |
| **Profile** | 12+ | View profile, edit goals, logout, settings |
| **Supplements** | 12+ | Add, edit, delete, log supplements (Pro feature) |
| **Barcode Scanner** | 5+ | Camera permissions, scanning, manual entry |
| **Navigation** | 10+ | Tab navigation, deep linking, state persistence |
| **User Journeys** | 5+ | Complete end-to-end user workflows |

## Test Users

The test suite includes 5 predefined test users with different profiles and data:

| User | Email | Password | Subscription | Use Case |
|------|-------|----------|--------------|----------|
| **Primary** | `test@nutri-e2e.local` | `TestPass123!` | Free | Standard testing |
| **Pro** | `pro@nutri-e2e.local` | `ProPass123!` | Pro | Pro features |
| **Trial** | `trial@nutri-e2e.local` | `TrialPass123!` | Pro Trial | Trial features |
| **Empty** | `empty@nutri-e2e.local` | `EmptyPass123!` | Free | Empty states |
| **Athlete** | `athlete@nutri-e2e.local` | `AthletePass123!` | Pro | High data volume |

Each user (except Empty) comes with 14 days of:
- Meals (breakfast, lunch, dinner, snacks)
- Health metrics (RHR, HRV, sleep, steps, etc.)
- Activities (running, cycling, weights, etc.)
- Weight records
- Water intake records
- Supplements (Pro users only)

## Prerequisites

1. **Install Maestro CLI**:
   ```bash
   curl -Ls "https://get.maestro.mobile.dev" | bash
   ```

2. **For iOS Testing**: Boot an iOS simulator:
   ```bash
   xcrun simctl boot "iPhone 15 Pro"
   open -a Simulator
   ```

3. **Start the backend API**:
   ```bash
   cd server && npm run dev
   ```

4. **Seed test data**:
   ```bash
   cd server && npm run db:seed
   ```

5. **Start the development server**:
   ```bash
   npx expo start --port 8081
   ```

## Quick Start

### Using the Test Runner Script (Recommended)

```bash
# Run all tests
./e2e/scripts/run-e2e-tests.sh

# Run specific test suite
./e2e/scripts/run-e2e-tests.sh auth
./e2e/scripts/run-e2e-tests.sh meals
./e2e/scripts/run-e2e-tests.sh health
./e2e/scripts/run-e2e-tests.sh profile
./e2e/scripts/run-e2e-tests.sh supplements

# Run with database seeding
./e2e/scripts/run-e2e-tests.sh --seed

# Run tests in parallel
./e2e/scripts/run-e2e-tests.sh --parallel

# Combine options
./e2e/scripts/run-e2e-tests.sh --seed --parallel auth
```

### Using Maestro Directly

```bash
cd e2e

# Run single test
maestro test tests/auth/sign_in_comprehensive.yaml

# Run test suite
maestro test tests/auth/

# Run all tests
maestro test tests/

# Run in parallel
maestro test --parallel tests/

# Interactive mode
maestro studio
```

## Directory Structure

```
e2e/
├── config.yaml                    # Global config with all test users and data
├── README.md                      # This file
├── flows/                         # Reusable test flows
│   ├── launch_app.yaml            # Handles Expo Go startup
│   ├── sign_in.yaml               # Sign in with primary test user
│   ├── sign_in_pro_user.yaml      # Sign in with Pro user
│   ├── sign_in_empty_user.yaml    # Sign in with empty user
│   ├── sign_in_athlete_user.yaml  # Sign in with athlete user
│   ├── add_breakfast_meal.yaml    # Add predefined breakfast
│   ├── add_lunch_meal.yaml        # Add predefined lunch
│   ├── add_dinner_meal.yaml       # Add predefined dinner
│   ├── add_snack_meal.yaml        # Add predefined snack
│   ├── navigate_to_health_tab.yaml
│   ├── navigate_to_profile_tab.yaml
│   └── logout.yaml
├── tests/
│   ├── auth/
│   │   ├── sign_in.yaml
│   │   ├── sign_in_comprehensive.yaml    # All sign-in scenarios
│   │   ├── sign_up_comprehensive.yaml    # All sign-up scenarios
│   │   ├── password_reset_comprehensive.yaml
│   │   └── ...
│   ├── home/
│   │   ├── dashboard.yaml
│   │   ├── dashboard_comprehensive.yaml  # All dashboard scenarios
│   │   └── navigation.yaml
│   ├── meals/
│   │   ├── add_meal.yaml
│   │   ├── add_meal_comprehensive.yaml   # All meal types
│   │   ├── edit_meal_comprehensive.yaml
│   │   └── ...
│   ├── health/
│   │   ├── health_comprehensive.yaml     # All health scenarios
│   │   ├── add_health_metric.yaml
│   │   └── health_settings.yaml
│   ├── profile/
│   │   ├── profile_comprehensive.yaml    # All profile scenarios
│   │   ├── edit_goals.yaml
│   │   └── logout.yaml
│   ├── scan/
│   │   ├── scan_food.yaml
│   │   └── barcode_comprehensive.yaml
│   ├── supplements/
│   │   └── supplements_comprehensive.yaml
│   └── common/
│       ├── navigation_comprehensive.yaml
│       ├── user_journey_complete.yaml    # Full user workflows
│       └── session_persistence.yaml
├── scripts/
│   └── run-e2e-tests.sh           # Automated test runner
└── reports/                       # Test reports (gitignored)
```

## Environment Variables

All test data is predefined in `config.yaml`. Key variables:

### Test Users
| Variable | Value |
|----------|-------|
| `TEST_USER_EMAIL` | `test@nutri-e2e.local` |
| `TEST_USER_PASSWORD` | `TestPass123!` |
| `PRO_USER_EMAIL` | `pro@nutri-e2e.local` |
| `PRO_USER_PASSWORD` | `ProPass123!` |
| `EMPTY_USER_EMAIL` | `empty@nutri-e2e.local` |
| `EMPTY_USER_PASSWORD` | `EmptyPass123!` |
| `ATHLETE_USER_EMAIL` | `athlete@nutri-e2e.local` |
| `ATHLETE_USER_PASSWORD` | `AthletePass123!` |

### Predefined Meal Data
| Variable | Value |
|----------|-------|
| `MEAL_BREAKFAST_NAME` | Oatmeal with Berries |
| `MEAL_BREAKFAST_CALORIES` | 350 |
| `MEAL_LUNCH_NAME` | Grilled Chicken Salad |
| `MEAL_LUNCH_CALORIES` | 450 |
| `MEAL_DINNER_NAME` | Grilled Salmon with Rice |
| `MEAL_DINNER_CALORIES` | 620 |
| `MEAL_SNACK_NAME` | Protein Bar |
| `MEAL_SNACK_CALORIES` | 220 |

### Invalid Data (for validation tests)
| Variable | Value |
|----------|-------|
| `INVALID_EMAIL` | notanemail |
| `INVALID_PASSWORD` | short |
| `WRONG_PASSWORD` | WrongPassword123! |

## Test Design Principles

### 1. Predefined Test Data

All test inputs are predefined in `config.yaml`:

```yaml
# Uses config variable
- tapOn:
    id: "add-meal-name-input"
- inputText: ${MEAL_BREAKFAST_NAME}
```

### 2. Multiple User Profiles

Tests can run with different user profiles for comprehensive coverage:

```yaml
# Test with primary user (Free tier)
- runFlow: ../../flows/sign_in.yaml

# Test with Pro user (has supplements, more features)
- runFlow: ../../flows/sign_in_pro_user.yaml

# Test empty states
- runFlow: ../../flows/sign_in_empty_user.yaml

# Test with high data volume
- runFlow: ../../flows/sign_in_athlete_user.yaml
```

### 3. Stable Selectors (testID)

All tests use `testID` props for element selection:

```yaml
# Good - uses testID
- tapOn:
    id: "signin-submit-button"

# Avoid - fragile text selectors
- tapOn:
    text: "Sign In"
```

### 4. Reusable Flows

Common actions are extracted into reusable flows:

```yaml
# Add a meal using predefined data
- runFlow: ../../flows/add_breakfast_meal.yaml
```

### 5. Optional Assertions

For platform-specific or feature-gated functionality:

```yaml
# Pro feature - won't fail for Free users
- assertVisible:
    id: "supplements-screen"
    optional: true
```

## Adding New Tests

### 1. Add testID to component

```tsx
<TouchableOpacity
  onPress={handlePress}
  testID="my-feature-button"
>
  <Text>My Feature</Text>
</TouchableOpacity>
```

### 2. Add test data to config.yaml (if needed)

```yaml
env:
  MY_FEATURE_INPUT_VALUE: "Test Value"
```

### 3. Create test file

```yaml
# e2e/tests/my-feature/my_test.yaml
appId: host.exp.Exponent

---

# Test: My Feature with Predefined Data
- launchApp:
    clearState: true

- runFlow: ../../flows/sign_in.yaml

- tapOn:
    id: "my-feature-button"

- tapOn:
    id: "my-feature-input"
- inputText: ${MY_FEATURE_INPUT_VALUE}

- assertVisible:
    id: "my-feature-result"
```

## Database Seeding

The seed script (`server/prisma/seed.ts`) creates:

- **5 test users** with different subscription tiers
- **14 days of meals** per user (breakfast, lunch, dinner, snacks)
- **30 days of health metrics** (RHR, HRV, sleep, steps, calories)
- **30 days of activities** (running, cycling, weights, yoga, etc.)
- **30 days of weight records**
- **14 days of water intake**
- **3-5 supplements** with logs (Pro users only)
- **ML profiles** for each user

```bash
# Seed database
cd server && npm run db:seed

# Output shows all created test users with credentials
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Maestro
        run: curl -Ls "https://get.maestro.mobile.dev" | bash

      - name: Install dependencies
        run: |
          npm ci
          cd server && npm ci

      - name: Setup database
        run: |
          cd server
          npm run db:generate
          npm run db:push
          npm run db:seed

      - name: Start backend
        run: cd server && npm run dev &

      - name: Start Metro
        run: npx expo start --port 8081 &

      - name: Boot iOS Simulator
        run: |
          xcrun simctl boot "iPhone 15 Pro"
          open -a Simulator

      - name: Wait for services
        run: sleep 30

      - name: Run E2E tests
        run: ./e2e/scripts/run-e2e-tests.sh --parallel

      - name: Upload test reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: e2e-reports
          path: e2e/reports/
```

## Troubleshooting

### Tests failing to find elements

1. Verify testID is correctly added to the component
2. Check if element is visible (not hidden or scrolled out)
3. Use `extendedWaitUntil` with timeout for async elements

### Database not seeded

```bash
cd server
npm run db:generate   # Generate Prisma client
npm run db:push       # Push schema to DB
npm run db:seed       # Seed test data
```

### Tests flaking intermittently

1. Add explicit waits before actions
2. Increase timeouts for network-dependent operations
3. Ensure clean state with `clearState: true`

### Wrong test user data

If tests expect data that doesn't exist:
1. Re-run database seed: `cd server && npm run db:seed`
2. Verify the correct user flow is being used

### Camera/Barcode tests not working

Camera tests require permission. These tests use `optional: true` assertions and will pass even if camera access is denied.

### Pro features failing for Free user

Check you're using the correct sign-in flow:
- `sign_in.yaml` - Free user
- `sign_in_pro_user.yaml` - Pro user

## Performance Tips

1. **Use parallel execution** for faster CI runs: `maestro test --parallel`
2. **Group related tests** to reuse authentication state
3. **Use appropriate timeouts** - 10s default, 30s for network
4. **Seed once, run many** - Don't re-seed between tests
