# Nutri E2E Tests

End-to-end tests for the Nutri mobile app using [Maestro](https://maestro.mobile.dev/).

## Overview

This test suite covers all major user flows in the Nutri app:
- **Authentication**: Sign in, sign up, forgot password, reset password
- **Home/Dashboard**: View summary, refresh, navigation
- **Meal Tracking**: Add meals, select meal types
- **Profile**: View profile, edit goals, logout
- **Health Settings**: Apple Health integration (iOS only)
- **Food Scanning**: Camera permissions, photo capture

## Prerequisites

1. **Install Maestro CLI**:
   ```bash
   curl -Ls "https://get.maestro.mobile.dev" | bash
   ```

2. **For iOS Testing**: Boot an iOS simulator:
   ```bash
   # List available simulators
   xcrun simctl list devices

   # Boot a simulator (e.g., iPhone 16 Pro)
   xcrun simctl boot "iPhone 16 Pro"

   # Open the Simulator app
   open -a Simulator
   ```

3. **Start the development server**:
   ```bash
   # From project root
   npx expo start --port 8081
   ```

4. **Start the backend API** (required for full flow tests):
   ```bash
   cd server && npm run dev
   ```

5. **Seed test user** (if not already done):
   ```bash
   # Create a test user in the database
   cd server && npm run db:seed  # Or manually create via API
   ```

## How E2E Tests Work with Expo Go

Since we're testing a development build via Expo Go, tests use a special `launch_app.yaml` flow:

1. **Launch Expo Go** - Opens the Expo Go app with clean state
2. **Connect to dev server** - Opens `exp://localhost:8081` deep link
3. **Wait for bundle** - Waits up to 90 seconds for JavaScript bundle to load
4. **Dismiss developer menu** - Taps "Continue" to dismiss the dev menu modal
5. **Dismiss inspector** - Handles the React Native element inspector overlay
6. **Ready for testing** - App is now on the welcome screen

All tests should use:
```yaml
- runFlow: ../../flows/launch_app.yaml
```

Instead of direct `launchApp` calls (which would only launch Expo Go, not your app).

## Directory Structure

```
e2e/
├── config.yaml           # Global configuration and environment variables
├── README.md             # This file
├── flows/                # Reusable test flows
│   ├── launch_app.yaml   # Required for all tests (handles Expo Go startup)
│   ├── sign_in.yaml
│   ├── sign_up.yaml
│   ├── logout.yaml
│   ├── navigate_to_add_meal.yaml
│   ├── add_meal.yaml
│   └── navigate_to_health_settings.yaml
├── tests/                # Test suites organized by feature
│   ├── auth/            # Authentication tests
│   │   ├── sign_in.yaml
│   │   ├── sign_in_validation.yaml
│   │   ├── sign_up.yaml
│   │   ├── sign_up_validation.yaml
│   │   ├── forgot_password.yaml
│   │   └── reset_password.yaml
│   ├── home/            # Home/Dashboard tests
│   │   ├── dashboard.yaml
│   │   └── navigation.yaml
│   ├── meals/           # Meal tracking tests
│   │   ├── add_meal.yaml
│   │   └── meal_types.yaml
│   ├── profile/         # Profile tests
│   │   ├── view_profile.yaml
│   │   ├── edit_goals.yaml
│   │   └── logout.yaml
│   ├── health/          # Health settings tests
│   │   └── health_settings.yaml
│   └── scan/            # Food scanning tests
│       └── scan_food.yaml
├── scripts/             # Helper scripts
│   ├── run-e2e-local.sh # All-in-one local test runner (starts services)
│   └── run-tests.sh     # Test runner with parallel support
└── reports/             # Test reports (gitignored)
```

## Running Tests

### Quick Start (Recommended)

The easiest way to run E2E tests locally - handles everything automatically:

```bash
cd e2e

# Run all tests (starts backend, Metro, iOS simulator if needed)
./scripts/run-e2e-local.sh

# Run specific test
./scripts/run-e2e-local.sh tests/auth/sign_in.yaml

# Run auth tests only
./scripts/run-e2e-local.sh tests/auth/

# Skip backend if already running
./scripts/run-e2e-local.sh -s tests/auth/sign_in.yaml

# Clean up and restart everything
./scripts/run-e2e-local.sh -c
```

The script will:
1. Check prerequisites (Maestro, Node.js, iOS simulator)
2. Start backend server and Metro bundler
3. Wait for services to be ready
4. Run the E2E tests
5. Clean up on exit

### Run All Tests Sequentially

```bash
cd e2e
./scripts/run-tests.sh
```

### Run Tests in Parallel

```bash
./scripts/run-tests.sh -p
```

### Run with Custom Shard Count

```bash
./scripts/run-tests.sh -p -s 8  # Use 8 parallel shards
```

### Run Specific Test Suite

```bash
./scripts/run-tests.sh -t auth              # Run all auth tests
./scripts/run-tests.sh -t home              # Run all home tests
./scripts/run-tests.sh -t tests/auth/sign_in.yaml  # Run specific test
```

### Using Maestro Directly

```bash
# Run a single test
maestro test tests/auth/sign_in.yaml

# Run all tests with environment variables
maestro test tests/ \
  -e TEST_USER_EMAIL="test@nutri-e2e.local" \
  -e TEST_USER_PASSWORD="TestPass123!" \
  -e TEST_USER_NAME="E2E Test User"

# Run in Maestro Studio (interactive mode)
maestro studio
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_USER_EMAIL` | Email for test user | `test@nutri-e2e.local` |
| `TEST_USER_PASSWORD` | Password for test user | `TestPass123!` |
| `TEST_USER_NAME` | Name for test user | `E2E Test User` |
| `API_BASE_URL` | Backend API URL | `http://localhost:3000/api` |

## Test Design Principles

### 1. Stable Selectors (testID)

All tests use `testID` props for element selection instead of text or accessibility labels:

```yaml
# Good - uses testID
- tapOn:
    id: "signin-submit-button"

# Avoid - fragile text selectors
- tapOn:
    text: "Sign In"
```

### 2. Reusable Flows

Common actions are extracted into reusable flows:

```yaml
# In any test file:
- runFlow: ../../flows/sign_in.yaml
```

### 3. Proper Wait Strategies

Tests use `extendedWaitUntil` for waiting on elements with timeouts (Maestro 2.x):

```yaml
# Use extendedWaitUntil for explicit timeouts
- extendedWaitUntil:
    visible:
      id: "home-screen"
    timeout: 10000  # 10 second timeout

# Use assertVisible for immediate assertions (no timeout parameter)
- assertVisible:
    id: "home-screen"
```

### 4. Test Isolation

Each test starts with a clean state:

```yaml
- launchApp:
    clearState: true
```

### 5. Optional Assertions

For platform-specific features (like Apple Health):

```yaml
- assertVisible:
    id: "health-settings-connect-button"
    optional: true  # Won't fail if not present
```

## Adding New Tests

### 1. Add testID to your component

```tsx
<TouchableOpacity
  onPress={handlePress}
  testID="my-feature-button"  // Add testID
>
  <Text>My Feature</Text>
</TouchableOpacity>
```

### 2. Create test file

```yaml
# e2e/tests/my-feature/my_test.yaml
appId: host.exp.Exponent

---

# Test: My Feature
- launchApp:
    clearState: true

# Sign in first
- runFlow: ../../flows/sign_in.yaml

# Your test steps
- tapOn:
    id: "my-feature-button"

# Assert result
- assertVisible:
    id: "my-feature-result"
```

## Updating Tests When UI Changes

If UI changes cause test failures:

1. **Update testID if renamed**: Search and replace in test files
2. **Update flow if navigation changes**: Modify the affected flow file
3. **Update assertions if screen changes**: Check element visibility assertions

The testID-based approach means most UI changes (text, styling, layout) won't break tests.

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

      - name: Install Maestro
        run: curl -Ls "https://get.maestro.mobile.dev" | bash

      - name: Install dependencies
        run: npm ci

      - name: Start backend
        run: cd server && npm ci && npm run dev &

      - name: Build app
        run: npm run build:ios

      - name: Run E2E tests
        run: cd e2e && ./scripts/run-tests.sh -p
```

## Troubleshooting

### Tests failing to find elements

1. Verify testID is correctly added to the component
2. Check if element is actually visible (not hidden, scrolled out of view)
3. Increase timeout if element takes time to appear

### Tests flaking intermittently

1. Add explicit waits before actions
2. Increase timeouts for network-dependent operations
3. Ensure clean state with `clearState: true`

### Camera tests not working

Camera tests require actual camera permission. On simulators:
- iOS: Grant permission in Settings
- Android: Use `adb` to grant permission

### Apple Health tests failing on Android

Health settings tests use `optional: true` for iOS-specific features. These assertions will be skipped on Android.

## Performance Tips

1. **Use parallel execution** for faster CI runs
2. **Group related tests** to reuse authentication state
3. **Keep flows minimal** - only include essential steps
4. **Use appropriate timeouts** - not too short, not too long
