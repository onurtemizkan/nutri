# Smoke Tests

Individual smoke test files for fast, isolated testing.

## Test Files

1. **01-login-navigation.yaml** (~30s)
   - Tests authentication flow
   - Validates tab navigation
   - Verifies logout functionality

2. **02-add-meal.yaml** (~45s)
   - Tests meal creation flow
   - Validates form input handling
   - Verifies success notification

3. **03-profile-logout.yaml** (~30s)
   - Tests profile view
   - Validates user data display
   - Verifies logout flow

4. **04-registration.yaml** (~30s)
   - Tests new user signup
   - Validates form validation
   - Verifies auto-login after registration

## Running Tests

### Run All Smoke Tests (Sequential)

```bash
./scripts/run-maestro-headless.sh smoke
```

### Run Individual Test

```bash
maestro test \
  --env APP_ID=com.anonymous.nutri \
  --env TEST_EMAIL=testuser@example.com \
  --env TEST_PASSWORD=Test123456 \
  --env TEST_NAME="Test User" \
  .maestro/flows/smoke/01-login-navigation.yaml
```

### Run from .maestro directory

```bash
cd .maestro
maestro test \
  --env APP_ID=com.anonymous.nutri \
  --env TEST_EMAIL=testuser@example.com \
  --env TEST_PASSWORD=Test123456 \
  flows/smoke/01-login-navigation.yaml
```

## CI Integration

These tests run automatically on:
- **Pull Requests**: Smoke tests only
- **Push to master/develop**: Full smoke test suite
- **Manual trigger**: Via GitHub Actions workflow_dispatch

## Prerequisites

- **Maestro CLI** installed
- **Metro bundler** running (npm start)
- **Backend server** running (cd server && npm run dev)
- **iOS Simulator** booted with app installed
- **Test user** exists in database (testuser@example.com)

## Troubleshooting

### docker-compose.yml Conflict

If tests fail locally with "Config Field Required" error mentioning docker-compose.yml:

The headless script automatically handles this by temporarily moving docker-compose.yml during test execution.

### XCUITest Driver Issues

If tests fail after running multiple tests in sequence, try:

```bash
# Restart simulator
xcrun simctl shutdown all
xcrun simctl boot <SIMULATOR_ID>

# Or reboot the simulator from the UI
```

### App Not Responding

Ensure Metro bundler is running:
```bash
npx expo start
```

The app needs the dev server in development mode.

## Test Structure

Each test follows this pattern:

1. **Launch app** - Start fresh
2. **Login** (if needed) - Authenticate user
3. **Test actions** - Perform specific flow
4. **Assertions** - Verify expected state
5. **Cleanup** (if needed) - Logout or reset

Tests are independent and can run in any order.
