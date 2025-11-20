# Maestro E2E Testing

This directory contains end-to-end tests for the Nutri mobile app using [Maestro](https://maestro.mobile.dev/).

## Test Structure

```
.maestro/
├── config.yaml              # Global test configuration
├── ci-config.yaml          # CI-specific configuration
├── flows/                  # Test flows organized by feature
│   ├── suites/
│   │   └── smoke-tests.yaml    # Quick sanity checks
│   ├── 01-auth/
│   │   └── password-reset.yaml
│   ├── 02-meals/
│   │   ├── edit-meal.yaml
│   │   └── delete-meal.yaml
│   ├── 03-health/
│   │   └── add-health-metric.yaml
│   ├── 04-activity/
│   │   └── log-activity.yaml
│   ├── 05-profile/
│   │   └── update-profile.yaml
│   └── 06-validation/
│       └── error-handling.yaml
└── README.md               # This file
```

## Prerequisites

1. **Install Maestro CLI**:
   ```bash
   curl -Ls "https://get.maestro.mobile.dev" | bash
   ```

2. **Start Backend Server**:
   ```bash
   cd server && npm run dev
   ```

3. **Create Test User** (one-time setup):
   ```bash
   curl -X POST http://localhost:3000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "email": "testuser@example.com",
       "password": "Test123456",
       "name": "Test User"
     }'
   ```

4. **Build and Run App**:
   ```bash
   # iOS
   npx expo run:ios

   # Android
   npx expo run:android
   ```

## Running Tests Locally

### Run Smoke Tests (Quick)
```bash
maestro test \
  --env APP_ID=com.anonymous.nutri \
  --env TEST_EMAIL=testuser@example.com \
  --env TEST_PASSWORD=Test123456 \
  --env TEST_NAME="Test User" \
  .maestro/flows/suites/smoke-tests.yaml
```

### Run All Tests
```bash
maestro test \
  --env APP_ID=com.anonymous.nutri \
  .maestro/flows/
```

### Run Specific Test Suite
```bash
# Authentication tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/01-auth/

# Meal tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/02-meals/

# Health tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/03-health/

# Activity tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/04-activity/

# Profile tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/05-profile/

# Validation tests
maestro test --env APP_ID=com.anonymous.nutri .maestro/flows/06-validation/
```

## Headless Testing (CI/CD)

### Using the Headless Script

We provide a convenient script for running tests headlessly:

```bash
# Make script executable (one-time)
chmod +x scripts/run-maestro-headless.sh

# Run smoke tests
./scripts/run-maestro-headless.sh smoke

# Run all tests
./scripts/run-maestro-headless.sh all

# Run specific test suite
./scripts/run-maestro-headless.sh auth
./scripts/run-maestro-headless.sh meals
./scripts/run-maestro-headless.sh health
./scripts/run-maestro-headless.sh activity
./scripts/run-maestro-headless.sh profile
./scripts/run-maestro-headless.sh validation
```

### Configuration Options

Set environment variables to customize test execution:

```bash
# Change app ID
export APP_ID=com.yourapp.id

# Change platform (ios or android)
export PLATFORM=android

# Change output format (junit, json, or pretty)
export FORMAT=json

# Change output directory
export OUTPUT_DIR=test-results

# Run tests
./scripts/run-maestro-headless.sh smoke
```

### Output Formats

- **junit**: XML format for CI integration (default for headless)
- **json**: JSON format for custom parsing
- **pretty**: Human-readable console output (default for local)

## GitHub Actions

Tests run automatically on:
- **Push to main/develop**: Runs smoke tests only
- **Pull requests**: Runs smoke tests
- **Manual trigger**: Can run full test suite
- **Scheduled**: Daily full test suite (optional)

### View Test Results

1. Go to the **Actions** tab in GitHub
2. Select the **Maestro E2E Tests** workflow
3. View test results and download artifacts:
   - Test results (JUnit XML)
   - Screenshots (on failure)

### Required Secrets

Configure these in GitHub repository settings:

```
DATABASE_URL=your_database_url
JWT_SECRET=your_jwt_secret
```

## Test Development

### Adding New Tests

1. **Create a new test file** in the appropriate directory:
   ```yaml
   appId: ${APP_ID}

   # Test: Your Test Name
   # Validates: What this test checks

   - launchApp
   - tapOn: "Button Text"
   - assertVisible: "Expected Text"
   ```

2. **Follow naming conventions**:
   - Use kebab-case: `add-meal.yaml`
   - Group related tests in folders

3. **Use environment variables** for test data:
   ```yaml
   - inputText: "${TEST_EMAIL}"
   ```

### Best Practices

1. **Keep tests independent**: Each test should work in isolation
2. **Use descriptive names**: Clear test and step descriptions
3. **Add comments**: Explain complex interactions
4. **Verify state**: Assert expected UI state after actions
5. **Clean up**: Reset app state if needed

### Debugging Failed Tests

1. **Check screenshots** in `~/.maestro/tests/`
2. **Review logs** in test output
3. **Run interactively** without headless mode
4. **Use Maestro Studio** for visual debugging:
   ```bash
   maestro studio
   ```

## Troubleshooting

### Tests Fail with "Device Not Found"

Make sure simulator/emulator is running:
```bash
# iOS
xcrun simctl boot <SIMULATOR_ID>

# Android
emulator -avd <AVD_NAME>
```

### Environment Variables Not Loading

Pass them explicitly:
```bash
maestro test \
  --env APP_ID=com.anonymous.nutri \
  --env TEST_EMAIL=testuser@example.com \
  .maestro/flows/suites/smoke-tests.yaml
```

### Test User Doesn't Exist

Create test user via API (see Prerequisites)

### App Not Installing

Rebuild the app:
```bash
# iOS
npx expo run:ios --device "iPhone 16 Pro"

# Android
npx expo run:android
```

## Resources

- [Maestro Documentation](https://maestro.mobile.dev/)
- [Maestro Cloud](https://cloud.mobile.dev/)
- [GitHub Actions Integration](https://maestro.mobile.dev/ci-integration)
