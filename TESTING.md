# Testing Guide - Nutri Food Scanner

This document provides comprehensive guidance on testing the AR-powered food scanning feature.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
  - [Mobile App Tests](#mobile-app-tests)
  - [ML Service Tests](#ml-service-tests)
  - [E2E Tests](#e2e-tests)
- [Test Coverage](#test-coverage)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The Nutri food scanner feature has comprehensive test coverage across multiple levels:

```
                    E2E Tests (Maestro)
                 Critical User Journeys
               /                         \
         Integration Tests             Integration Tests
      (Mobile + ML Service)           (API Endpoints)
    /                                                 \
Unit Tests (Mobile)                                Unit Tests (ML Service)
- API Client                                       - Food Analysis Service
- Type Guards                                      - Portion Estimation
- Error Handling                                   - Nutrition Calculation
```

**Test Coverage Goals:**
- Unit Tests: 80%+ for critical paths
- Integration Tests: 70%+ for integration paths
- E2E Tests: 100% of critical user journeys

## Test Structure

### Mobile App Tests (`__tests__/`)

```
__tests__/
├── fixtures/
│   └── food-analysis-fixtures.ts    # Reusable test data and mocks
├── unit/
│   └── api/
│       └── food-analysis.test.ts    # API client unit tests
└── integration/
    └── (future integration tests)
```

### ML Service Tests (`ml-service/tests/`)

```
ml-service/tests/
├── fixtures/
│   └── conftest.py                  # Pytest fixtures
├── unit/
│   └── test_food_analysis_service.py # Service unit tests
└── integration/
    └── test_api_endpoints.py         # API endpoint integration tests
```

### E2E Tests (`.maestro/flows/`)

```
.maestro/flows/
├── food-scanner/
│   ├── scan-and-save-meal.yaml      # Happy path flow
│   ├── camera-permissions.yaml      # Permission handling
│   ├── scan-error-handling.yaml     # Error scenarios
│   └── scan-with-editing.yaml       # Manual editing flow
└── suites/
    └── food-scanner-suite.yaml      # Test suite configuration
```

## Running Tests

### Prerequisites

**Mobile App:**
- Node.js 20+
- npm dependencies installed (`npm ci`)

**ML Service:**
- Python 3.9+
- Dependencies installed: `cd ml-service && pip install -r requirements.txt`
- Test dependencies: `pip install pytest pytest-asyncio pytest-cov httpx`

**E2E Tests:**
- macOS (for iOS simulator)
- Xcode installed
- Maestro CLI installed: `curl -Ls "https://get.maestro.mobile.dev" | bash`
- iOS Simulator booted
- Backend and ML services running

### Mobile App Tests

#### Run All Tests

```bash
npm test
```

#### Run Tests in Watch Mode

```bash
npm test -- --watch
```

#### Run Tests with Coverage

```bash
npm test -- --coverage
```

#### Run Specific Test File

```bash
npm test -- __tests__/unit/api/food-analysis.test.ts
```

#### Run Tests Matching Pattern

```bash
npm test -- --testNamePattern="should handle network errors"
```

**Expected Output:**
```
PASS  __tests__/unit/api/food-analysis.test.ts
  FoodAnalysisAPI
    analyzeFood
      ✓ should successfully analyze food image (45ms)
      ✓ should include AR measurements when provided (32ms)
      ✓ should handle network errors with retry (1523ms)
      ✓ should not retry on client errors (4xx) (28ms)
      ✓ should handle timeout errors (31ms)
      ...

Test Suites: 1 passed, 1 total
Tests:       24 passed, 24 total
Snapshots:   0 total
Time:        4.832s
```

### ML Service Tests

Navigate to ML service directory:

```bash
cd ml-service
```

#### Run All Tests

```bash
pytest
```

#### Run Unit Tests Only

```bash
pytest tests/unit
```

#### Run Integration Tests Only

```bash
pytest tests/integration
```

#### Run with Coverage

```bash
pytest --cov=app --cov-report=html --cov-report=term
```

Open coverage report:

```bash
open htmlcov/index.html
```

#### Run Specific Test File

```bash
pytest tests/unit/test_food_analysis_service.py
```

#### Run Specific Test

```bash
pytest tests/unit/test_food_analysis_service.py::TestFoodAnalysisService::test_analyze_food_without_measurements
```

#### Run with Verbose Output

```bash
pytest -v
```

#### Run Tests Matching Pattern

```bash
pytest -k "nutrition"
```

**Expected Output:**
```
======================== test session starts =========================
platform darwin -- Python 3.9.18, pytest-7.4.3, pluggy-1.3.0
rootdir: /Users/you/nutri/ml-service
plugins: asyncio-0.21.1, cov-4.1.0
collected 65 items

tests/unit/test_food_analysis_service.py .................... [ 30%]
.................................................               [100%]

---------- coverage: platform darwin, python 3.9.18 -----------
Name                                       Stmts   Miss  Cover
--------------------------------------------------------------
app/services/food_analysis_service.py       156      8    95%
app/schemas/food_analysis.py                 32      0   100%
app/api/food_analysis.py                     89      5    94%
--------------------------------------------------------------
TOTAL                                       277     13    95%

======================== 65 passed in 3.45s =========================
```

### E2E Tests

E2E tests require the full stack to be running:

#### Step 1: Start Backend Services

In one terminal:

```bash
cd server
npm start
```

Wait for:
```
Server running on http://localhost:3000
```

#### Step 2: Start ML Service (Optional - for production tests)

In another terminal:

```bash
cd ml-service
uvicorn app.main:app --reload
```

Wait for:
```
INFO:     Application startup complete.
```

#### Step 3: Start iOS Simulator

```bash
# List available simulators
xcrun simctl list devices available

# Boot iPhone 15 (or your preferred device)
open -a Simulator
```

#### Step 4: Build and Install App

From project root:

```bash
# Prebuild
npx expo prebuild --platform ios

# Build for simulator
xcodebuild \
  -workspace ios/nutri.xcworkspace \
  -scheme nutri \
  -configuration Debug \
  -sdk iphonesimulator \
  -derivedDataPath ios/build \
  build

# Install on simulator
APP_PATH=$(find ios/build -name "*.app" | head -1)
SIMULATOR_ID=$(xcrun simctl list devices | grep "Booted" | grep -oE '[0-9A-F-]{36}')
xcrun simctl install "$SIMULATOR_ID" "$APP_PATH"
```

#### Step 5: Run Maestro Tests

Run all food scanner tests:

```bash
maestro test .maestro/flows/suites/food-scanner-suite.yaml
```

Run specific flow:

```bash
maestro test .maestro/flows/food-scanner/scan-and-save-meal.yaml
```

Run with recording:

```bash
maestro test --record .maestro/flows/food-scanner/scan-and-save-meal.yaml
```

**Expected Output:**
```
 ║  Running flow scan-and-save-meal.yaml
 ║
 ║  ✅ Launch app
 ║  ✅ Navigate to Add Meal screen
 ║  ✅ Tap on Scan Food with Camera
 ║  ✅ Grant camera permission
 ║  ✅ Take photo
 ║  ✅ Analyze food
 ║  ✅ Verify results shown
 ║  ✅ Use these results
 ║  ✅ Save meal
 ║  ✅ Verify meal saved
 ║
 ║  ✅ Flow completed successfully
```

## Test Coverage

### Current Coverage Metrics

**Mobile App:**
- API Client: ~90% (24 test cases)
- Error Handling: 100%
- Type Guards: 100%

**ML Service:**
- Food Analysis Service: ~85% (30+ test cases)
- API Endpoints: ~90% (45+ test cases)
- Schemas: 100%

**E2E:**
- Critical user paths: 100% (4 flows)
- Permission handling: 100%
- Error scenarios: 80%

### Viewing Coverage Reports

**Mobile App:**

```bash
npm test -- --coverage --coverageReporters=html
open coverage/index.html
```

**ML Service:**

```bash
cd ml-service
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### Coverage Goals

- **Must Have (MVP):** 70% overall coverage
- **Should Have:** 80% coverage for critical paths
- **Nice to Have:** 90%+ coverage across all modules

## Writing New Tests

### Mobile App Test Example

```typescript
// __tests__/unit/api/food-analysis.test.ts
import { FoodAnalysisAPI } from '@/lib/api/food-analysis';
import { fixtures } from '../../fixtures/food-analysis-fixtures';

describe('FoodAnalysisAPI', () => {
  let api: FoodAnalysisAPI;

  beforeEach(() => {
    api = new FoodAnalysisAPI({ baseUrl: 'http://test' });
  });

  it('should handle your test case', async () => {
    // Arrange
    const request = { imageUri: fixtures.images.valid };

    // Act
    const result = await api.analyzeFood(request);

    // Assert
    expect(result).toBeDefined();
    expect(result.foodItems).toHaveLength(1);
  });
});
```

### ML Service Test Example

```python
# ml-service/tests/unit/test_food_analysis_service.py
import pytest
from app.services.food_analysis_service import FoodAnalysisService

@pytest.mark.asyncio
async def test_your_test_case(service, sample_food_image):
    """Test description."""
    # Arrange
    # ... setup

    # Act
    result = await service.analyze_food(sample_food_image, None)

    # Assert
    assert result is not None
    assert len(result[0]) == 1  # food_items
```

### E2E Test Example

```yaml
# .maestro/flows/food-scanner/your-test.yaml
appId: com.nutri.app
---
- launchApp:
    clearState: true

- tapOn: "Your Button"
- assertVisible: "Expected Element"
```

## CI/CD Integration

Tests run automatically on:
- **Pull Requests:** All tests run
- **Push to master/develop:** All tests run
- **Nightly:** Extended E2E test suite

### CI Pipeline Jobs

1. **Lint** (~30s)
   - ESLint for mobile and backend
   - Python linting (planned)

2. **Backend Tests** (~2min)
   - Unit tests with PostgreSQL
   - Coverage reporting

3. **Mobile Tests** (~1min)
   - Unit tests with Jest
   - Coverage reporting

4. **ML Service Tests** (~2min)
   - Unit tests with pytest
   - Integration tests with FastAPI TestClient
   - Coverage reporting

5. **E2E Tests** (~10min)
   - iOS simulator tests with Maestro
   - Smoke tests + Food scanner suite
   - Screenshot/video capture on failure

### Viewing CI Results

1. Go to GitHub Actions tab
2. Click on your PR workflow run
3. View test results and coverage reports
4. Download artifacts (test results, screenshots, videos)

### Local CI Simulation

Run the same tests locally as CI:

```bash
# Lint
npm run lint
cd server && npm run lint

# Backend tests
cd server && npm test -- --coverage

# Mobile tests
npm test -- --coverage

# ML service tests
cd ml-service && pytest --cov=app

# E2E tests (requires services running)
maestro test .maestro/flows/suites/smoke-tests.yaml
maestro test .maestro/flows/suites/food-scanner-suite.yaml
```

## Troubleshooting

### Common Issues

#### Mobile Tests

**Issue:** `Cannot find module '@/lib/api/food-analysis'`

**Solution:** Check that paths are configured in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./*"]
    }
  }
}
```

**Issue:** `Network request failed` in tests

**Solution:** Mock axios properly:

```typescript
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;
```

#### ML Service Tests

**Issue:** `ModuleNotFoundError: No module named 'app'`

**Solution:** Run pytest from the `ml-service` directory:

```bash
cd ml-service
pytest
```

**Issue:** `AsyncioDeprecationWarning`

**Solution:** Add to `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

**Issue:** Database connection errors

**Solution:** Tests use SQLite by default. Check `conftest.py` for database setup.

#### E2E Tests

**Issue:** Maestro can't find simulator

**Solution:** Ensure simulator is booted:

```bash
xcrun simctl list devices | grep "Booted"
```

**Issue:** App not installed on simulator

**Solution:** Rebuild and reinstall:

```bash
# Clean
rm -rf ios/build

# Rebuild and install (see Step 4 above)
```

**Issue:** Backend not responding

**Solution:** Verify backend is running and accessible:

```bash
curl http://localhost:3000/health
```

**Issue:** Camera permission tests failing

**Solution:** Reset simulator permissions:

```bash
xcrun simctl privacy booted reset all
```

### Debug Mode

**Mobile Tests:**

```bash
npm test -- --verbose --no-cache
```

**ML Service Tests:**

```bash
pytest -vv --tb=short
```

**E2E Tests:**

```bash
maestro test --debug .maestro/flows/food-scanner/scan-and-save-meal.yaml
```

### Test Performance

If tests are slow:

1. **Use `test.only`** for focused testing during development:

```typescript
it.only('should test this specific case', () => {
  // ...
});
```

2. **Run specific test files** instead of entire suite

3. **Use `--maxWorkers`** to limit parallelism:

```bash
npm test -- --maxWorkers=2
```

4. **Skip expensive operations** in unit tests (use integration tests instead)

## Best Practices

### General

- ✅ Write tests before or alongside code (TDD)
- ✅ Use descriptive test names: "should do X when Y"
- ✅ Follow Arrange-Act-Assert pattern
- ✅ Test one thing per test
- ✅ Use fixtures and factories for test data
- ✅ Mock external dependencies
- ✅ Clean up after tests (database, files, etc.)

### Mobile Tests

- ✅ Mock API calls with axios mock
- ✅ Test error boundaries
- ✅ Use fixtures from `__tests__/fixtures/`
- ✅ Test loading and error states
- ✅ Verify type safety (no `any` types)

### ML Service Tests

- ✅ Use pytest fixtures from `conftest.py`
- ✅ Test async functions with `@pytest.mark.asyncio`
- ✅ Test edge cases (empty input, extreme values)
- ✅ Verify response schemas
- ✅ Test error handling and validation

### E2E Tests

- ✅ Test critical user paths only
- ✅ Use realistic test data
- ✅ Grant permissions early in flow
- ✅ Wait for animations to end
- ✅ Use meaningful element IDs
- ✅ Keep flows focused and atomic

## Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Native Testing Library](https://callstack.github.io/react-native-testing-library/)
- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Maestro Documentation](https://maestro.mobile.dev/getting-started/installing-maestro)
- [Test Strategy Document](./FOOD_SCANNER_TEST_STRATEGY.md)

## Support

For issues or questions:
1. Check this guide and troubleshooting section
2. Review test examples in the codebase
3. Check CI logs for details on failures
4. Ask the team in #engineering-tests Slack channel

---

**Last Updated:** 2024-01-15
**Maintained By:** Engineering Team
