# GitHub Actions Workflows

This directory contains CI/CD workflows for the Nutri application.

## Workflows

### 1. CI (`ci.yml`)

**Trigger:** Push to `main`/`develop` or Pull Requests

**Jobs:**
- **Lint** - Runs ESLint on the codebase
- **Backend Tests** - Runs Jest tests for the backend API with PostgreSQL
- **Mobile Tests** - Runs Jest tests for the mobile app
- **E2E Tests** - Runs Maestro smoke tests on iOS simulator
- **CI Success** - Summary job that requires all tests to pass

**Duration:** ~30-45 minutes

### 2. E2E Manual (`e2e-manual.yml`)

**Trigger:** Manual dispatch (workflow_dispatch)

**Purpose:** Run specific E2E test suites on-demand

**Available Test Suites:**
- `smoke` - Quick smoke tests (~3-5 minutes)
- `all` - All tests
- `auth` - Authentication tests
- `meals` - Meal management tests
- `health` - Health metrics tests
- `activity` - Activity tracking tests
- `navigation` - Navigation and routing tests
- `profile` - Profile and goals tests
- `validation` - Input validation and error handling tests

**How to Run:**
1. Go to Actions tab in GitHub
2. Select "E2E Tests (Manual)" workflow
3. Click "Run workflow"
4. Choose test suite from dropdown
5. Click "Run workflow"

**Duration:** ~30-60 minutes depending on test suite

### 3. E2E Nightly (`e2e-nightly.yml`)

**Trigger:**
- Scheduled: Every night at 2 AM UTC
- Manual dispatch

**Purpose:** Comprehensive E2E testing across all test suites

**Test Suites Run:**
- Authentication
- Meals
- Navigation
- Profile
- Validation

**Strategy:** Matrix execution (runs suites in parallel)

**Duration:** ~45-60 minutes (parallel execution)

## Environment Setup

All workflows automatically set up the required environment:

### Backend Requirements
- Node.js 20
- PostgreSQL 16
- Redis 7
- Prisma migrations
- Test user creation

### Mobile Requirements (E2E only)
- macOS runner
- Xcode (latest)
- iOS Simulator
- CocoaPods
- Maestro CLI

## Test Results

Test results are published as:
- **JUnit XML** - Uploaded as artifacts
- **Check Annotations** - Visible in PR checks
- **Code Coverage** - Backend tests upload to Codecov

## Artifacts

### Uploaded Artifacts
- **Backend Coverage** - `coverage/lcov.info`
- **Maestro Results** - `maestro-results/*.xml` (30 days retention)
- **iOS Debug App** - Only on failure (7 days retention)

## Environment Variables

### Backend Tests
```yaml
DATABASE_URL: postgresql://postgres:postgres@localhost:5432/nutri_test_db
JWT_SECRET: test-jwt-secret-key-for-ci
JWT_EXPIRES_IN: 7d
NODE_ENV: test
```

### E2E Tests
```yaml
DATABASE_URL: postgresql://postgres:postgres@localhost:5432/nutri_db
JWT_SECRET: test-jwt-secret-key-for-ci
JWT_EXPIRES_IN: 7d
REDIS_URL: redis://localhost:6379
NODE_ENV: development
PORT: 3000

# Maestro test credentials
APP_ID: com.anonymous.nutri
TEST_EMAIL: testuser@example.com
TEST_PASSWORD: Test123456
TEST_NAME: Test User
```

## Secrets Required

No secrets are required for basic CI. Optional:
- `CODECOV_TOKEN` - For code coverage upload (recommended)

## Troubleshooting

### E2E Tests Failing

Common issues:
1. **Simulator boot timeout** - Increase wait time in workflow
2. **App build failure** - Check Xcode version compatibility
3. **Backend not ready** - Increase server startup wait time
4. **Maestro installation** - Check Maestro version compatibility

### Backend Tests Failing

Common issues:
1. **Database connection** - Check PostgreSQL service health
2. **Migration errors** - Ensure migrations are up to date
3. **Test user creation** - Check `create-test-user.ts` script

### Mobile Tests Failing

Common issues:
1. **Jest config** - Check `jest.config.js`
2. **Module resolution** - Check transform patterns
3. **Dependencies** - Run `npm ci` to ensure clean install

## Performance Optimization

### Caching
All workflows use npm caching to speed up dependency installation.

### Parallel Execution
- Nightly E2E tests run suites in parallel using matrix strategy
- CI runs lint, backend, and mobile tests in parallel

### Timeout
- CI: Default (6 hours)
- E2E Manual: 60 minutes
- E2E Nightly: 90 minutes

## Local Development

To run the same tests locally:

```bash
# Lint
npm run lint

# Backend tests
cd server && npm test

# Mobile tests
npm test

# E2E tests (requires setup)
./scripts/run-maestro-headless.sh smoke
```

## Continuous Improvement

Consider adding:
- [ ] Code coverage thresholds
- [ ] Performance regression testing
- [ ] Visual regression testing
- [ ] Accessibility testing
- [ ] Security scanning
