# GitHub Actions Workflows

This directory contains CI/CD workflows for the Nutri application.

## Workflows

### CI (`ci.yml`)

**Trigger:** Push to `main`/`develop` or Pull Requests

**Jobs:**
- **Lint** - Runs ESLint on the codebase
- **Backend Tests** - Runs Jest tests for the backend API with PostgreSQL
- **Mobile Tests** - Runs Jest tests for the mobile app
- **CI Success** - Summary job that requires all tests to pass

**Duration:** ~10-15 minutes

## Environment Setup

All workflows automatically set up the required environment:

### Backend Requirements
- Node.js 20
- PostgreSQL 16
- Prisma migrations
- Test user creation

## Test Results

Test results are published as:
- **JUnit XML** - Uploaded as artifacts
- **Check Annotations** - Visible in PR checks
- **Code Coverage** - Backend tests upload to Codecov

## Artifacts

### Uploaded Artifacts
- **Backend Coverage** - `coverage/lcov.info`

## Environment Variables

### Backend Tests
```yaml
DATABASE_URL: postgresql://postgres:postgres@localhost:5432/nutri_test_db
JWT_SECRET: test-jwt-secret-key-for-ci
JWT_EXPIRES_IN: 7d
NODE_ENV: test
```

## Secrets Required

No secrets are required for basic CI. Optional:
- `CODECOV_TOKEN` - For code coverage upload (recommended)

## Troubleshooting

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
- CI runs lint, backend, and mobile tests in parallel

## Local Development

To run the same tests locally:

```bash
# Lint
npm run lint

# Backend tests
cd server && npm test

# Mobile tests
npm test
```

## Continuous Improvement

Consider adding:
- [ ] Code coverage thresholds
- [ ] Performance regression testing
- [ ] Visual regression testing
- [ ] Accessibility testing
- [ ] Security scanning
