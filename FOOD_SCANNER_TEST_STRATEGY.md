# Food Scanner Test Strategy

## Overview

Comprehensive testing strategy for AR-powered food scanning feature covering unit tests, integration tests, E2E tests, and performance benchmarks.

## Test Pyramid

```
                    E2E Tests
                 (Maestro - iOS)
               /                 \
         Integration Tests
      (Mobile + ML Service)
    /                           \
Unit Tests                    Unit Tests
(Mobile Components)          (ML Service)
```

## Testing Levels

### 1. Unit Tests

**Mobile App (Jest + React Native Testing Library)**
- Camera screen component logic
- API client request/response handling
- Type validation and guards
- Error handling utilities
- State management

**ML Service (pytest)**
- Food classification logic
- Portion estimation algorithms
- Nutrition calculation
- Image preprocessing
- Schema validation

**Coverage Target**: 80%+ for critical paths

### 2. Integration Tests

**Mobile App**
- Camera → API → Result display flow
- Permission handling end-to-end
- Error recovery scenarios
- Navigation flows

**ML Service**
- Full analysis pipeline (image → result)
- Database interactions
- Redis caching
- API endpoint chains

**Coverage Target**: 70%+ for integration paths

### 3. E2E Tests (Maestro)

**User Journeys**
- Complete food scanning flow
- Permission grants
- Error handling
- Meal creation with scan data

**Coverage Target**: Critical user paths

### 4. Performance Tests

**ML Service**
- Response time benchmarks
- Concurrent request handling
- Large image processing
- Memory usage

**Targets**:
- P50: <2s
- P95: <5s
- P99: <10s

### 5. Security Tests

**Input Validation**
- Malicious file uploads
- Oversized images
- Invalid formats
- SQL injection attempts

## Test Structure

### Mobile App Tests

```
__tests__/
├── unit/
│   ├── components/
│   │   ├── ScanFood.test.tsx
│   │   └── AddMeal.test.tsx
│   ├── api/
│   │   └── food-analysis.test.ts
│   ├── types/
│   │   └── food-analysis.test.ts
│   └── utils/
│       └── image-processing.test.ts
├── integration/
│   ├── food-scan-flow.test.tsx
│   └── camera-permissions.test.tsx
└── fixtures/
    ├── mock-images.ts
    ├── mock-responses.ts
    └── test-data.ts
```

### ML Service Tests

```
ml-service/tests/
├── unit/
│   ├── test_food_analysis_service.py
│   ├── test_portion_estimation.py
│   ├── test_nutrition_calculation.py
│   └── test_schemas.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_full_pipeline.py
│   └── test_error_handling.py
├── performance/
│   ├── test_response_times.py
│   └── test_concurrent_requests.py
├── fixtures/
│   ├── conftest.py
│   ├── sample_images.py
│   └── mock_data.py
└── e2e/
    └── test_complete_flow.py
```

### E2E Tests (Maestro)

```
.maestro/flows/
├── food-scanner/
│   ├── scan-and-save-meal.yaml
│   ├── camera-permissions.yaml
│   ├── scan-error-handling.yaml
│   └── scan-with-editing.yaml
└── suites/
    └── food-scanner-suite.yaml
```

## Test Data

### Sample Images
- Apple (clear, good lighting)
- Banana (partial view)
- Chicken breast (on plate)
- Mixed meal (multiple items)
- Blurry image (edge case)
- Dark image (poor lighting)

### Mock Responses
- High confidence (>90%)
- Medium confidence (70-80%)
- Low confidence (<60%)
- Multiple food items
- Unknown food
- Error scenarios

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Food Scanner Tests

on:
  pull_request:
    paths:
      - 'app/scan-food.tsx'
      - 'lib/api/food-analysis.ts'
      - 'ml-service/app/**'
  push:
    branches: [main, develop]

jobs:
  mobile-tests:
    - Unit tests
    - Integration tests
    - Coverage report

  ml-service-tests:
    - Unit tests
    - Integration tests
    - API tests
    - Coverage report

  e2e-tests:
    - Maestro tests
    - Screenshot capture
    - Video recording
```

## Test Scenarios

### Happy Path
1. User opens camera
2. Takes clear photo of single food item
3. ML service identifies correctly (>90% confidence)
4. Nutrition values calculated accurately
5. User confirms and saves meal

### Edge Cases
1. **No Camera Permission**: Graceful error handling
2. **Network Failure**: Retry logic works
3. **ML Service Down**: User-friendly error message
4. **Blurry Image**: Low confidence warning
5. **Unknown Food**: Suggestions provided
6. **Multiple Foods**: Primary item selected
7. **Large Image**: Compression works
8. **Slow Network**: Loading states shown

### Error Scenarios
1. **Invalid Image Format**: Rejected with clear message
2. **Oversized Image**: Rejected or compressed
3. **Corrupted Image**: Handled gracefully
4. **API Timeout**: Retry with exponential backoff
5. **Invalid Response**: Fallback to manual entry

## Test Metrics

### Success Criteria

**Unit Tests**:
- ✅ 80%+ code coverage
- ✅ All critical paths tested
- ✅ Edge cases handled
- ✅ Fast execution (<30s)

**Integration Tests**:
- ✅ 70%+ integration coverage
- ✅ All API endpoints tested
- ✅ Error recovery validated
- ✅ Execution time <2min

**E2E Tests**:
- ✅ Critical flows passing
- ✅ Permission handling works
- ✅ Error scenarios handled
- ✅ Execution time <5min

**Performance**:
- ✅ P95 response time <5s
- ✅ Handles 10 concurrent requests
- ✅ Memory usage stable
- ✅ No memory leaks

## Test Maintenance

### Regular Updates
- Weekly review of flaky tests
- Monthly test coverage audit
- Quarterly performance benchmarks
- Update mocks when API changes

### Documentation
- Keep test scenarios up-to-date
- Document complex test setups
- Maintain fixture data
- Update CI/CD configurations

## Tools & Libraries

### Mobile App
- **Jest**: Test runner
- **React Native Testing Library**: Component testing
- **MSW**: API mocking
- **@testing-library/jest-native**: Custom matchers

### ML Service
- **pytest**: Test framework
- **pytest-asyncio**: Async testing
- **pytest-cov**: Coverage reporting
- **httpx**: API testing
- **Pillow**: Image fixtures
- **locust**: Load testing

### E2E
- **Maestro**: Mobile E2E testing
- **Appium** (future): Advanced scenarios

## Next Steps

1. ✅ Implement mobile unit tests
2. ✅ Implement ML service unit tests
3. ✅ Create integration tests
4. ✅ Add E2E Maestro flows
5. ✅ Set up CI/CD pipeline
6. ✅ Establish coverage baselines
7. ✅ Document test procedures

## Success Metrics

### Definition of Done
- [ ] All tests passing
- [ ] Coverage targets met
- [ ] CI/CD pipeline green
- [ ] Documentation complete
- [ ] Performance benchmarks established
- [ ] Security tests passing

---

**Test Coverage Goal**: 80%+ critical paths, 70%+ integration, 100% critical user journeys
