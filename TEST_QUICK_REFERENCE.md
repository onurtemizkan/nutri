# Test Quick Reference

Quick commands for running tests during development.

## 🚀 Quick Start

```bash
# Run all mobile tests
npm test

# Run all ML service tests
cd ml-service && pytest

# Run E2E tests (requires services running)
maestro test .maestro/flows/suites/food-scanner-suite.yaml
```

## 📱 Mobile App Tests

```bash
# All tests
npm test

# Watch mode (auto-rerun on changes)
npm test -- --watch

# With coverage
npm test -- --coverage

# Specific file
npm test -- __tests__/unit/api/food-analysis.test.ts

# Tests matching pattern
npm test -- --testNamePattern="network errors"

# Debug mode
npm test -- --verbose --no-cache
```

## 🐍 ML Service Tests

```bash
cd ml-service

# All tests
pytest

# Unit tests only
pytest tests/unit

# Integration tests only
pytest tests/integration

# With coverage
pytest --cov=app --cov-report=html

# Specific test
pytest tests/unit/test_food_analysis_service.py::TestFoodAnalysisService::test_analyze_food

# Watch mode (requires pytest-watch)
ptw

# Verbose output
pytest -vv

# Tests matching pattern
pytest -k "nutrition"
```

## 🎭 E2E Tests (Maestro)

```bash
# Prerequisites: Backend running, iOS simulator booted, app installed

# Full food scanner suite
maestro test .maestro/flows/suites/food-scanner-suite.yaml

# Specific flow
maestro test .maestro/flows/food-scanner/scan-and-save-meal.yaml

# With recording
maestro test --record .maestro/flows/food-scanner/scan-and-save-meal.yaml

# Debug mode
maestro test --debug .maestro/flows/food-scanner/scan-and-save-meal.yaml
```

## 🔧 Setup Commands

### Start Services

```bash
# Backend (terminal 1)
cd server && npm start

# ML Service (terminal 2)
cd ml-service && uvicorn app.main:app --reload

# Mobile app (terminal 3)
npm start
```

### Setup Simulator

```bash
# Boot simulator
open -a Simulator

# Build and install app
npx expo prebuild --platform ios
xcodebuild -workspace ios/nutri.xcworkspace -scheme nutri -configuration Debug -sdk iphonesimulator build
xcrun simctl install booted $(find ios/build -name "*.app" | head -1)
```

## 📊 Coverage Reports

```bash
# Mobile (open in browser)
npm test -- --coverage --coverageReporters=html
open coverage/index.html

# ML Service (open in browser)
cd ml-service
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## 🐛 Debugging

```bash
# Reset simulator
xcrun simctl erase all

# Reset simulator permissions
xcrun simctl privacy booted reset all

# Check backend health
curl http://localhost:3000/health

# Check ML service health
curl http://localhost:8000/health

# View simulator logs
xcrun simctl spawn booted log stream --level debug
```

## 🎯 Common Workflows

### Before Committing

```bash
# 1. Run linting
npm run lint
cd server && npm run lint

# 2. Run unit tests
npm test
cd ml-service && pytest

# 3. Verify no regressions
cd server && npm test
```

### Testing a Feature

```bash
# 1. Write unit tests (TDD)
npm test -- --watch

# 2. Implement feature
# ... code ...

# 3. Run integration tests
cd ml-service && pytest tests/integration

# 4. Run E2E tests
maestro test .maestro/flows/food-scanner/your-flow.yaml
```

### Fixing Failing Tests

```bash
# 1. Run specific failing test
npm test -- --testNamePattern="your failing test"

# 2. Add debugging
console.log() in test
--verbose flag

# 3. Check coverage
npm test -- --coverage

# 4. Verify fix
npm test
```

## 📝 Test File Locations

```
Mobile Tests:
  __tests__/unit/api/food-analysis.test.ts
  __tests__/fixtures/food-analysis-fixtures.ts

ML Service Tests:
  ml-service/tests/unit/test_food_analysis_service.py
  ml-service/tests/integration/test_api_endpoints.py
  ml-service/tests/fixtures/conftest.py

E2E Tests:
  .maestro/flows/food-scanner/*.yaml
  .maestro/flows/suites/food-scanner-suite.yaml
```

## 🎨 Test Patterns

### Mobile (TypeScript/Jest)

```typescript
describe('Feature', () => {
  it('should do something', async () => {
    // Arrange
    const input = { ... };

    // Act
    const result = await yourFunction(input);

    // Assert
    expect(result).toBeDefined();
  });
});
```

### ML Service (Python/pytest)

```python
@pytest.mark.asyncio
async def test_feature(service, fixture):
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = await service.method(input_data)

    # Assert
    assert result is not None
```

### E2E (Maestro YAML)

```yaml
- tapOn: "Button Text"
- assertVisible: "Expected Element"
- inputText: "Hello World"
```

## 🔗 Links

- [Full Testing Guide](./TESTING.md)
- [Test Strategy](./FOOD_SCANNER_TEST_STRATEGY.md)
- [Architecture](./FOOD_ANALYSIS_ARCHITECTURE.md)

## 💡 Tips

- Use `it.only()` for focused testing
- Use fixtures for reusable test data
- Mock external dependencies
- Test error cases, not just happy path
- Keep tests fast and isolated
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

---

For detailed information, see [TESTING.md](./TESTING.md)
