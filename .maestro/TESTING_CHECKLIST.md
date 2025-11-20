# Maestro E2E Testing Checklist

## 📋 Pre-Test Setup Checklist

### ✅ Environment Setup
- [ ] Node.js 18+ installed (`node --version`)
- [ ] Yarn installed (`yarn --version`)
- [ ] Maestro CLI installed (`maestro --version`)
- [ ] Xcode installed (for iOS testing)
- [ ] Android Studio/SDK installed (for Android testing)
- [ ] Project dependencies installed (`yarn install`)

### ✅ Device/Simulator Ready
- [ ] iOS Simulator booted (or physical device connected)
- [ ] Android Emulator running (or physical device connected)
- [ ] App built and installed on device:
  - [ ] iOS: `npx expo run:ios` completed successfully
  - [ ] Android: `npx expo run:android` completed successfully

### ✅ Configuration
- [ ] App ID matches your `app.json` bundle identifier
- [ ] Test credentials configured in `.maestro/config.yaml`
- [ ] Environment variables set correctly

---

## 🧪 Test Execution Checklist

### Before Each Test Run
- [ ] Device/simulator is running and responsive
- [ ] App is installed (not just built)
- [ ] Previous app state cleared if needed (fresh install for registration tests)
- [ ] Network connection available (for API calls)

### Smoke Tests (Daily/Every Commit)
**Expected Duration:** 3-5 minutes

Run: `./maestro/scripts/run-tests.sh --suite smoke`

- [ ] Login flow works
- [ ] Basic navigation works (tabs)
- [ ] Can add a meal
- [ ] Can view profile
- [ ] Can logout

**Pass Criteria:** All tests pass, no crashes

### Critical Path Tests (Before PRs)
**Expected Duration:** 8-10 minutes

Run: `./maestro/scripts/run-tests.sh --suite critical`

- [ ] New user registration → add meals → edit goals → logout flow
- [ ] Existing user daily workflow (login → add meals → update goals → logout)
- [ ] Form validation journey (all critical validations work)
- [ ] Error recovery (wrong password → retry, cancel forms → retry)

**Pass Criteria:** All critical user journeys complete successfully

### Full Regression (Before Releases)
**Expected Duration:** 20-30 minutes

Run: `./maestro/scripts/run-tests.sh --suite regression`

- [ ] All auth flows pass
- [ ] All meal tracking flows pass
- [ ] All profile management flows pass
- [ ] All navigation flows pass
- [ ] End-to-end verification passes

**Pass Criteria:** 100% test pass rate, no flaky tests

---

## 📱 Platform-Specific Testing

### iOS Testing
- [ ] Test on iPhone simulator
- [ ] Test on iPad simulator (if supported)
- [ ] Test on latest iOS version
- [ ] Test on iOS 14+ (minimum supported version)
- [ ] Verify dark mode (if applicable)
- [ ] Verify different screen sizes

### Android Testing
- [ ] Test on Phone emulator
- [ ] Test on Tablet emulator (if supported)
- [ ] Test on Android 12+ (minimum supported version)
- [ ] Test on different API levels
- [ ] Verify dark mode (if applicable)
- [ ] Verify different screen sizes/densities

---

## 🐛 Debugging Checklist

### When Tests Fail

1. **Check Screenshots**
   - [ ] Review failure screenshots in `~/.maestro/tests/`
   - [ ] Identify which step failed
   - [ ] Check if UI looks correct at failure point

2. **Run in Maestro Studio**
   - [ ] Open Maestro Studio (`maestro studio`)
   - [ ] Load the failing test
   - [ ] Run step-by-step
   - [ ] Observe exact failure point

3. **Check App Logs**
   - [ ] View simulator/emulator logs
   - [ ] Look for errors/exceptions
   - [ ] Check network requests

4. **Verify Test Data**
   - [ ] Confirm test credentials are valid
   - [ ] Check if test user exists (for login tests)
   - [ ] Verify environment variables are set

5. **Check App State**
   - [ ] App is in expected state before test
   - [ ] No lingering modals/alerts from previous tests
   - [ ] User is logged out if test starts with login

### Common Fixes

- [ ] Increase timeout for slow operations
- [ ] Add `waitUntil` before assertions
- [ ] Clear app state between tests
- [ ] Restart device/simulator
- [ ] Rebuild app if code changed
- [ ] Update test to match UI changes

---

## 📊 Test Quality Checklist

### Test Maintainability
- [ ] Tests use environment variables (not hardcoded data)
- [ ] Tests are independent (don't rely on other tests)
- [ ] Tests clean up after themselves
- [ ] Test names are descriptive
- [ ] Comments explain complex steps

### Test Coverage
- [ ] Happy path tested
- [ ] Error cases tested
- [ ] Validation tested
- [ ] Edge cases tested
- [ ] Cancel/back navigation tested

### Test Reliability
- [ ] Tests pass consistently (not flaky)
- [ ] No hardcoded waits (use `waitUntil` instead)
- [ ] Timeouts are reasonable
- [ ] Tests handle async operations correctly
- [ ] Tests work on different devices/simulators

---

## 🚀 CI/CD Integration Checklist

### GitHub Actions Setup
- [ ] Workflow file created (`.github/workflows/maestro-tests.yml`)
- [ ] Secrets configured (if needed)
- [ ] App builds in CI
- [ ] Emulator starts correctly
- [ ] Tests run successfully
- [ ] Test results published
- [ ] Screenshots uploaded on failure

### Pre-Release Checklist
- [ ] All smoke tests pass on main branch
- [ ] Critical path tests pass
- [ ] Full regression suite passes
- [ ] Tests pass on both iOS and Android
- [ ] No known flaky tests
- [ ] Test coverage is adequate
- [ ] New features have tests

---

## 📝 Test Reporting Checklist

### After Test Run
- [ ] Review test results
- [ ] Check pass/fail rate
- [ ] Review failure screenshots
- [ ] Log any new bugs discovered
- [ ] Update test documentation if needed
- [ ] Share results with team

### Metrics to Track
- [ ] Total tests run
- [ ] Pass rate percentage
- [ ] Test execution time
- [ ] Flaky test count
- [ ] Coverage percentage
- [ ] Bugs found vs bugs missed

---

## 🔄 Regular Maintenance Checklist

### Weekly
- [ ] Run full regression suite
- [ ] Review and fix flaky tests
- [ ] Update test data if needed
- [ ] Check for new Maestro version
- [ ] Review test coverage

### Monthly
- [ ] Audit test quality
- [ ] Remove obsolete tests
- [ ] Add tests for new features
- [ ] Update documentation
- [ ] Review CI/CD performance

### Quarterly
- [ ] Major test suite refactoring (if needed)
- [ ] Performance optimization
- [ ] Test strategy review
- [ ] Team training/knowledge sharing

---

## ✨ Test Best Practices Checklist

### Writing Tests
- [ ] Start with simple smoke tests
- [ ] Add critical path tests next
- [ ] Build comprehensive regression suite last
- [ ] Use descriptive test names
- [ ] Add comments for complex logic
- [ ] Keep tests DRY (use config variables)

### Running Tests
- [ ] Run smoke tests frequently
- [ ] Run critical path before PRs
- [ ] Run regression before releases
- [ ] Use Maestro Studio for debugging
- [ ] Generate reports for documentation

### Maintaining Tests
- [ ] Fix failures immediately
- [ ] Update tests when UI changes
- [ ] Remove/update obsolete tests
- [ ] Keep test data fresh
- [ ] Monitor test execution time

---

## 🎯 Success Criteria

### Test Suite is Healthy When:
- ✅ Smoke tests pass in < 5 minutes
- ✅ Critical path tests pass in < 10 minutes
- ✅ Full regression passes in < 30 minutes
- ✅ Pass rate > 95%
- ✅ Zero flaky tests
- ✅ All new features have tests
- ✅ Tests run automatically in CI/CD
- ✅ Team actively uses and maintains tests

---

## 📞 Getting Help

### Resources
- Maestro Documentation: https://maestro.mobile.dev
- Project README: `.maestro/README.md`
- Quick Start Guide: `.maestro/QUICK_START.md`

### Troubleshooting
1. Check this checklist
2. Review Quick Start Guide
3. Run with Maestro Studio for visual debugging
4. Check GitHub Issues for known problems
5. Ask team for help

---

**Last Updated:** 2025-11-18

**Version:** 1.0.0
