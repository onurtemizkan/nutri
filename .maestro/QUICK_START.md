# Maestro E2E Testing - Quick Start Guide

## ⚡ Fast Track (5 Minutes)

### 1. Prerequisites Check ✅

```bash
# Verify Node.js (required: 18+)
node --version

# Verify Yarn
yarn --version

# Verify Maestro installation
maestro --version

# If Maestro not installed, run:
./.maestro/scripts/setup-maestro.sh
```

### 2. Build the App 🏗️

**For iOS:**
```bash
# Build for iOS Simulator
npx expo run:ios
```

**For Android:**
```bash
# Build for Android Emulator
npx expo run:android
```

> **Note:** The app must be built at least once before running tests. Maestro will launch the previously built app.

### 3. Start a Device 📱

**iOS Simulator:**
```bash
# List available simulators
xcrun simctl list devices

# Boot a simulator (iPhone 16 Pro)
xcrun simctl boot "iPhone 16 Pro"

# Or just open Simulator.app from Launchpad
```

**Android Emulator:**
```bash
# List emulators
emulator -list-avds

# Start an emulator
emulator -avd <emulator_name>
```

### 4. Run Your First Test 🎬

```bash
# Quick smoke tests (~3-5 minutes)
maestro test .maestro/flows/suites/smoke-tests.yaml \
  --env APP_ID=com.yourcompany.nutri

# Or use the helper script
./.maestro/scripts/run-tests.sh --suite smoke
```

**Success!** You should see tests executing in the simulator/emulator. 🎉

---

## 📚 Complete Testing Workflow

### Test Suites Overview

| Suite | Duration | When to Use | Command |
|-------|----------|-------------|---------|
| **Smoke Tests** | 3-5 min | Quick validation, every commit | `./run-tests.sh -s smoke` |
| **Critical Path** | 8-10 min | Before PR, key features | `./run-tests.sh -s critical` |
| **Regression** | 20-30 min | Before release, major changes | `./run-tests.sh -s regression` |
| **All Flows** | 25-35 min | Full validation, weekly | `./run-tests.sh -s all` |

### Running Individual Tests

```bash
# Auth flows
maestro test .maestro/flows/01-auth/signin.yaml --env APP_ID=com.yourcompany.nutri
maestro test .maestro/flows/01-auth/signup.yaml --env APP_ID=com.yourcompany.nutri
maestro test .maestro/flows/01-auth/logout.yaml --env APP_ID=com.yourcompany.nutri

# Meal tracking
maestro test .maestro/flows/02-meals/add-meal.yaml --env APP_ID=com.yourcompany.nutri

# Profile management
maestro test .maestro/flows/03-profile/edit-goals.yaml --env APP_ID=com.yourcompany.nutri

# Navigation
maestro test .maestro/flows/04-navigation/tab-navigation.yaml --env APP_ID=com.yourcompany.nutri
```

### Using Maestro Studio (Visual Testing) 🎨

Maestro Studio provides an interactive UI for running and debugging tests:

```bash
# Launch Maestro Studio
maestro studio

# Then:
# 1. Select your device from the dropdown
# 2. Browse and select a test flow file
# 3. Click "Run" to execute
# 4. Watch the test execute in real-time
# 5. Click on individual commands to see details
```

**Studio Benefits:**
- Visual test execution
- Step-by-step debugging
- Screenshot capture
- Edit flows on the fly
- Record new flows

---

## 🎯 Test Configuration

### Environment Variables

Edit `.maestro/config.yaml` to customize test data:

```yaml
env:
  # Test user credentials
  TEST_EMAIL: "testuser@example.com"
  TEST_PASSWORD: "Test123456"
  TEST_NAME: "Test User"

  # New user registration (uses random email)
  NEW_USER_EMAIL: "newuser${RANDOM}@example.com"
  NEW_USER_PASSWORD: "NewUser123"
  NEW_USER_NAME: "New Test User"

  # Test meal data
  TEST_MEAL_NAME: "Grilled Chicken Salad"
  TEST_MEAL_CALORIES: "450"
  TEST_MEAL_PROTEIN: "35"
  TEST_MEAL_CARBS: "40"
  TEST_MEAL_FAT: "15"
```

### App ID Configuration

Update `APP_ID` based on your app configuration:

**In `app.json`:**
```json
{
  "expo": {
    "ios": {
      "bundleIdentifier": "com.yourcompany.nutri"
    },
    "android": {
      "package": "com.yourcompany.nutri"
    }
  }
}
```

**In test commands:**
```bash
maestro test flow.yaml --env APP_ID=com.yourcompany.nutri
```

---

## 🔧 Advanced Usage

### Running Tests with Reports

```bash
# JUnit report (for CI/CD)
maestro test .maestro/flows/suites/smoke-tests.yaml \
  --env APP_ID=com.yourcompany.nutri \
  --format junit \
  --output test-results/results.xml

# HTML report (for humans)
maestro test .maestro/flows/suites/smoke-tests.yaml \
  --env APP_ID=com.yourcompany.nutri \
  --format html \
  --output test-results/report.html
```

### Running on Specific Platform

```bash
# Force iOS
maestro test flow.yaml --platform ios

# Force Android
maestro test flow.yaml --platform android
```

### Parallel Test Execution

```bash
# Run across 3 devices simultaneously
maestro test .maestro/flows/ \
  --env APP_ID=com.yourcompany.nutri \
  --shards 3
```

### Continuous Mode (Watch for Changes)

```bash
# Automatically re-run tests when flows change
maestro test flow.yaml --continuous
```

---

## 🐛 Debugging Failed Tests

### 1. Check Screenshots

Maestro automatically captures screenshots on failure:

```bash
# View screenshots
open ~/.maestro/tests/
```

### 2. Run with Maestro Studio

Studio shows exactly which step failed:

```bash
maestro studio
# Select the failing flow and run it
```

### 3. Add Debug Steps

Add to your test flow:

```yaml
# Take manual screenshot
- screenshot: debug-step-1.png

# Add delays to observe behavior
- waitUntil:
    visible: "Expected Text"
    timeout: 10000

# Print debug info (visible in terminal)
- evalScript: console.log("Current state:", JSON.stringify(state))
```

### 4. Check App Logs

**iOS:**
```bash
# View simulator logs
xcrun simctl spawn booted log stream --predicate 'processImagePath contains "nutri"'
```

**Android:**
```bash
# View emulator logs
adb logcat | grep nutri
```

---

## 🚨 Common Issues & Solutions

### Issue: "No devices found"

**Solution:**
```bash
# iOS: Start simulator
open -a Simulator

# Android: Start emulator
emulator -list-avds
emulator -avd <name>

# Verify device is connected
maestro test flow.yaml  # Will show available devices
```

### Issue: "App not installed"

**Solution:**
```bash
# Rebuild the app
npx expo run:ios    # or npx expo run:android

# Verify app is installed
# iOS: Check in simulator (Cmd+Shift+H to see home screen)
# Android: adb shell pm list packages | grep nutri
```

### Issue: "Element not found"

**Solution:**
1. Check if text is visible in app (case-sensitive!)
2. Add wait/delay before assertion:
   ```yaml
   - waitUntil:
       visible: "Expected Text"
       timeout: 5000
   ```
3. Use Maestro Studio to inspect actual UI elements

### Issue: "Tests timeout"

**Solution:**
1. Increase timeout in config.yaml:
   ```yaml
   defaults:
     assertionTimeout: 15000  # Increase from 10s to 15s
   ```
2. Add explicit waits before slow operations
3. Check if device is too slow (use faster simulator)

---

## 📊 Test Coverage

### Current Test Coverage

**Authentication (01-auth/):**
- ✅ User registration with validation
- ✅ Login with credentials
- ✅ Logout with confirmation
- ✅ Password reset flow
- ✅ Form validation (email, password, empty fields)

**Meal Tracking (02-meals/):**
- ✅ Add meal with all fields
- ✅ Select meal types (breakfast, lunch, dinner, snack)
- ✅ Add optional fields (fiber, serving size, notes)
- ✅ Form validation
- ✅ Daily summary display
- ✅ Pull to refresh

**Profile Management (03-profile/):**
- ✅ View user profile
- ✅ View daily goals
- ✅ Edit nutrition goals
- ✅ Cancel edits
- ✅ Goal validation

**Navigation (04-navigation/):**
- ✅ Tab switching (Home ↔ Profile)
- ✅ Modal navigation (Add Meal)
- ✅ Auth guard (protected routes)
- ✅ Session persistence

---

## 🎓 Writing New Tests

### 1. Create a New Flow File

```bash
# Create new test file
touch .maestro/flows/05-my-feature/my-test.yaml
```

### 2. Use This Template

```yaml
appId: ${APP_ID}

---
# Test: My Feature Test
# Description: What this test validates

- launchApp

# Login (if needed)
- tapOn: "Sign In"
- tapOn:
    text: "your@email.com"
- inputText: "${TEST_EMAIL}"
- tapOn:
    text: "Enter your password"
- inputText: "${TEST_PASSWORD}"
- tapOn: "Sign In"

# Your test steps
- tapOn: "My Button"
- assertVisible: "Expected Text"
- inputText: "Some Input"

# Verify results
- assertVisible: "Success Message"
```

### 3. Test Your Flow

```bash
# Run with Studio (recommended for development)
maestro studio

# Or run from command line
maestro test .maestro/flows/05-my-feature/my-test.yaml \
  --env APP_ID=com.yourcompany.nutri
```

### 4. Add to Test Suite

Edit `.maestro/flows/suites/regression.yaml`:

```yaml
# Add your new flow
- runFlow: ../05-my-feature/my-test.yaml
```

---

## 📖 Resources

**Official Maestro Documentation:**
- Maestro Homepage: https://maestro.mobile.dev
- Getting Started: https://maestro.mobile.dev/getting-started/introduction
- API Reference: https://maestro.mobile.dev/api-reference
- Examples: https://maestro.mobile.dev/examples

**Test Files in This Project:**
- Test flows: `.maestro/flows/`
- Configuration: `.maestro/config.yaml`
- Full README: `.maestro/README.md`
- Scripts: `.maestro/scripts/`

**Helper Scripts:**
- Setup: `.maestro/scripts/setup-maestro.sh`
- Run tests: `.maestro/scripts/run-tests.sh`

---

## 🎉 You're Ready!

You now have a complete E2E testing setup for your Nutri app. Happy testing! 🚀

**Quick Commands Reference:**
```bash
# Setup (one-time)
./.maestro/scripts/setup-maestro.sh

# Build app (when code changes)
npx expo run:ios  # or npx expo run:android

# Run smoke tests (regular)
./.maestro/scripts/run-tests.sh --suite smoke

# Run critical path (before PRs)
./.maestro/scripts/run-tests.sh --suite critical

# Interactive testing (development)
maestro studio

# Full regression (before releases)
./.maestro/scripts/run-tests.sh --suite regression
```
