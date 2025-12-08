# Maestro E2E Tests

This directory contains end-to-end tests for the Nutri mobile app using [Maestro](https://maestro.mobile.dev/).

## Prerequisites

1. Install Maestro CLI:
   ```bash
   # macOS
   brew install maestro

   # Linux
   curl -Ls "https://get.maestro.mobile.dev" | bash
   ```

2. Build and run the app:
   ```bash
   # iOS Simulator
   npm run ios

   # Android Emulator
   npm run android
   ```

3. Ensure the backend server is running:
   ```bash
   cd server && npm run dev
   ```

## Running Tests

### Run All Tests
```bash
maestro test .maestro/flows/
```

### Run Specific Test
```bash
maestro test .maestro/flows/onboarding_complete_flow.yaml
```

### Run Tests by Tag
```bash
maestro test .maestro/flows/ --include-tags=onboarding
```

### Run in Studio (Interactive Mode)
```bash
maestro studio
```

## Test Flows

### Onboarding Tests

| Flow | Description | Tags |
|------|-------------|------|
| `onboarding_complete_flow.yaml` | Full onboarding happy path | onboarding, e2e, happy-path |
| `onboarding_skip_optional_steps.yaml` | Skip Health Background and Lifestyle | onboarding, e2e, skip-flow |
| `onboarding_navigation.yaml` | Back navigation and data persistence | onboarding, e2e, navigation |
| `onboarding_validation.yaml` | Form validation testing | onboarding, e2e, validation |

## Configuration

See `config.yaml` for global configuration:
- App ID
- Environment variables
- Timeout settings
- Retry behavior
- Screenshot/video recording

## Writing New Tests

1. Create a new YAML file in `flows/`
2. Start with app ID and metadata:
   ```yaml
   appId: com.nutri.app
   name: Test Name
   tags:
     - tag1
     - tag2
   ---
   ```
3. Add test steps using Maestro commands
4. Run the test to verify

## Debugging

### Interactive Mode
Use `maestro studio` to build tests interactively.

### Screenshots
Screenshots are taken automatically on failure and saved to `output/`.

### Logs
View test logs:
```bash
maestro test --debug .maestro/flows/test_name.yaml
```

## CI Integration

Add to your CI pipeline:
```yaml
- name: Run E2E Tests
  run: |
    npm run ios -- --no-install
    sleep 30
    maestro test .maestro/flows/ --format junit --output test-results.xml
```

## Troubleshooting

### App Not Found
Ensure the app is built and running:
```bash
npm run ios
```

### Element Not Found
- Check element IDs match in the app
- Use `maestro studio` to inspect elements
- Increase timeout in `config.yaml`

### Flaky Tests
- Add explicit waits: `- wait: 2000`
- Use `assertVisible` before interactions
- Increase retry count in config
