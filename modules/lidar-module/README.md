# LiDAR Native Module — Developer Instructions

This repository contains a small native module (`LiDARModule`) that exposes iOS LiDAR and ARKit Scene Depth functionality to the JS app. When running in Expo Go or environments without the native bridge, the app uses a JS fallback.

## When do you need a native dev client?
- To access true LiDAR/ARKit depth frames on a physical iOS device, you must run a build that includes the native iOS module. Expo Go does not include custom native modules.

## Option A — EAS Development Build (recommended)
1. Install EAS CLI (one-time):

```bash
npm install -g eas-cli
```

2. Log in and configure (if needed):

```bash
eas login
# follow prompts
```

3. Create a development build for iOS:

```bash
eas build --profile development --platform ios
```

4. Install the generated dev client on your device (EAS will provide a link), then open the project using the EAS dev client. The `LiDARModule` native bridge will be available.

Notes:
- This requires an Apple developer account to install on physical devices (unless using TestFlight/adhoc provisioning provided by EAS).
- Use `eas build --help` to set credentials or configure `eas.json` for custom profiles.

## Option B — Local prebuild + Xcode (macOS)
1. Prebuild native projects:

```bash
npx expo prebuild
```

2. Install CocoaPods in the `ios` directory:

```bash
npx pod-install ios
```

3. Open Xcode and run on a physical device (recommended for LiDAR):

```bash
open ios/YourApp.xcworkspace
# Build & run from Xcode onto your device
```

Notes:
- Make sure the `modules/lidar-module/expo-module.config.json` is present (it is), so the prebuild includes the native module.
- Building from Xcode gives you full native debugging tools.

## Quick checks
- In JS, the module exports `isAvailable` — you can check it to determine whether native LiDAR is present:

```ts
import LiDARModule from '@/lib/modules/LiDARModule';
if (LiDARModule.isAvailable) {
  // use LiDARModule
} else {
  // fallback behavior (simulator / message)
}
```

## Troubleshooting
- If you still see `(NOBRIDGE)` errors after installing a dev client, ensure you:
  - Rebuild the dev client after changing native code.
  - Open the app in the installed dev client (not Expo Go).
  - Confirm the native module was compiled into the app by checking Xcode build logs.

If you want, I can add a short CLI script to automate local prebuild + run steps.
