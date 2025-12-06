# ML Service Setup Guide

## Problem Fixed
The ML service was not accessible from your physical iPhone because the app was trying to connect to `localhost:8000`, which only works on your Mac. Physical devices need to use your machine's actual IP address on the network.

## Your Machine's IP Address
**192.168.1.69**

This is automatically configured in the new development scripts.

## Service Status
✅ **ML Service Running** on port 8000
- Health: http://localhost:8000/api/food/health
- Analyze: http://localhost:8000/api/food/analyze

## How to Use on Physical iPhone with Expo Go

### Step 1: Start All Services
```bash
npm run dev:full
```
This starts:
- ✅ Docker (PostgreSQL + Redis)
- ✅ Backend API (port 3000)
- ✅ ML Service (port 8000)

### Step 2: Start Expo Dev Server with ML Service URL
For **physical device (iPhone)**:
```bash
npm run start:device
```

This automatically sets:
- `ML_SERVICE_URL=http://192.168.1.69:8000`

For **simulator only**:
```bash
npm start
```

### Step 3: Connect Your iPhone
1. Make sure your iPhone is on the **same WiFi network** (192.168.1.x)
2. Open Expo Go app
3. Scan the QR code from the terminal
4. App should connect and load

### Step 4: Test Food Analysis
1. Navigate to the scan-food page
2. Take a photo of food
3. It should now successfully analyze with the ML service

## Network Requirements
- Your Mac and iPhone must be on the **same WiFi network**
- The iOS device must be able to reach `192.168.1.69:8000`
- Check: Can you ping 192.168.1.69 from iPhone? (using Terminal app)

## Troubleshooting

### Still getting "ML service unavailable"?

1. **Verify ML service is running:**
   ```bash
   curl http://localhost:8000/api/food/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Check network connectivity from iPhone:**
   - Verify both devices are on same WiFi
   - Try: `curl http://192.168.1.69:8000/api/food/health` from your Mac to verify

3. **Update the IP address if it changed:**
   Edit `package.json` and update `start:device` script with your new IP:
   ```json
   "start:device": "ML_SERVICE_URL=http://YOUR_IP:8000 expo start"
   ```
   Get IP: `ipconfig getifaddr en0`

## Environment Variables

You can override ML service URL:
```bash
ML_SERVICE_URL=http://192.168.1.69:8000 npm start
```

This is automatically handled by:
1. `app.config.js` reads `ML_SERVICE_URL` env var
2. `lib/api/food-analysis.ts` uses the config value

## File Changes Made

1. ✅ **app.config.js**: Added ML_SERVICE_URL env var support
2. ✅ **lib/api/food-analysis.ts**: Updated to read from app config
3. ✅ **package.json**: Added `start:device` script

## Development Workflow

```bash
# Terminal 1: Start all backend services
npm run dev:full

# Terminal 2: Start Expo dev server (for physical device)
npm run start:device

# Terminal 3: Connect from iPhone Expo Go app and scan QR code
```

Food analysis should now work! 🎉
