# Task ID: 16

**Title:** Integrate Continuous Glucose Monitor (CGM) Data Sources

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Add real-time glucose monitoring integration with major CGM platforms (Dexcom, Abbott Libre, Levels). This enables groundbreaking meal-specific glucose response tracking and positions Nutri as the premier nutrition-metabolic health app.

**Details:**

## Overview
CGM integration is the next frontier in personalized nutrition. By correlating individual meal consumption with glucose response curves, we can provide unprecedented insights into how specific foods affect each user.

## Technical Implementation

### 1. CGM Platform Integrations

#### Dexcom API
- OAuth 2.0 authentication
- Endpoints: /egvs (estimated glucose values)
- 5-minute reading intervals
- Developer portal: https://developer.dexcom.com/

#### Abbott LibreView API
- OAuth 2.0 authentication
- Historical glucose data access
- Developer access via partnership application

#### Levels Health API (for Levels users)
- REST API with API key
- Enriched glucose data with metabolic scores
- Partnership integration

### 2. Data Models

New Prisma models:
```prisma
model GlucoseReading {
  id          String   @id @default(uuid())
  userId      String
  value       Float    // mg/dL
  source      GlucoseSource
  recordedAt  DateTime
  user        User     @relation(fields: [userId], references: [id])
  
  @@index([userId, recordedAt])
}

enum GlucoseSource {
  DEXCOM
  LIBRE
  LEVELS
  MANUAL
}

model MealGlucoseResponse {
  id              String   @id @default(uuid())
  mealId          String   @unique
  meal            Meal     @relation(fields: [mealId], references: [id])
  baselineGlucose Float
  peakGlucose     Float
  peakTime        Int      // minutes after meal
  returnToBaseline Int     // minutes
  areaUnderCurve  Float
  glucoseScore    Float    // 0-100 metabolic response score
}
```

### 3. Meal-Glucose Correlation

New ML service: `ml-service/app/services/glucose_analysis.py`

Features:
- Detect meal timing from glucose spikes
- Calculate glucose response metrics per meal
- Identify problematic foods for individual users
- Learn personal glycemic index adjustments

### 4. API Endpoints

```
POST /api/v1/integrations/cgm/connect
POST /api/v1/integrations/cgm/sync
GET /api/v1/glucose/readings?from=&to=
GET /api/v1/glucose/meal-response/:mealId
GET /api/v1/glucose/insights
```

### 5. Mobile UI

- `app/settings/cgm-connect.tsx` - OAuth connection flow
- `app/(tabs)/glucose.tsx` - Glucose dashboard tab (optional)
- Glucose response card on meal detail view
- Real-time glucose widget on home screen
- Glucose trend visualization

### 6. Privacy & Security

- Encrypted storage of CGM credentials
- User consent flow for data access
- Data retention policies
- HIPAA considerations for health data

## Success Metrics
- Integration success rate: >90% successful connections
- Data sync reliability: >99% uptime
- User value: 80%+ of CGM users find insights valuable

## Dependencies
- Backend OAuth infrastructure
- Secure credential storage
- Existing meal logging system

**Test Strategy:**

1. Unit tests for glucose data processing
2. Mock OAuth flow tests
3. Integration tests for each CGM provider API
4. Test meal-glucose correlation calculations
5. E2E test: connect CGM → sync data → view meal response
6. Security audit for credential handling
7. Test reconnection and token refresh flows

## Subtasks

### 16.1. Add Prisma Schema for GlucoseReading and MealGlucoseResponse Models

**Status:** pending  
**Dependencies:** None  

Extend the existing Prisma schema with new models for CGM data storage: GlucoseReading for raw glucose values from CGM devices, MealGlucoseResponse for meal-specific glucose response analytics, CGMConnection for OAuth credential storage, and GlucoseSource enum for data provenance tracking.

**Details:**

1. Add GlucoseSource enum (DEXCOM, LIBRE, LEVELS, MANUAL) to `server/prisma/schema.prisma`
2. Create GlucoseReading model with fields: id (cuid), userId (relation to User), value (Float for mg/dL), source (GlucoseSource), trendArrow (optional String), recordedAt (DateTime), sourceId (optional), metadata (Json). Add indexes on [userId, recordedAt] and [userId, source, recordedAt]
3. Create MealGlucoseResponse model with fields: id, mealId (unique relation to Meal), baselineGlucose, peakGlucose, peakTime (Int minutes), returnToBaseline (Int minutes), areaUnderCurve, glucoseScore (0-100), analyzedAt DateTime
4. Create CGMConnection model for OAuth tokens: id, userId (unique), provider (GlucoseSource), accessToken (encrypted), refreshToken (encrypted), expiresAt, scope, lastSyncAt, isActive. Note: This codebase uses Prisma with PostgreSQL, so leverage existing patterns from User model for relations.
5. Add 'cgmConnections' and 'glucoseReadings' relations to User model
6. Run `npm run db:generate` to regenerate Prisma client after schema changes
7. Create and run migration with `npm run db:migrate`

### 16.2. Implement OAuth Integration Service for Dexcom and Libre APIs

**Status:** pending  
**Dependencies:** 16.1  

Create a new CGM OAuth integration service in the backend that handles OAuth 2.0 authentication flows for Dexcom and Abbott Libre APIs, including token management, refresh logic, and secure credential storage using encryption.

**Details:**

1. Create `server/src/services/cgmIntegrationService.ts` following existing service patterns (see healthMetricService.ts for structure)
2. Implement OAuth 2.0 authorization code flow for Dexcom API (https://developer.dexcom.com/):
   - Generate authorization URL with required scopes (egv.read)
   - Handle callback with authorization code exchange
   - Store encrypted tokens in CGMConnection model
   - Implement automatic token refresh using refreshToken before expiry
3. Implement similar flow for Abbott LibreView API (partnership API)
4. Add encryption utility in `server/src/utils/encryption.ts` for token storage (use crypto module with AES-256-GCM)
5. Add new environment variables: DEXCOM_CLIENT_ID, DEXCOM_CLIENT_SECRET, DEXCOM_REDIRECT_URI, LIBRE_CLIENT_ID, LIBRE_CLIENT_SECRET
6. Create controller `server/src/controllers/cgmController.ts` with methods: initiateConnect, handleCallback, disconnect, getConnectionStatus
7. Add routes in `server/src/routes/cgmRoutes.ts` following existing route patterns with authenticate middleware
8. Add Zod validation schemas in `server/src/validation/schemas.ts` for CGM endpoints
9. Add rate limiting for OAuth endpoints (5 requests/15 min per user) in rateLimiter.ts

### 16.3. Build Glucose Data Sync Service with Background Processing

**Status:** pending  
**Dependencies:** 16.1, 16.2  

Create a glucose sync service that fetches glucose readings from connected CGM platforms, handles incremental syncing, stores data efficiently, and supports real-time updates with proper error handling and retry logic.

**Details:**

1. Create `server/src/services/glucoseSyncService.ts` with methods:
   - syncDexcomReadings(userId, cgmConnection): Fetch EGV data from /v3/users/self/egvs endpoint, handle 5-minute intervals
   - syncLibreReadings(userId, cgmConnection): Fetch historical data from LibreView API
   - syncAll(userId): Orchestrate sync across all connected providers
   - getLastSyncTimestamp(userId, source): Get incremental sync start point
2. Implement incremental sync logic using lastSyncAt from CGMConnection, fetching only new readings since last sync
3. Create bulk insert method in glucoseReadingService using Prisma createMany with skipDuplicates (pattern from healthMetricService.createBulkHealthMetrics)
4. Add API endpoints in cgmRoutes.ts:
   - POST /api/cgm/sync (trigger manual sync)
   - GET /api/glucose/readings (with startDate, endDate, source query params)
   - GET /api/glucose/latest (most recent reading)
5. Handle API rate limits with exponential backoff (Dexcom: 60 requests/hour)
6. Map CGM trend arrows to standardized format (RISING, FALLING, STABLE, etc.)
7. Add error handling for common scenarios: expired tokens, API downtime, invalid data
8. Update CGMConnection.lastSyncAt after successful sync
9. Add glucose source type to HealthMetricSource enum or create dedicated GlucoseSource type

### 16.4. Develop Meal-Glucose Correlation Analysis in ML Service

**Status:** pending  
**Dependencies:** 16.1, 16.3  

Create a new glucose analysis service in the ML service that correlates meal consumption with glucose response curves, calculates metabolic response metrics (peak, baseline, AUC), and generates personalized glucose scores for each meal.

**Details:**

1. Create `ml-service/app/services/glucose_analysis.py` following existing service patterns (see food_analysis_service.py for structure)
2. Implement core analysis functions:
   - detect_meal_glucose_window(meal_time, glucose_readings, window_hours=3): Find glucose readings within meal impact window
   - calculate_baseline(readings, minutes_before=30): Average glucose before meal
   - find_peak(readings, meal_time): Identify peak glucose and time-to-peak
   - calculate_auc(readings, baseline, duration): Area under curve above baseline
   - calculate_glucose_score(peak, time_to_peak, return_time, auc): 0-100 metabolic response score
3. Create Pydantic schemas in `ml-service/app/schemas/glucose_analysis.py`: GlucoseReadingInput, MealGlucoseAnalysisRequest, MealGlucoseResponse
4. Add API routes in `ml-service/app/api/glucose_analysis.py`:
   - POST /analyze-meal-response: Analyze specific meal's glucose impact
   - POST /batch-analyze: Analyze multiple meals
   - GET /user-patterns/{user_id}: Get aggregate patterns for user
5. Implement personal glycemic index adjustment: Learn from historical data which foods cause higher responses for individual user
6. Add statistical analysis for identifying problematic foods (foods with consistently high glucose scores)
7. Create glucose_analysis router and register in main.py
8. Add Redis caching for computed meal responses (24h TTL) using existing redis_client pattern

### 16.5. Create Mobile UI for CGM Connection and Glucose Dashboard

**Status:** pending  
**Dependencies:** 16.2, 16.3, 16.4  

Build the mobile UI components for CGM platform connection flow (OAuth), real-time glucose display, meal-glucose response visualization, and glucose trends dashboard that integrates with the existing health tab and settings.

**Details:**

1. Create `app/settings/cgm-connect.tsx` - CGM connection settings screen:
   - List of available CGM platforms (Dexcom, Libre, Levels) with connection status
   - OAuth flow initiation via in-app browser (expo-auth-session or expo-web-browser)
   - Handle OAuth callback and store connection status
   - Disconnect option with confirmation dialog
   - Follow existing health-settings.tsx patterns for UI consistency
2. Create `lib/api/glucose.ts` - API client functions:
   - connectCGM(provider), disconnectCGM(), getCGMStatus()
   - syncGlucose(), getGlucoseReadings(params), getLatestGlucose()
   - getMealGlucoseResponse(mealId)
3. Add glucose display to meal detail view (`app/edit-meal/[id].tsx`):
   - GlucoseResponseCard component showing peak, baseline, score, AUC
   - Mini glucose curve chart using existing chart patterns
4. Create reusable components in `lib/components/glucose/`:
   - GlucoseValueDisplay: Current glucose with trend arrow and color coding
   - GlucoseTrendChart: 24h glucose line chart
   - MealGlucoseResponseCard: Metrics display for meal impact
5. Add glucose widget to home dashboard (`app/(tabs)/index.tsx`):
   - Current glucose reading with trend
   - Time since last reading
   - Quick link to detailed glucose view
6. Optional: Create `app/(tabs)/glucose.tsx` as dedicated glucose tab (or add section to health tab)
7. Register new screens in `app/_layout.tsx` with headerShown: false
8. Add Ionicons/IconSymbol for glucose-related icons (water-drop, analytics)
9. Use existing theme colors, spacing, and typography from `lib/theme/colors.ts`
