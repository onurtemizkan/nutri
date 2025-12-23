# Task ID: 5

**Title:** Implement Apple HealthKit Integration

**Status:** done

**Dependencies:** 3 ✓, 4 ✓

**Priority:** medium

**Description:** Enable automatic sync of comprehensive health data from Apple HealthKit including cardiovascular metrics (RHR, Heart Rate, HRV SDNN/RMSSD), sleep quality metrics (duration, deep sleep, REM, efficiency, score), respiratory data (respiratory rate, oxygen saturation), and VO2Max.

**Details:**

1. Install and configure react-native-health (recommended for Expo managed workflow):
   - Add `react-native-health` to package.json dependencies
   - Configure `app.json` with NSHealthShareUsageDescription and NSHealthUpdateUsageDescription
   - Add HealthKit entitlement to iOS build configuration
   - Request HealthKit permissions on iOS for all metric types

2. **Schema Consideration**: The existing Prisma HealthMetricType enum in `server/prisma/schema.prisma` already supports:
   - Cardiovascular: `RESTING_HEART_RATE`, `HEART_RATE_VARIABILITY_SDNN`, `HEART_RATE_VARIABILITY_RMSSD`
   - Respiratory: `RESPIRATORY_RATE`, `OXYGEN_SATURATION`, `VO2_MAX`
   - Sleep: `SLEEP_DURATION`, `DEEP_SLEEP_DURATION`, `REM_SLEEP_DURATION`, `SLEEP_EFFICIENCY`, `SLEEP_SCORE`
   - NOTE: Consider adding `HEART_RATE` (instantaneous/average HR, separate from resting) if needed for workout HR data

3. Create health sync service in `lib/services/healthkit.ts`:
```typescript
export interface HealthKitConfig {
  permissions: {
    read: HealthKitPermission[];
    write?: HealthKitPermission[];
  };
}

export const healthKitService = {
  // Initialize and request permissions
  requestPermissions: () => Promise<boolean>,
  isAvailable: () => Promise<boolean>,
  
  // Sync cardiovascular data
  syncCardiovascularMetrics: (startDate: Date, endDate: Date) => Promise<HealthMetric[]>,
  // - HKQuantityTypeIdentifierRestingHeartRate → RESTING_HEART_RATE
  // - HKQuantityTypeIdentifierHeartRate → instantaneous HR samples
  // - HKQuantityTypeIdentifierHeartRateVariabilitySDNN → HEART_RATE_VARIABILITY_SDNN
  // - HKQuantityTypeIdentifier.heartRateVariabilityRMSSD (if available)
  
  // Sync respiratory data
  syncRespiratoryMetrics: (startDate: Date, endDate: Date) => Promise<HealthMetric[]>,
  // - HKQuantityTypeIdentifierRespiratoryRate → RESPIRATORY_RATE
  // - HKQuantityTypeIdentifierOxygenSaturation → OXYGEN_SATURATION
  // - HKQuantityTypeIdentifierVO2Max → VO2_MAX
  
  // Sync sleep data
  syncSleepMetrics: (startDate: Date, endDate: Date) => Promise<HealthMetric[]>,
  // - HKCategoryTypeIdentifierSleepAnalysis → parse into:
  //   - SLEEP_DURATION (total sleep time)
  //   - DEEP_SLEEP_DURATION (deep/core sleep stages)
  //   - REM_SLEEP_DURATION (REM stages)
  //   - SLEEP_EFFICIENCY (time asleep / time in bed)
  //   - SLEEP_SCORE (if available from source)
  
  // Sync activity data
  syncActivityMetrics: (startDate: Date, endDate: Date) => Promise<HealthMetric[]>,
  // - HKQuantityTypeIdentifierStepCount → STEPS
  // - HKQuantityTypeIdentifierActiveEnergyBurned → ACTIVE_CALORIES
  // - HKWorkoutType → Activity model with exercise HR data
  
  // Background sync
  setupBackgroundSync: () => void,
}
```

4. HealthKit type identifiers mapping:
```typescript
const HEALTHKIT_TYPE_MAP = {
  // Cardiovascular
  HKQuantityTypeIdentifierRestingHeartRate: 'RESTING_HEART_RATE',
  HKQuantityTypeIdentifierHeartRateVariabilitySDNN: 'HEART_RATE_VARIABILITY_SDNN',
  // Note: RMSSD might need manual calculation from RR intervals
  
  // Respiratory
  HKQuantityTypeIdentifierRespiratoryRate: 'RESPIRATORY_RATE',
  HKQuantityTypeIdentifierOxygenSaturation: 'OXYGEN_SATURATION',
  HKQuantityTypeIdentifierVO2Max: 'VO2_MAX',
  
  // Sleep (requires parsing HKCategoryTypeIdentifierSleepAnalysis)
  // Sleep stages: inBed, asleepUnspecified, awake, asleepCore, asleepDeep, asleepREM
  
  // Activity
  HKQuantityTypeIdentifierStepCount: 'STEPS',
  HKQuantityTypeIdentifierActiveEnergyBurned: 'ACTIVE_CALORIES',
};
```

5. Implement data transformation layer:
   - Convert HealthKit units to API units (bpm, ms, %, steps, kcal, etc.)
   - Match existing Zod validation schemas in `server/src/validation/schemas.ts`
   - Use source: 'apple_health' to match healthMetricSourceSchema
   - Include device metadata: {device: "Apple Watch", quality: "high"}

6. Create sync flow:
   - Initial sync: Fetch last 30 days of data on first connect
   - Incremental sync: Fetch data since last sync timestamp
   - Store lastSyncTimestamp in Expo SecureStore (per metric type for efficiency)
   - Batch API calls using bulkCreateHealthMetricsSchema (50 items per request)
   - Handle timezone conversions (HealthKit returns local time, API expects UTC)

7. Handle data deduplication:
   - Use existing (userId, metricType, recordedAt, source) unique constraint in schema
   - Server handles conflicts via upsert
   - Store sourceId from HealthKit sample UUID for traceability

8. Add sync UI in profile settings (app/profile.tsx or new app/settings/health.tsx):
   - Connect/Disconnect Apple Health button
   - Permission status for each metric category
   - Last sync timestamp display (per category)
   - Manual sync button with progress indicator
   - Sync status indicator (syncing, synced, error)
   - Data preview showing recently synced metrics

9. Handle background sync (future enhancement):
   - Configure iOS background fetch capability
   - Sync when app becomes active via AppState listener
   - Respect battery and data usage settings
   - Consider using HealthKit's HKObserverQuery for real-time updates

**Test Strategy:**

1. Unit tests for HealthKit service:
   - Mock react-native-health module for simulator testing
   - Test permission request flow and error handling
   - Test data transformation for each metric type (HK format → API format)
   - Test unit conversions (HK units → standard units)

2. Test cardiovascular data sync:
   - Mock RHR samples → verify RESTING_HEART_RATE records
   - Mock HRV samples → verify HEART_RATE_VARIABILITY_SDNN records
   - Test edge cases: missing data, invalid values

3. Test respiratory data sync:
   - Mock respiratory rate → verify RESPIRATORY_RATE records
   - Mock SpO2 → verify OXYGEN_SATURATION records
   - Mock VO2Max → verify VO2_MAX records

4. Test sleep data parsing:
   - Mock sleep analysis categories → verify correct stage classification
   - Test sleep efficiency calculation (time asleep / time in bed)
   - Test sleep duration aggregation across fragmented sleep

5. Test sync error handling and retry logic:
   - Network errors during bulk upload
   - Partial failures in batch operations
   - Permission denied scenarios

6. Test deduplication with existing data:
   - Re-sync same data → verify no duplicates
   - Test sourceId matching

7. Test UI state updates during sync:
   - Loading states
   - Progress indication
   - Error display

8. Manual testing on physical device:
   - Test with real Apple Watch data
   - Verify data accuracy against Health app
   - Test permission prompts
   - Test background sync behavior

## Subtasks

### 5.1. Install and configure react-native-health package

**Status:** done  
**Dependencies:** None  

Add react-native-health dependency and configure iOS build settings with proper entitlements and Info.plist descriptions

**Details:**

Install react-native-health via npm/yarn. Update app.json with NSHealthShareUsageDescription explaining why the app needs read access to health data. Add HealthKit entitlement. Configure iOS build to include HealthKit framework. Test that the package builds correctly on iOS simulator/device.

### 5.2. Create HealthKit permission management system

**Status:** done  
**Dependencies:** 5.1  

Implement permission request flow for all required HealthKit data types including cardiovascular, respiratory, and sleep metrics

**Details:**

Create lib/services/healthkit/permissions.ts. Define permission sets for each metric category. Implement isAvailable() check for HealthKit. Implement requestPermissions() with granular permission requests. Handle partial permission grants gracefully. Store permission status in context/state.

### 5.3. Implement cardiovascular metrics sync (RHR, HR, HRV)

**Status:** done  
**Dependencies:** 5.2  

Create sync functions for Resting Heart Rate, Heart Rate samples, and Heart Rate Variability (SDNN and RMSSD)

**Details:**

Implement syncCardiovascularMetrics() in healthkit.ts. Query HKQuantityTypeIdentifierRestingHeartRate for RHR. Query HKQuantityTypeIdentifierHeartRateVariabilitySDNN for HRV. Transform HealthKit samples to match HealthMetric API format. Handle unit conversions (HK returns bpm/ms). Include device metadata from sample source.

### 5.4. Implement respiratory metrics sync (respiratory rate, SpO2, VO2Max)

**Status:** done  
**Dependencies:** 5.2  

Create sync functions for respiratory rate, oxygen saturation, and VO2Max data from HealthKit

**Details:**

Implement syncRespiratoryMetrics() in healthkit.ts. Query HKQuantityTypeIdentifierRespiratoryRate, HKQuantityTypeIdentifierOxygenSaturation, and HKQuantityTypeIdentifierVO2Max. Transform to RESPIRATORY_RATE, OXYGEN_SATURATION, and VO2_MAX metric types. Handle different sample frequencies (VO2Max is less frequent).

### 5.5. Implement sleep metrics sync with stage classification

**Status:** done  
**Dependencies:** 5.2  

Create sync function for sleep analysis including duration, deep sleep, REM, and efficiency calculations

**Details:**

Implement syncSleepMetrics() in healthkit.ts. Query HKCategoryTypeIdentifierSleepAnalysis. Parse sleep stages (asleepCore→DEEP_SLEEP, asleepDeep→DEEP_SLEEP, asleepREM→REM_SLEEP). Calculate total SLEEP_DURATION. Calculate SLEEP_EFFICIENCY (asleep time / in bed time). Handle fragmented sleep sessions.

### 5.6. Implement activity metrics sync (steps, active calories)

**Status:** done  
**Dependencies:** 5.2  

Create sync function for daily activity data including step count and active energy burned

**Details:**

Implement syncActivityMetrics() in healthkit.ts. Query HKQuantityTypeIdentifierStepCount and HKQuantityTypeIdentifierActiveEnergyBurned. Aggregate daily totals. Transform to STEPS and ACTIVE_CALORIES metric types. Handle timezone boundaries for daily aggregation.

### 5.7. Create batch sync orchestration with API integration

**Status:** done  
**Dependencies:** 5.3, 5.4, 5.5, 5.6  

Implement the main sync orchestration that coordinates all metric syncs and uploads to the backend API

**Details:**

Create syncAllHealthData() coordinator function. Implement incremental sync using stored lastSyncTimestamp from SecureStore. Batch API uploads using bulkCreateHealthMetricsSchema (50 items per request). Handle partial failures and retry logic. Update lastSyncTimestamp per metric category on success.

### 5.8. Build HealthKit settings UI with sync controls

**Status:** done  
**Dependencies:** 5.7  

Create the user interface for managing HealthKit connection, viewing sync status, and triggering manual syncs

**Details:**

Create app/settings/health.tsx screen or add section to app/profile.tsx. Display Connect/Disconnect Apple Health button. Show permission status per metric category with toggle indicators. Display last sync timestamp and synced data counts. Add manual Sync Now button with progress indicator. Show sync status (syncing/synced/error) with appropriate feedback.
