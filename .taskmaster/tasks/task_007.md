# Task ID: 7

**Title:** Create Predictions Visualization Mobile UI

**Status:** pending

**Dependencies:** 6

**Priority:** medium

**Description:** Build mobile screens to display ML predictions (RHR, HRV forecasts) with confidence intervals and historical context.

**Details:**

1. Create new screens:
   - `app/predictions/index.tsx` - Predictions dashboard
   - `app/predictions/[metric].tsx` - Detailed prediction view

2. Predictions Dashboard (`app/predictions/index.tsx`):
   - Card for each predictable metric (RHR, HRV)
   - Display: predicted value, confidence score, direction indicator
   - Comparison to 30-day average
   - 'No prediction available' state if model not trained
   - Pull-to-refresh to get latest predictions

3. Detailed Prediction View (`app/predictions/[metric].tsx`):
   - Chart showing:
     - Historical values (last 30 days)
     - Predicted value for tomorrow
     - Confidence interval as shaded region
   - Interpretation text (AI-generated explanation)
   - Recommendation based on prediction
   - Feature importance breakdown (what drove this prediction)

4. Create API client in `lib/api/predictions.ts`:
```typescript
export const predictionsApi = {
  predict: (metric: string, targetDate: string) =>
    apiClient.post('/api/predictions/predict', { metric, target_date: targetDate }),
  batchPredict: (metrics: string[], targetDate: string) =>
    apiClient.post('/api/predictions/batch-predict', { metrics, target_date: targetDate }),
  listModels: () => apiClient.get('/api/predictions/models'),
}
```

5. Chart implementation:
   - Use Victory Native or react-native-chart-kit
   - Line chart for historical + predicted
   - Shaded area for confidence interval
   - Animate prediction point

6. Handle states:
   - Loading: Show skeleton
   - No model trained: Show CTA to collect more data
   - Prediction available: Show full UI
   - Error: Show error message with retry

**Test Strategy:**

1. Component tests for dashboard and detail screens
2. Test chart rendering with mock data
3. Test loading/error/empty states
4. Test confidence interval visualization
5. Test API integration with mock responses
6. Snapshot tests for consistent UI

## Subtasks

### 7.1. Create Predictions API Client in lib/api/predictions.ts

**Status:** pending  
**Dependencies:** None  

Implement the TypeScript API client for predictions endpoints following the existing pattern from health-metrics.ts

**Details:**

Create lib/api/predictions.ts with proper TypeScript types matching the ML service schemas (PredictionResult, PredictResponse, BatchPredictResponse, ModelInfo). Implement methods: predict(userId, metric, targetDate), batchPredict(userId, metrics, targetDate), listModels(userId). Use the existing axios client from lib/api/client.ts. Create corresponding TypeScript interfaces in lib/types/predictions.ts for PredictionMetric enum (RHR, HRV_SDNN, HRV_RMSSD, SLEEP_DURATION, RECOVERY_SCORE), PredictionResult interface with predicted_value, confidence_interval_lower/upper, confidence_score, historical_average, deviation_from_average, percentile fields, and ModelInfo interface. Follow the error handling pattern from health-metrics.ts with isNotFoundError helper.

### 7.2. Build Predictions Dashboard Screen at app/predictions/index.tsx

**Status:** pending  
**Dependencies:** 7.1  

Create the main predictions dashboard showing prediction cards for RHR and HRV metrics with pull-to-refresh functionality

**Details:**

Create app/predictions/index.tsx following the pattern from app/(tabs)/health.tsx. Include: SafeAreaView with ScrollView, custom header with back button, metric prediction cards displaying predicted_value, confidence_score (as percentage), direction indicator (trending-up/down icons), comparison to 30-day historical_average. Use the batchPredict API to fetch predictions for ['RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_SDNN'] on screen load. Implement pull-to-refresh using RefreshControl. Cards should be TouchableOpacity navigating to /predictions/[metric]. Register the screen in app/_layout.tsx with headerShown: false and slide_from_right animation.

### 7.3. Create Detailed Prediction View at app/predictions/[metric].tsx

**Status:** pending  
**Dependencies:** 7.1, 7.2  

Build the detailed prediction screen with historical chart, confidence interval visualization, and AI interpretation

**Details:**

Create app/predictions/[metric].tsx following app/health/[metricType].tsx pattern. Include: custom header with metric name, LineChart from react-native-chart-kit showing last 30 days historical data plus predicted value for tomorrow. Use healthMetricsApi.getTimeSeries() for historical data and predictionsApi.predict() for the prediction. Display confidence interval as shaded region (use chartConfig with fillShadowGradient). Show interpretation text from PredictResponse.interpretation, recommendation from PredictResponse.recommendation, and a 'Feature Importance' section (placeholder for now, can show key features like 'Recent sleep patterns', 'Nutrition quality'). Register in app/_layout.tsx with slide_from_right animation.

### 7.4. Implement Chart Components with Confidence Interval Visualization

**Status:** pending  
**Dependencies:** 7.1  

Create reusable chart component that displays historical values with predicted value and confidence interval shading

**Details:**

Create lib/components/PredictionChart.tsx as a reusable component wrapping react-native-chart-kit LineChart. Props: historicalData (TimeSeriesDataPoint[]), predictedValue (number), confidenceIntervalLower (number), confidenceIntervalUpper (number), metricType (for formatting). Component should: merge historical data with prediction point, use different colors for historical (purple) vs predicted (orange), animate the predicted point with a pulse effect using react-native Animated API. For confidence interval, use a semi-transparent fill between the bounds - this may require overlaying multiple datasets or using a custom decorator. Include proper chart config matching the existing health metric charts (colors.background.tertiary, borderRadius.lg, etc.).

### 7.5. Implement Empty, Loading, and Error States with Model Availability Checks

**Status:** pending  
**Dependencies:** 7.1, 7.2, 7.3  

Add comprehensive state handling for loading, error, no model trained, and successful prediction states

**Details:**

Enhance both prediction screens with proper state handling following the existing patterns in health.tsx and [metricType].tsx. States: 1) Loading - ActivityIndicator with skeleton placeholders, 2) Error - error card with Ionicons alert-circle-outline, error message, and retry button, 3) No Model Trained - custom empty state with analytics-outline icon, 'No prediction model available' title, subtitle explaining 'We need at least 30 days of data to train a prediction model', and CTA button to navigate to health tracking, 4) Model Not Production Ready - warning state showing quality issues list from TrainModelResponse.quality_issues. Use the listModels() API to check model availability before attempting prediction. Add error boundaries to catch unexpected failures gracefully.
