# Task ID: 15

**Title:** Build What-If Simulation Engine for Nutrition Impact Prediction

**Status:** pending

**Dependencies:** 13

**Priority:** high

**Description:** Create an interactive simulation feature that allows users to see predicted health impacts of potential nutrition changes before making them. This is a unique differentiator leveraging our ML correlation engine that no competitor offers.

**Details:**

## Overview
What-If Simulation answers: "If I change my diet in X way, how will it affect my health metrics?" This transforms our ML predictions from passive insights to actionable planning tools.

## Technical Implementation

### 1. Simulation Engine (ML Service)
Create new service: `ml-service/app/services/simulation_engine.py`

Core functionality:
- Accept hypothetical nutrition changes (e.g., +20g protein daily)
- Use trained LSTM models to predict health metric changes
- Calculate confidence intervals for predictions
- Return time-series projections (7-day, 14-day, 30-day)

### 2. Simulation API Endpoints
```python
POST /api/v1/simulate/nutrition-change
{
  "user_id": "uuid",
  "changes": [
    {"nutrient": "protein", "delta": 20, "unit": "g"},
    {"nutrient": "sugar", "delta": -15, "unit": "g"}
  ],
  "duration_days": 14,
  "metrics_to_predict": ["rhr", "hrv_rmssd", "recovery_score"]
}

Response:
{
  "predictions": [
    {
      "metric": "rhr",
      "baseline": 62,
      "projected": 59,
      "confidence_interval": [57, 61],
      "trajectory": [62, 61, 61, 60, 60, 59, 59, ...]
    }
  ],
  "confidence_score": 0.78,
  "based_on_correlations": [...]
}
```

### 3. Mobile UI Components
- `app/simulate.tsx` - Main simulation screen
- Slider controls for nutrition adjustments
- Real-time prediction visualization
- Before/after comparison charts
- Save simulation as "goal" feature

### 4. Visualization
- Line chart showing projected metric trajectory
- Confidence band visualization (shaded area)
- Color coding: green (improvement), red (concern), gray (neutral)
- Comparison overlay with current baseline

### 5. Leveraging Existing ML
Build on `correlation_engine.py`:
- Use discovered correlations to weight predictions
- Apply time-lag findings (e.g., protein affects HRV after 48h)
- Use per-user trained models for personalized predictions

### 6. Safety Guardrails
- Warn if simulated changes are extreme (>50% change)
- Note that predictions are estimates, not medical advice
- Cap prediction confidence for users with <30 days data

## Success Metrics
- Simulation accuracy: predictions within 15% of actual outcomes
- User engagement: 30%+ users try simulation feature
- Goal conversion: 20%+ of simulations saved as goals

## Dependencies
- Task 13: USDA database (for accurate nutrition data)
- Existing correlation engine and LSTM models

**Test Strategy:**

1. Unit tests for simulation engine calculations
2. Mock prediction model tests
3. Integration tests for full simulation pipeline
4. Validation: run simulations on historical data, compare to actual outcomes
5. E2E test: create simulation → verify predictions displayed
6. Test edge cases (insufficient data, extreme values)

## Subtasks

### 15.1. Create SimulationEngine Service with Multi-Day Time-Series Predictions

**Status:** pending  
**Dependencies:** None  

Extend the existing what_if.py service to create a new SimulationEngine that generates multi-day time-series projections (7/14/30 days) with confidence intervals for hypothetical nutrition changes.

**Details:**

Create ml-service/app/services/simulation_engine.py building on existing WhatIfService:

1. SimulationEngine class with:
   - simulate_nutrition_impact() - Main method accepting nutrition change deltas (e.g., {protein: +20g, sugar: -15g})
   - generate_trajectory() - Uses trained LSTM models to predict day-by-day health metric values
   - calculate_confidence_bands() - Computes 95% confidence intervals using model MAE from validation
   - apply_time_lag_effects() - Leverages correlation_engine.py's lag analysis (e.g., protein affects HRV after 48h)

2. Integrate with existing components:
   - Use HealthMetricLSTM from ml_models/lstm.py for predictions
   - Use DataPreparationService for feature preparation
   - Use CorrelationEngineService to apply discovered time-lag correlations
   - Load per-user trained models from models_dir

3. Time-series generation:
   - For each day in projection period:
     a. Apply hypothetical nutrition changes to feature vector
     b. Run LSTM prediction
     c. Use prediction as input for next day (autoregressive)
     d. Accumulate uncertainty (confidence bands widen over time)

4. Return SimulationTrajectory with: metric_name, baseline_values[], projected_values[], confidence_lower[], confidence_upper[], timestamps[]

### 15.2. Create Pydantic Schemas and API Endpoints for Simulation Requests

**Status:** pending  
**Dependencies:** 15.1  

Define new Pydantic schemas for simulation requests/responses and create FastAPI endpoints at /api/v1/simulate/ for the simulation engine.

**Details:**

Create ml-service/app/schemas/simulation.py with:

1. SimulationRequest schema:
   - user_id: str
   - changes: List[NutritionChange] where NutritionChange has nutrient (protein/carbs/fat/calories/fiber/sugar), delta (float), unit (g/kcal)
   - duration_days: int (7, 14, or 30)
   - metrics_to_predict: List[PredictionMetric] (rhr, hrv_rmssd, hrv_sdnn, recovery_score)

2. SimulationResponse schema:
   - predictions: List[MetricPrediction]
   - MetricPrediction has: metric, baseline_value, projected_final_value, confidence_interval (lower, upper), trajectory (list of daily values), trajectory_confidence (list of {day, lower, upper})
   - confidence_score: float (0-1)
   - based_on_correlations: List[CorrelationReference]
   - warnings: List[str]

3. Create ml-service/app/api/simulation.py:
   - POST /simulate/nutrition-change - Main simulation endpoint
   - GET /simulate/supported-nutrients - List nutrients that can be simulated
   - GET /simulate/user-readiness/{user_id} - Check if user has sufficient data/trained model

4. Register router in ml-service/app/main.py under /api/v1/simulate prefix

5. Add request validation using existing PredictionMetric enum from schemas/predictions.py

### 15.3. Build Mobile Simulation Screen with Nutrient Adjustment Sliders

**Status:** pending  
**Dependencies:** 15.2  

Create the main simulation screen (app/simulate.tsx) with interactive slider controls for adjusting nutrition values and triggering simulations.

**Details:**

Create app/simulate.tsx following the patterns from health.tsx:

1. Screen structure:
   - Header with back navigation and title 'What-If Simulation'
   - Nutrient adjustment section with sliders
   - Duration selector (7/14/30 days)
   - Metric selection (checkboxes for RHR, HRV, Recovery Score)
   - 'Run Simulation' button with gradient styling

2. Slider controls (using @react-native-community/slider or custom):
   - Protein: -50g to +50g (step: 5g)
   - Carbohydrates: -100g to +100g (step: 10g)
   - Fat: -30g to +30g (step: 5g)
   - Calories: -500 to +500 (step: 50)
   - Sugar: -30g to +30g (step: 5g)
   - Each slider shows current delta value and visual indicator

3. State management:
   - nutritionChanges: Record<string, number>
   - selectedMetrics: PredictionMetric[]
   - durationDays: 7 | 14 | 30
   - isLoading, error, simulationResult states

4. Create lib/api/simulation.ts:
   - simulateNutritionChange(request) - POST to /api/v1/simulate/nutrition-change
   - getSimulationReadiness(userId) - Check user data availability

5. Add responsive design using useResponsive hook pattern
6. Register route in app/_layout.tsx with headerShown: false

### 15.4. Implement Trajectory Visualization with Confidence Bands

**Status:** pending  
**Dependencies:** 15.3  

Create chart components to visualize predicted health metric trajectories with confidence interval bands and before/after comparisons.

**Details:**

Create lib/components/SimulationChart.tsx:

1. Line chart showing:
   - X-axis: Days (0 to duration)
   - Y-axis: Metric value (auto-scaled)
   - Solid line: Projected trajectory
   - Dashed line: Baseline (current trend continuation)
   - Shaded area: 95% confidence interval band

2. Use react-native-chart-kit or victory-native for charting:
   - Configure area chart for confidence band
   - Overlay line chart for trajectory
   - Custom tooltip showing day, value, and confidence range

3. Color coding:
   - Green gradient for improvement areas
   - Red gradient for concerning predictions
   - Gray for neutral/unchanged
   - Color logic based on metric type (lower RHR = good, higher HRV = good)

4. Create SimulationResultCard.tsx:
   - Metric name header
   - Current value vs projected final value
   - Percentage change indicator with color
   - Confidence score badge
   - Chart component embedded

5. Add comparison overlay feature:
   - Toggle to show/hide baseline comparison
   - Visual diff highlighting

6. Create lib/components/SimulationSummary.tsx:
   - Overall impact summary text
   - Key correlations that drove the prediction
   - Warning messages if applicable

### 15.5. Add Safety Guardrails and Validation Layer

**Status:** pending  
**Dependencies:** 15.1, 15.2, 15.3, 15.4  

Implement safety guardrails to warn users about extreme changes, validate data sufficiency, and ensure predictions include appropriate disclaimers.

**Details:**

Add validation and safety features across the stack:

1. In simulation_engine.py:
   - validate_changes() - Reject changes >50% of typical daily values
   - check_data_sufficiency() - Require minimum 30 days of data for predictions
   - cap_confidence() - Reduce confidence score if user has <30 days data or if model R² <0.5
   - detect_extrapolation() - Warn if predicted values fall outside historical range

2. Guardrail thresholds:
   - Protein: max ±100g/day
   - Carbs: max ±200g/day
   - Fat: max ±100g/day
   - Calories: max ±1000kcal/day
   - Flag any combination totaling >30% calorie change

3. Warning messages in SimulationResponse:
   - 'extreme_change': 'This represents a significant dietary change. Consult a nutritionist.'
   - 'low_confidence': 'Prediction confidence is reduced due to limited historical data.'
   - 'extrapolation': 'Predicted values are outside your historical range.'
   - 'disclaimer': 'These predictions are estimates and should not replace medical advice.'

4. Mobile UI warnings (app/simulate.tsx):
   - Display warning banner when extreme values detected
   - Show medical disclaimer at bottom of results
   - Disable run button if validation fails
   - Color-code slider values that exceed safe thresholds

5. Add 'Save as Goal' feature:
   - Button to save successful simulation as nutrition goal
   - Store target values in user profile
   - Link to existing goal tracking system
