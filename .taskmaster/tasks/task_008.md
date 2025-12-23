# Task ID: 8

**Title:** Implement ML Insights Engine and Recommendations

**Status:** pending

**Dependencies:** 6

**Priority:** medium

**Description:** Build the insights generation system that analyzes correlations and generates personalized nutrition recommendations stored in MLInsight model.

**Details:**

1. Create insights service in `ml-service/app/services/insights_engine.py`:
```python
class InsightsEngine:
    async def generate_insights(self, user_id: str) -> List[MLInsight]:
        correlations = await self._get_significant_correlations(user_id)
        predictions = await self._get_recent_predictions(user_id)
        anomalies = await self._detect_anomalies(user_id)
        
        insights = []
        insights.extend(self._correlation_insights(correlations))
        insights.extend(self._prediction_insights(predictions))
        insights.extend(self._anomaly_insights(anomalies))
        insights.extend(self._goal_progress_insights(user_id))
        
        return self._prioritize_and_limit(insights, max_insights=5)
```

2. Insight types to implement:
   - CORRELATION: 'Your protein intake correlates with better HRV (+0.65)'
   - PREDICTION: 'Tomorrow's RHR is predicted higher than average'
   - ANOMALY: 'Your sleep duration last night was unusually low'
   - RECOMMENDATION: 'Try eating dinner earlier to improve sleep quality'
   - GOAL_PROGRESS: 'You're 80% of the way to your protein goal this week'
   - PATTERN_DETECTED: 'You tend to eat more carbs on weekends'

3. Correlation-based recommendations:
   - Use CorrelationEngineService to find significant correlations
   - Filter by correlation strength (|r| > 0.5)
   - Generate natural language recommendations
   - Example: If protein ↔ HRV has r=0.7, recommend 'Increasing protein may improve your HRV'

4. Anomaly detection:
   - Z-score based detection (>2 std from 30-day mean)
   - Detect unusual: meal timing, calorie intake, sleep duration
   - Generate alerts for negative anomalies

5. Create API endpoints in `ml-service/app/api/insights.py`:
   - GET /api/insights - List user's active insights
   - POST /api/insights/generate - Trigger insight generation
   - PUT /api/insights/{id}/viewed - Mark as viewed
   - PUT /api/insights/{id}/dismissed - Dismiss insight
   - PUT /api/insights/{id}/feedback - Submit helpful/not helpful

6. Store insights in database using MLInsight model (already defined in Prisma schema)

**Test Strategy:**

1. Unit tests for each insight type generator
2. Test insight prioritization logic
3. Test anomaly detection thresholds
4. Test natural language generation
5. Integration test: end-to-end insight generation
6. Test user feedback tracking
7. Test insight expiration handling

## Subtasks

### 8.1. Create Insights Engine Service with Correlation-Based Insight Generation

**Status:** pending  
**Dependencies:** None  

Implement the core InsightsEngine class in ml-service/app/services/insights_engine.py that generates correlation-based insights by leveraging the existing CorrelationEngineService.

**Details:**

Create insights_engine.py with InsightsEngine class that:

1. Initialize with AsyncSession and instantiate CorrelationEngineService and PredictionService
2. Implement `async def generate_insights(user_id: str) -> List[MLInsight]` as main entry point
3. Implement `_get_significant_correlations(user_id)` method:
   - Call CorrelationEngineService.analyze_correlations() for key health metrics (RHR, HRV_SDNN, HRV_RMSSD, SLEEP_DURATION, RECOVERY_SCORE)
   - Filter correlations with |r| > 0.5 and p_value < 0.05
   - Return list of significant correlation results
4. Implement `_correlation_insights(correlations)` method:
   - Convert CorrelationResult objects to MLInsight format
   - Set insight_type = MLInsightType.CORRELATION
   - Calculate priority based on correlation strength (|r| > 0.7 = HIGH, |r| > 0.5 = MEDIUM)
   - Generate title like 'Your protein intake correlates with better HRV'
   - Generate description explaining the correlation direction and strength
   - Include correlation coefficient in the correlation field
5. Follow existing service patterns from correlation_engine.py and prediction.py for database session handling and error handling

### 8.2. Implement Z-Score Anomaly Detection for Health and Nutrition Data

**Status:** pending  
**Dependencies:** 8.1  

Add anomaly detection methods to InsightsEngine using z-score analysis to identify unusual values in meal timing, calorie intake, sleep duration, and other health metrics.

**Details:**

Extend InsightsEngine with anomaly detection capabilities:

1. Implement `async def _detect_anomalies(user_id: str) -> List[AnomalyResult]`:
   - Query last 30 days of data for each metric type using HealthMetric model
   - Calculate rolling 30-day mean and standard deviation for each metric
   - Compute z-score for today's/yesterday's values: z = (value - mean) / std
   - Flag as anomaly if |z| > 2.0 (configurable threshold)

2. Create AnomalyResult dataclass with fields:
   - metric_type: str
   - value: float
   - z_score: float
   - mean_30d: float
   - std_30d: float
   - direction: 'high' | 'low'
   - severity: 'warning' | 'critical' (|z| > 3 = critical)

3. Implement `_anomaly_insights(anomalies: List[AnomalyResult]) -> List[MLInsight]`:
   - Convert anomalies to MLInsight with insight_type = MLInsightType.ANOMALY
   - Set priority = HIGH for critical, MEDIUM for warning
   - Generate titles like 'Your sleep duration last night was unusually low'
   - Include z_score and deviation in metadata JSON

4. Detect anomalies for:
   - SLEEP_DURATION (low sleep is concerning)
   - Meal calorie totals (sudden spikes/drops)
   - Meal timing (late dinners, skipped meals)
   - RESTING_HEART_RATE (elevated RHR)
   - Exercise duration (unusual patterns)

5. Use scipy.stats for z-score calculation, similar to correlation_engine.py patterns

### 8.3. Build Natural Language Recommendation Generator with Template System

**Status:** pending  
**Dependencies:** 8.1, 8.2  

Create a recommendation generation system that transforms correlation data, anomalies, predictions, and goal progress into human-readable, actionable recommendations.

**Details:**

Implement natural language generation for insights:

1. Create `ml-service/app/services/nlg_recommendations.py` with NLGRecommendationGenerator class:

2. Implement template-based generation for each insight type:
   - CORRELATION templates:
     * Positive: 'Increasing {feature} may improve your {metric}' 
     * Negative: 'Reducing {feature} could help lower your {metric}'
   - ANOMALY templates:
     * Low sleep: 'Your sleep duration was {hours}h last night, well below your average of {avg}h. Consider an earlier bedtime tonight.'
     * High RHR: 'Your resting heart rate is elevated at {value} bpm. This could indicate stress or inadequate recovery.'
   - PREDICTION templates:
     * Above average: 'Tomorrow\'s {metric} is predicted to be {direction} than your average ({value} vs {avg})'
   - GOAL_PROGRESS templates:
     * 'You\'re {percent}% of the way to your {goal_type} goal this week'
   - PATTERN_DETECTED templates:
     * 'You tend to {pattern_description} on {temporal_pattern}'

3. Implement `generate_recommendation(insight_type, data_context)` method:
   - Select appropriate template based on insight_type and context
   - Fill template variables with actual values
   - Format numbers appropriately (decimals, percentages)
   - Return title, description, and recommendation strings

4. Add personalization:
   - Use metric display names from PredictionService._get_metric_display_name pattern
   - Include specific numerical values for credibility
   - Provide actionable suggestions (not just observations)

5. Integrate with InsightsEngine:
   - Call NLGRecommendationGenerator from _correlation_insights, _anomaly_insights, etc.
   - Populate MLInsight title, description, and recommendation fields

### 8.4. Create FastAPI Endpoints for Insights CRUD Operations

**Status:** pending  
**Dependencies:** 8.1, 8.2, 8.3  

Implement REST API endpoints in ml-service/app/api/insights.py for listing, generating, viewing, dismissing, and providing feedback on ML insights.

**Details:**

Create ml-service/app/api/insights.py with FastAPI router:

1. Register router in ml-service/app/api/__init__.py:
   - Add `from .insights import router as insights_router`
   - Include router with prefix='/insights' and tags=['insights']

2. Create Pydantic schemas in ml-service/app/schemas/insights.py:
   - InsightResponse (mirrors MLInsight model fields)
   - InsightListResponse (list of insights with pagination)
   - GenerateInsightsRequest (user_id, optional force_regenerate)
   - GenerateInsightsResponse (generated insights list, count)
   - InsightFeedbackRequest (helpful: bool, comment: Optional[str])

3. Implement endpoints:
   - GET /api/insights/{user_id}
     * Query active (not dismissed, not expired) insights for user
     * Order by priority DESC, createdAt DESC
     * Limit to 10 most recent
     * Return InsightListResponse
   
   - POST /api/insights/generate
     * Accept GenerateInsightsRequest
     * Call InsightsEngine.generate_insights(user_id)
     * Store generated insights in database (handled in subtask 5)
     * Return GenerateInsightsResponse
   
   - PUT /api/insights/{insight_id}/viewed
     * Update viewed=True, viewedAt=now()
     * Return updated InsightResponse
   
   - PUT /api/insights/{insight_id}/dismissed
     * Update dismissed=True, dismissedAt=now()
     * Return success message
   
   - PUT /api/insights/{insight_id}/feedback
     * Accept InsightFeedbackRequest
     * Update helpful field
     * Store feedback in metadata JSON for analytics
     * Return success message

4. Follow patterns from existing api/correlations.py and api/predictions.py for error handling, dependency injection, and response formatting

### 8.5. Integrate Database Persistence with Prisma MLInsight Model

**Status:** pending  
**Dependencies:** 8.1, 8.4  

Implement database operations in InsightsEngine to persist and retrieve MLInsight records using SQLAlchemy async models that mirror the Prisma schema.

**Details:**

Create database integration for insights storage:

1. Create SQLAlchemy model in ml-service/app/models/ml_insight.py:
   - Mirror Prisma MLInsight model from server/prisma/schema.prisma
   - Fields: id, userId, insightType (enum), priority (enum), title, description, recommendation, correlation, confidence, dataPoints, metadata (JSON), viewed, viewedAt, dismissed, dismissedAt, helpful, createdAt, expiresAt
   - Create MLInsightType and InsightPriority enums matching Prisma enums

2. Add InsightRepository class for database operations:
   - `async def create_insight(insight_data: dict) -> MLInsight`
   - `async def get_user_insights(user_id: str, include_dismissed: bool = False, include_expired: bool = False) -> List[MLInsight]`
   - `async def get_insight_by_id(insight_id: str) -> Optional[MLInsight]`
   - `async def update_insight(insight_id: str, updates: dict) -> MLInsight`
   - `async def mark_viewed(insight_id: str) -> MLInsight`
   - `async def mark_dismissed(insight_id: str) -> MLInsight`
   - `async def set_feedback(insight_id: str, helpful: bool) -> MLInsight`

3. Update InsightsEngine.generate_insights():
   - After generating insights, persist each to database
   - Set expiresAt based on insight type (correlation: 7 days, anomaly: 1 day, prediction: 2 days)
   - Avoid duplicate insights by checking for similar recent insights
   - Return persisted MLInsight objects with database IDs

4. Implement `_prioritize_and_limit(insights, max_insights=5)`:
   - Sort by priority (CRITICAL > HIGH > MEDIUM > LOW)
   - Secondary sort by confidence DESC
   - Return top max_insights
   - Mark lower-priority duplicates for deferred storage

5. Add database migration or ensure schema sync with Prisma-generated tables
