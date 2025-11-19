# Nutri ML Integration - Quick Reference Guide

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                     MOBILE FRONTEND                         │
│        React Native (Expo) + TypeScript + Axios             │
├─────────────────────────────────────────────────────────────┤
│  Dashboard │ Profile │ Add Meal │ Auth Screens              │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         │ JWT Auth
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    EXPRESS.JS API                           │
│          Node.js + TypeScript + Zod Validation              │
├──────────────────┬──────────────────┬──────────────────────┤
│  Auth Routes    │  Meal Routes     │  ML Routes (NEW)     │
│  ├─ Register   │  ├─ Create       │  ├─ Predictions      │
│  ├─ Login      │  ├─ Read         │  ├─ Recommendations  │
│  ├─ Profile    │  ├─ Update       │  ├─ Insights         │
│  └─ Password   │  ├─ Delete       │  └─ Anomalies        │
│                │  └─ Summary      │                       │
├──────────────────┴──────────────────┴──────────────────────┤
│  Controllers → Services → Database                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  POSTGRESQL DATABASE                         │
│            Prisma ORM + Optimized Indexes                   │
├─────────────────────────────────────────────────────────────┤
│  ├─ User (4 nutrition goals, physical profile)             │
│  ├─ Meal (7 nutrition fields, timestamps)                  │
│  ├─ WaterIntake (future use)                               │
│  └─ WeightRecord (future use)                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Available for ML

| Category | Fields | Ready? |
|----------|--------|--------|
| **User Profile** | height, weight, goal_weight, goals | ✅ |
| **Activity Level** | 5 enum values | ✅ |
| **Meal History** | name, type, all macros, timestamp | ✅ |
| **Time Data** | consumedAt, createdAt (precise) | ✅ |
| **Images** | imageUrl field exists | ✅ |
| **Weight Tracking** | WeightRecord model | ✅ |
| **Daily Aggregates** | Pre-calculated summaries | ✅ |

## ML Features to Build (In Order of Complexity)

### Easy (1-3 days each)
1. **TDEE Calculator** - Math-based, no ML
   - Input: height, weight, activity level
   - Output: recommended daily calories
   - Location: `/server/src/services/mlService.ts`

2. **Macro Distributions** - Rule-based recommendations
   - Input: daily calorie goal
   - Output: protein/carbs/fat grams
   - Location: Same service

### Medium (1 week each)
3. **Meal Pattern Detection** - Time-series analysis
   - Input: User's last 30 meals
   - Output: Eating patterns, favorite meals, timing
   - Location: New method in mlService

4. **Simple Recommendations** - Content filtering
   - Input: Remaining macros today, meal history
   - Output: Suggested meals from history
   - Location: mlService + new API endpoint

### Hard (2+ weeks each)
5. **Image Classification** - Deep learning
   - Input: Food photo
   - Output: Food type + estimated macros
   - Location: External API integration or local model

6. **Weight Prediction** - Regression model
   - Input: Historical meals, weight, consistency
   - Output: Predicted weight in X weeks
   - Location: mlService with ML library

7. **Anomaly Detection** - Statistical analysis
   - Input: User's typical patterns + today's meals
   - Output: Alerts for unusual intake
   - Location: mlService + alerts

## Integration Checklist

### Phase 1: Setup (Day 1)
- [ ] Create `/server/src/services/mlService.ts`
- [ ] Create `/server/src/types/ml.ts`
- [ ] Create `/server/src/routes/mlRoutes.ts`
- [ ] Register routes in `/server/src/index.ts`

### Phase 2: First Feature - TDEE (Days 2-3)
- [ ] Implement `calculateBMR()` in mlService
- [ ] Implement `calculateTDEE()` in mlService
- [ ] Create controller method for predictions
- [ ] Add API endpoint POST `/api/ml/predict-calories`
- [ ] Test with curl/Postman

### Phase 3: Frontend Integration (Days 4-5)
- [ ] Create `/lib/api/ml.ts` for ML API calls
- [ ] Add ML results to profile screen
- [ ] Show predictions in UI

### Phase 4: Next Features (Repeat for each)
- [ ] Design feature (decide inputs/outputs)
- [ ] Implement service method
- [ ] Add controller
- [ ] Add route
- [ ] Frontend integration
- [ ] Test end-to-end

## Key Database Queries for ML

```typescript
// Get all meals for a user in date range
const meals = await prisma.meal.findMany({
  where: {
    userId: "user-id",
    consumedAt: {
      gte: new Date("2025-01-01"),
      lte: new Date("2025-01-31")
    }
  },
  orderBy: { consumedAt: 'asc' }
});

// Get user profile
const user = await prisma.user.findUnique({
  where: { id: "user-id" }
});

// Get weekly summary (already calculated)
const weeklySummary = await mealService.getWeeklySummary(userId);

// Count meals by type
const breakfasts = meals.filter(m => m.mealType === 'breakfast');
```

## Common ML Formulas to Use

### Basal Metabolic Rate (BMR) - Mifflin-St Jeor
```javascript
function calculateBMR(weight_kg, height_cm, age_years, isMale) {
  if (isMale) {
    return 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5;
  } else {
    return 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161;
  }
}
```

### Total Daily Energy Expenditure (TDEE)
```javascript
const activityMultipliers = {
  'sedentary': 1.2,      // little/no exercise
  'light': 1.375,        // 1-3 days/week
  'moderate': 1.55,      // 3-5 days/week
  'active': 1.725,       // 6-7 days/week
  'veryActive': 1.9      // intense exercise 2x/day
};

function calculateTDEE(bmr, activityLevel) {
  return bmr * activityMultipliers[activityLevel];
}
```

### Macro Distribution (Standard 40-30-30)
```javascript
function calculateMacros(dailyCalories) {
  return {
    protein_grams: (dailyCalories * 0.30) / 4,  // 4 cal/gram
    carbs_grams: (dailyCalories * 0.40) / 4,    // 4 cal/gram
    fat_grams: (dailyCalories * 0.30) / 9       // 9 cal/gram
  };
}
```

## Testing ML Features Locally

### Test TDEE Calculator
```bash
curl -X POST http://localhost:3000/api/ml/predict-calories \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "height": 180,
    "weight": 80,
    "age": 30,
    "gender": "male",
    "activityLevel": "moderate"
  }'
```

### Test Recommendations
```bash
curl -X GET "http://localhost:3000/api/ml/meal-recommendations?userId=USER_ID" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## File Structure After ML Integration

```
/server/src/
├── services/
│   ├── authService.ts
│   ├── mealService.ts
│   └── mlService.ts ..................... NEW
├── controllers/
│   ├── authController.ts
│   ├── mealController.ts
│   └── mlController.ts .................. NEW
├── routes/
│   ├── authRoutes.ts
│   ├── mealRoutes.ts
│   └── mlRoutes.ts ...................... NEW
└── types/
    ├── index.ts
    └── ml.ts ............................ NEW
```

## Performance Tips

1. **Cache Results**: TDEE/BMR don't change often, cache for 24 hours
2. **Batch Queries**: Get 90 days of meals once, process in memory
3. **Index Usage**: Composite (userId, consumedAt) index is fast
4. **Lazy Load**: Don't calculate insights unless user requests
5. **Pagination**: For historical data, paginate in chunks of 30 days

## Error Handling in ML

```typescript
try {
  const meals = await mealService.getMeals(userId);
  
  if (!meals || meals.length < 3) {
    return { 
      error: 'Not enough data',
      message: 'Need at least 3 meals to make predictions'
    };
  }
  
  const prediction = calculatePrediction(meals);
  return { success: true, data: prediction };
} catch (error) {
  logger.error('ML prediction failed:', error);
  return { error: 'Prediction failed' };
}
```

## Next Steps

1. Save this guide locally
2. Create `/server/src/services/mlService.ts` with base structure
3. Implement TDEE calculator (easiest feature)
4. Test API endpoint
5. Integrate into frontend
6. Deploy and gather user feedback
7. Iterate on next feature

---

**Remember**: Start simple, iterate fast, gather user feedback!
