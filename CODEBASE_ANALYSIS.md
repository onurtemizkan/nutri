# Nutri Codebase Architecture & ML Integration Planning

## Executive Summary
Nutri is a **full-stack nutrition tracking application** with a React Native (Expo) mobile frontend and Node.js/Express backend. The architecture is well-structured for ML integration, featuring comprehensive nutrition data collection, user profiles, and historical meal tracking capabilities.

---

## 1. CURRENT TECH STACK

### Frontend (Mobile)
- **Framework**: React Native with Expo 52.0.24
- **Routing**: Expo Router (file-based routing)
- **State Management**: React Context (AuthContext)
- **HTTP Client**: Axios
- **Secure Storage**: Expo Secure Store (JWT token storage)
- **UI Components**: React Native native components
- **Language**: TypeScript
- **Build**: Metro bundler

### Backend API
- **Runtime**: Node.js (v16+)
- **Framework**: Express.js (4.21.2)
- **Language**: TypeScript
- **Database**: PostgreSQL (v12+)
- **ORM**: Prisma (6.2.0)
- **Authentication**: JWT + bcryptjs password hashing
- **Validation**: Zod schema validation
- **HTTP**: RESTful API
- **Other**: CORS enabled, Error handling middleware

### Database
- **Type**: PostgreSQL
- **ORM**: Prisma
- **Migrations**: Prisma migrations

### Development Tools
- **Package Manager**: npm
- **Testing**: Jest with expo preset
- **Linting**: ESLint
- **TypeScript**: 5.3.3+

---

## 2. EXISTING DATA MODELS

### Core Models in Database (Prisma Schema)

#### User Model
```typescript
- id (String, Primary Key - CUID)
- email (String, Unique, Indexed)
- password (String, hashed with bcryptjs)
- name (String)
- createdAt (DateTime)
- updatedAt (DateTime)

// Password Reset
- resetToken (String, Optional)
- resetTokenExpiresAt (DateTime, Optional)

// Nutrition Goals
- goalCalories (Int, Default: 2000)
- goalProtein (Float, Default: 150)
- goalCarbs (Float, Default: 200)
- goalFat (Float, Default: 65)

// Physical Profile
- currentWeight (Float, Optional, in kg)
- goalWeight (Float, Optional, in kg)
- height (Float, Optional, in cm)
- activityLevel (String, Default: "moderate")
  - Values: sedentary, light, moderate, active, veryActive

// Relations
- meals (Meal[])
- waterIntakes (WaterIntake[])
- weightRecords (WeightRecord[])

// Indexes
- @@index([email])
- @@index([resetToken])
```

#### Meal Model
```typescript
- id (String, Primary Key - CUID)
- userId (String, Foreign Key)
- user (User relation, Cascade on delete)

// Meal Details
- name (String)
- mealType (String: breakfast, lunch, dinner, snack)
- servingSize (String, Optional)
- notes (String, Optional)
- imageUrl (String, Optional)

// Nutrition Data
- calories (Float)
- protein (Float, in grams)
- carbs (Float, in grams)
- fat (Float, in grams)
- fiber (Float, Optional, in grams)
- sugar (Float, Optional, in grams)

// Timestamps
- consumedAt (DateTime, indexed)
- createdAt (DateTime)
- updatedAt (DateTime)

// Indexes
- @@index([userId, consumedAt])
- @@index([userId, mealType])
```

#### WaterIntake Model
```typescript
- id (String, Primary Key - CUID)
- userId (String, Foreign Key)
- user (User relation, Cascade on delete)
- amount (Float, in ml)
- recordedAt (DateTime)
- createdAt (DateTime)
- @@index([userId, recordedAt])
```

#### WeightRecord Model
```typescript
- id (String, Primary Key - CUID)
- userId (String, Foreign Key)
- user (User relation, Cascade on delete)
- weight (Float, in kg)
- recordedAt (DateTime)
- createdAt (DateTime)
- @@index([userId, recordedAt])
```

### TypeScript Type Definitions (Frontend)

**User Interface**:
```typescript
interface User {
  id: string;
  email: string;
  name: string;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight?: number;
  goalWeight?: number;
  height?: number;
  activityLevel: ActivityLevel;
  createdAt: string;
}
```

**Meal Interface**:
```typescript
interface Meal {
  id: string;
  userId: string;
  name: string;
  mealType: MealType;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  servingSize?: string;
  notes?: string;
  imageUrl?: string;
  consumedAt: string;
  createdAt: string;
  updatedAt: string;
}
```

**Daily Summary Interface**:
```typescript
interface DailySummary {
  totalCalories: number;
  totalProtein: number;
  totalCarbs: number;
  totalFat: number;
  totalFiber: number;
  totalSugar: number;
  mealCount: number;
  goals: {
    goalCalories: number;
    goalProtein: number;
    goalCarbs: number;
    goalFat: number;
  };
  meals: Meal[];
}
```

**Weekly Summary Interface**:
```typescript
interface WeeklySummary {
  date: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  mealCount: number;
}
```

---

## 3. API STRUCTURE & ENDPOINTS

### Authentication Endpoints
**Base URL**: `/api/auth`

| Method | Endpoint | Protected | Purpose |
|--------|----------|-----------|---------|
| POST | `/register` | No | Register new user with email, password, name |
| POST | `/login` | No | Login user, returns JWT token |
| GET | `/profile` | Yes | Get user profile & goals |
| PUT | `/profile` | Yes | Update user profile & nutrition goals |
| POST | `/forgot-password` | No | Request password reset token |
| POST | `/reset-password` | No | Reset password with token |
| GET | `/verify-token` | No | Verify password reset token validity |

### Meal Endpoints
**Base URL**: `/api/meals`
**All endpoints protected with JWT**

| Method | Endpoint | Purpose | Query/Body |
|--------|----------|---------|-----------|
| POST | `/` | Create new meal | Body: CreateMealInput |
| GET | `/` | Get meals for day | Query: `date` (optional, ISO string) |
| GET | `/:id` | Get meal by ID | - |
| PUT | `/:id` | Update meal | Body: UpdateMealInput (partial) |
| DELETE | `/:id` | Delete meal | - |
| GET | `/summary/daily` | Get daily nutrition summary | Query: `date` (optional) |
| GET | `/summary/weekly` | Get weekly summary (last 7 days) | - |

### Request/Response Examples

**Create Meal Request**:
```json
{
  "name": "Grilled Chicken Salad",
  "mealType": "lunch",
  "calories": 450,
  "protein": 35,
  "carbs": 30,
  "fat": 18,
  "fiber": 8,
  "sugar": 5,
  "servingSize": "1 bowl",
  "notes": "No dressing",
  "consumedAt": "2025-01-17T12:30:00Z"
}
```

**Daily Summary Response**:
```json
{
  "totalCalories": 1850,
  "totalProtein": 95,
  "totalCarbs": 210,
  "totalFat": 65,
  "totalFiber": 32,
  "totalSugar": 18,
  "mealCount": 3,
  "goals": {
    "goalCalories": 2000,
    "goalProtein": 150,
    "goalCarbs": 200,
    "goalFat": 65
  },
  "meals": [...]
}
```

---

## 4. DATABASE SCHEMA DETAILS

### Schema Indexes (Optimized for Common Queries)

```
User:
  - PRIMARY: id
  - UNIQUE: email
  - INDEX: email (for login lookups)
  - INDEX: resetToken (for password reset flow)

Meal:
  - PRIMARY: id
  - COMPOSITE INDEX: (userId, consumedAt) 
    → Optimizes: Get meals by user & date range
  - COMPOSITE INDEX: (userId, mealType)
    → Optimizes: Get meals by user & type (breakfast, lunch, etc.)

WaterIntake:
  - PRIMARY: id
  - COMPOSITE INDEX: (userId, recordedAt)
    → Optimizes: Get water intake by user & date range

WeightRecord:
  - PRIMARY: id
  - COMPOSITE INDEX: (userId, recordedAt)
    → Optimizes: Get weight history by user & date range
```

### Relationships & Cascading

All child models (Meal, WaterIntake, WeightRecord) have:
- Foreign key to User
- **Cascade on Delete**: Deleting user deletes all their meals, water intakes, weight records

---

## 5. SERVICE LAYER ARCHITECTURE

### MealService
Located: `/server/src/services/mealService.ts`

**Methods**:
- `createMeal(userId, data)` - Create meal for user
- `getMeals(userId, date?)` - Get meals for specific day (defaults to today)
- `getMealById(userId, mealId)` - Get single meal (with ownership check)
- `updateMeal(userId, mealId, data)` - Update meal (with ownership check)
- `deleteMeal(userId, mealId)` - Delete meal (with ownership check)
- `getDailySummary(userId, date?)` - Aggregate daily totals + goals
- `getWeeklySummary(userId)` - Group meals by day for 7 days

**Daily Summary Logic**:
- Fetches all meals for a specific day (midnight to 11:59:59 PM)
- Sums nutrition values across all meals
- Retrieves user's current goals
- Returns meals array + aggregated totals + goals

**Weekly Summary Logic**:
- Fetches all meals from 7 days ago
- Groups by date (YYYY-MM-DD)
- Calculates daily totals for each day
- Returns array of daily summaries

### AuthService
Located: `/server/src/services/authService.ts`

**Methods**:
- `register(data)` - Create user account (email must be unique)
- `login(data)` - Validate credentials & return JWT token
- `getUserProfile(userId)` - Get user details
- `requestPasswordReset(email)` - Generate reset token
- `resetPassword(token, newPassword)` - Update password
- `verifyResetToken(token)` - Check token validity

---

## 6. CONTROLLER LAYER

### MealController
Located: `/server/src/controllers/mealController.ts`

**Responsibilities**:
- Input validation using Zod schemas
- Call appropriate service methods
- Handle errors & return appropriate HTTP status codes
- Error handling for ZodError (validation), business logic errors, and server errors

**Input Schemas**:
- `createMealSchema` - Validates all meal fields with constraints
- `updateMealSchema` - Same fields but all optional

### AuthController
Similar pattern for authentication endpoints

---

## 7. MIDDLEWARE & UTILITIES

### Authentication Middleware
Located: `/server/src/middleware/auth.ts`

- Verifies JWT token from `Authorization: Bearer {token}` header
- Decodes and validates token
- Attaches userId to request object
- Returns 401 Unauthorized if token missing/invalid

### Error Handling Middleware
Located: `/server/src/middleware/errorHandler.ts`

- Global error handler (must be last middleware)
- Formats error responses consistently
- Handles 404s and unexpected errors

### Auth Helpers
Located: `/server/src/utils/authHelpers.ts`

- `requireAuth(req, res)` - Check auth and return userId or void

---

## 8. FRONTEND ARCHITECTURE

### Screens

**Auth Flows**:
- `/app/auth/welcome.tsx` - Welcome screen with signup/signin buttons
- `/app/auth/signin.tsx` - Email/password login
- `/app/auth/signup.tsx` - Email/password registration
- `/app/auth/forgot-password.tsx` - Password reset request
- `/app/auth/reset-password.tsx` - Reset password form

**Main App**:
- `/app/(tabs)/index.tsx` - Dashboard (home screen)
- `/app/(tabs)/profile.tsx` - User profile & settings
- `/app/add-meal.tsx` - Modal to add/edit meals
- `/app/_layout.tsx` - Root layout with auth routing

### State Management
- **AuthContext** (`/lib/context/AuthContext.tsx`)
  - Manages user login state
  - Stores JWT token in SecureStore
  - Provides user object to entire app

### API Client
- **Axios instance** (`/lib/api/client.ts`)
  - Centralized configuration
  - Automatic JWT injection via interceptors
  - Response error handling (401 logout)

### API Modules
- **mealsApi** (`/lib/api/meals.ts`) - All meal-related requests
- **authApi** (`/lib/api/auth.ts`) - All auth-related requests

### Dashboard Features
- Daily calorie progress ring
- Macro breakdown (protein, carbs, fat)
- Meals grouped by type
- Pull-to-refresh
- FAB (Floating Action Button) to add meals

---

## 9. DATA FLOW OVERVIEW

### Adding a Meal
```
Mobile UI (add-meal.tsx)
  ↓ (user fills form)
  ↓ mealsApi.createMeal(data)
    ↓
    Axios POST /api/meals
      ↓
      MealController.createMeal()
        ↓ (validate with Zod)
        ↓ MealService.createMeal()
          ↓
          Prisma: meal.create()
            ↓
            PostgreSQL INSERT
              ↓ (returns Meal object)
              ↓
  ↓ Response returned to frontend
  ↓ UI refreshes dashboard
```

### Getting Daily Summary
```
Mobile UI (home screen) onMount
  ↓ mealsApi.getDailySummary()
    ↓
    Axios GET /api/meals/summary/daily
      ↓
      MealController.getDailySummary()
        ↓ MealService.getDailySummary()
          ↓ getMeals(userId, date) - query all meals for day
          ↓ reduce() - sum all nutrition values
          ↓ prisma.user.findUnique() - get user goals
            ↓
  ↓ DailySummary object returned
  ↓ setState(summary)
  ↓ UI renders with daily progress
```

---

## 10. ML INTEGRATION OPPORTUNITIES

### 1. Meal Recognition (Image Classification)
**Data Available**: imageUrl field in Meal model (ready but not used yet)

**ML Features**:
- Image-to-macro prediction
- Portion size estimation from photo
- Food item detection

**Integration Points**:
- Before createMeal in MealService
- Optional field, wouldn't break existing flow

### 2. Personalized Calorie Goal Prediction
**Data Available**:
- currentWeight, goalWeight, height
- activityLevel enum
- goalCalories (current goal)
- Historical meal data

**ML Features**:
- BMR (Basal Metabolic Rate) calculation
- TDEE (Total Daily Energy Expenditure) estimation
- Optimal calorie goal for user's goal weight
- Confidence scores

**Integration Points**:
- New endpoint: POST /api/ml/predict-calorie-goal
- Called in profile update flow
- Updates goalCalories

### 3. Meal Recommendations
**Data Available**:
- Historical meals (mealType, nutrition values)
- Time-based patterns (breakfast at 8am, lunch at 12pm)
- Remaining macros for the day
- User goals

**ML Features**:
- Recommend meals based on historical patterns
- Suggest meals to hit macro targets
- Predict ideal meal timing

**Integration Points**:
- New endpoint: GET /api/ml/meal-recommendations
- Called when user opens add-meal screen
- Shows suggested meals

### 4. Nutrition Pattern Analysis
**Data Available**:
- Weekly summary data (aggregated)
- Weight records over time
- Daily meal counts
- Consistency patterns

**ML Features**:
- Identify eating patterns (consistent meals, timing)
- Detect nutritional gaps
- Predict weight changes
- Adherence scoring

**Integration Points**:
- New endpoint: GET /api/ml/nutrition-insights
- Called on dashboard or new insights screen

### 5. Anomaly Detection
**Data Available**:
- Historical meal patterns
- Daily nutrition values
- User's typical intake

**ML Features**:
- Detect unusual eating days
- Alert for significant macro imbalances
- Identify possible data entry errors

**Integration Points**:
- Flag meals during creation
- Highlight in weekly summary

---

## 11. READY-FOR-ML FEATURES

### Data Collection Points
1. ✅ User physical profile: height, weight, goals
2. ✅ Activity level selection (5 levels)
3. ✅ Meal type categorization
4. ✅ Detailed nutrition tracking (7 macro fields)
5. ✅ Meal photos (imageUrl field exists, not used)
6. ✅ Weight history (WeightRecord model exists)
7. ✅ Water intake (WaterIntake model exists)
8. ✅ Timestamps (exact meal timing)

### Database Strengths for ML
1. ✅ Composite indexes on (userId, timestamp) → fast time-range queries
2. ✅ Cascade deletes → clean data deletion
3. ✅ CUID primary keys → distributed system ready
4. ✅ User relations → easy to query per-user patterns

### API Design Strengths for ML
1. ✅ Separated services → easy to add ML service layer
2. ✅ Zod validation → can enhance with ML predictions
3. ✅ JWT auth → secure user-specific predictions
4. ✅ Existing daily/weekly summaries → foundation for patterns

---

## 12. ML ARCHITECTURE RECOMMENDATIONS

### Backend ML Service Architecture
```
/server/src/services/mlService.ts
  ├── Nutrition Analysis
  │   ├── calculateBMR(height, weight, age)
  │   ├── calculateTDEE(bmr, activityLevel)
  │   └── predictOptimalCalories(currentWeight, goalWeight)
  │
  ├── Pattern Recognition
  │   ├── getMealPatterns(userId, days)
  │   ├── getWeeklyConsistency(userId)
  │   └── predictNextMealTime(userId)
  │
  ├── Meal Recommendations
  │   ├── recommendMeals(userId, remainingMacros)
  │   ├── findSimilarMeals(mealName)
  │   └── getMacroBalance(meals)
  │
  ├── Image Analysis
  │   ├── predictNutritionFromImage(imageUrl)
  │   ├── estimatePortion(imageUrl, foodType)
  │   └── validateNutritionData(predicted, userInput)
  │
  └── Insights
      ├── generateWeeklyInsights(userId)
      ├── detectAnomalies(mealData)
      └── getProgressMetrics(userId, timeframe)

/server/src/routes/mlRoutes.ts (new)
  ├── POST /api/ml/predict-calories
  ├── GET /api/ml/meal-recommendations
  ├── POST /api/ml/analyze-nutrition
  ├── GET /api/ml/insights
  ├── POST /api/ml/predict-from-image
  └── GET /api/ml/anomalies

/server/src/types/ml.ts (new)
  ├── MLPrediction interface
  ├── MealRecommendation interface
  ├── NutritionInsight interface
  └── AnomalyAlert interface
```

### Database Enhancements
```prisma
model MealPrediction {
  id String @id @default(cuid())
  userId String
  user User @relation(fields: [userId], references: [id])
  
  mealId String
  meal Meal @relation(fields: [mealId], references: [id])
  
  predictedCalories Float
  predictedProtein Float
  predictedCarbs Float
  predictedFat Float
  
  confidence Float // 0.0 - 1.0
  source String // "image", "pattern", "suggestion"
  
  createdAt DateTime @default(now())
  @@index([userId, createdAt])
}

model UserInsight {
  id String @id @default(cuid())
  userId String
  user User @relation(fields: [userId], references: [id])
  
  type String // "pattern", "anomaly", "recommendation"
  title String
  description String
  data Json // flexible for different insight types
  
  actionable Boolean @default(true)
  acknowledged Boolean @default(false)
  
  createdAt DateTime @default(now())
  @@index([userId, createdAt])
}
```

---

## 13. KEY FILES & LOCATIONS

### Backend Structure
```
/server/
├── prisma/
│   └── schema.prisma ..................... Data models
├── src/
│   ├── config/
│   │   ├── env.ts ....................... Environment config
│   │   └── database.ts .................. Prisma client
│   ├── controllers/
│   │   ├── authController.ts ............ Auth request handlers
│   │   └── mealController.ts ............ Meal request handlers
│   ├── services/
│   │   ├── authService.ts .............. Auth business logic
│   │   └── mealService.ts .............. Meal business logic
│   │   └── mlService.ts (FUTURE) ....... ML business logic
│   ├── routes/
│   │   ├── authRoutes.ts ............... Auth endpoints
│   │   ├── mealRoutes.ts ............... Meal endpoints
│   │   └── mlRoutes.ts (FUTURE) ........ ML endpoints
│   ├── middleware/
│   │   ├── auth.ts ..................... JWT verification
│   │   └── errorHandler.ts ............. Error handling
│   ├── types/
│   │   ├── index.ts .................... Main type defs
│   │   └── ml.ts (FUTURE) .............. ML type defs
│   └── index.ts ........................ Main app entry
└── package.json ........................ Dependencies

```

### Frontend Structure
```
/
├── app/
│   ├── (tabs)/
│   │   ├── index.tsx ................... Dashboard
│   │   ├── profile.tsx ................. Profile screen
│   │   └── _layout.tsx ................. Tab navigation
│   ├── auth/
│   │   ├── welcome.tsx
│   │   ├── signin.tsx
│   │   ├── signup.tsx
│   │   ├── forgot-password.tsx
│   │   └── reset-password.tsx
│   ├── add-meal.tsx .................... Meal form modal
│   ├── _layout.tsx ..................... Root layout
│   └── +not-found.tsx
├── lib/
│   ├── api/
│   │   ├── client.ts ................... Axios instance
│   │   ├── auth.ts ..................... Auth API calls
│   │   └── meals.ts .................... Meal API calls
│   ├── context/
│   │   └── AuthContext.tsx ............. Auth state
│   └── types/
│       └── index.ts .................... Shared types
└── package.json

```

---

## 14. CURRENT STATE SUMMARY

| Aspect | Status | Details |
|--------|--------|---------|
| **Backend** | ✅ Production Ready | Express + Prisma + PostgreSQL |
| **Frontend** | ✅ Production Ready | React Native + Expo + TypeScript |
| **Authentication** | ✅ Complete | JWT + bcryptjs + password reset |
| **Meal Tracking** | ✅ Complete | CRUD + daily/weekly summaries |
| **Data Models** | ✅ Optimized | Indexed for queries, cascading deletes |
| **API Documentation** | ✅ Complete | RESTful, Type-safe |
| **Error Handling** | ✅ Implemented | Global middleware, Zod validation |
| **ML Integration** | 🟡 Ready | Data structures in place, no ML code yet |
| **Analytics** | 🟡 Ready | Weekly summary logic exists |
| **Image Support** | 🟡 Schema Ready | imageUrl field exists, not used |

---

## 15. RECOMMENDED ML INTEGRATION ROADMAP

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create mlService.ts with basic functions
- [ ] Add MLRoutes
- [ ] Implement simple TDEE calculation
- [ ] Add BMR calculation

### Phase 2: Recommendations (Weeks 3-4)
- [ ] Implement meal pattern recognition
- [ ] Create recommendation logic
- [ ] Add weekly consistency scoring
- [ ] Implement basic meal search

### Phase 3: Advanced (Weeks 5-6)
- [ ] Image classification integration (TensorFlow.js or external API)
- [ ] Anomaly detection
- [ ] Generate user insights
- [ ] Add frontend insight screens

### Phase 4: Polish (Weeks 7-8)
- [ ] Performance optimization
- [ ] Add ML confidence scores to UI
- [ ] User feedback loops
- [ ] A/B testing infrastructure

---

## 16. QUICK START FOR ML DEVELOPMENT

### To add a new ML feature:

1. **Add types** → `/server/src/types/ml.ts`
2. **Add service method** → `/server/src/services/mlService.ts`
3. **Add controller** → Update `/server/src/controllers/mealController.ts` or create new
4. **Add route** → `/server/src/routes/mlRoutes.ts`
5. **Register route** → `/server/src/index.ts` app.use()
6. **Add frontend call** → `/lib/api/meals.ts` or create `/lib/api/ml.ts`
7. **Update UI** → Add component to display results

### Database query patterns for ML:

```typescript
// Get user's meal history for pattern analysis
const meals = await prisma.meal.findMany({
  where: {
    userId,
    consumedAt: { gte: startDate, lte: endDate }
  },
  orderBy: { consumedAt: 'asc' }
});

// Get weekly aggregates
const weeklySummary = await mealService.getWeeklySummary(userId);

// Get user profile for predictions
const user = await prisma.user.findUnique({
  where: { id: userId }
});
```

---

## 17. POTENTIAL ML CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| **User Data Privacy** | All predictions user-specific, no data sharing |
| **Cold Start** (new users) | Use demographic defaults until pattern emerges |
| **Data Quality** | Implement validation rules in mealService |
| **Recommendation Relevance** | Use collaborative filtering + content-based |
| **Real-time Image Processing** | Batch processing or external API (AWS Rekognition, etc.) |
| **Model Deployment** | Start with rule-based, evolve to ML models |
| **Performance** | Cache daily summaries, index queries optimized |

---

## CONCLUSION

The Nutri app has a **well-structured, ML-ready architecture**:

✅ Comprehensive nutrition data collection
✅ Optimized database with proper indexes
✅ Clear separation of concerns (services, controllers, routes)
✅ Type-safe throughout (TypeScript, Zod validation)
✅ Extensible API design
✅ User-specific data patterns ready for analysis

The foundation is solid for implementing ML features without major refactoring. Start with simple features (TDEE calculation, meal patterns) and gradually add complexity.
