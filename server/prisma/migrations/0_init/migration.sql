-- CreateEnum
CREATE TYPE "SupplementFrequency" AS ENUM ('DAILY', 'TWICE_DAILY', 'THREE_TIMES_DAILY', 'WEEKLY', 'EVERY_OTHER_DAY', 'AS_NEEDED');

-- CreateEnum
CREATE TYPE "SupplementTimeOfDay" AS ENUM ('MORNING', 'AFTERNOON', 'EVENING', 'BEFORE_BED', 'WITH_BREAKFAST', 'WITH_LUNCH', 'WITH_DINNER', 'EMPTY_STOMACH');

-- CreateEnum
CREATE TYPE "SubscriptionTier" AS ENUM ('FREE', 'PRO_TRIAL', 'PRO');

-- CreateEnum
CREATE TYPE "BillingCycle" AS ENUM ('MONTHLY', 'ANNUAL');

-- CreateEnum
CREATE TYPE "HealthMetricType" AS ENUM ('RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_SDNN', 'HEART_RATE_VARIABILITY_RMSSD', 'BLOOD_PRESSURE_SYSTOLIC', 'BLOOD_PRESSURE_DIASTOLIC', 'RESPIRATORY_RATE', 'OXYGEN_SATURATION', 'VO2_MAX', 'SLEEP_DURATION', 'DEEP_SLEEP_DURATION', 'REM_SLEEP_DURATION', 'SLEEP_EFFICIENCY', 'SLEEP_SCORE', 'STEPS', 'ACTIVE_CALORIES', 'TOTAL_CALORIES', 'EXERCISE_MINUTES', 'STANDING_HOURS', 'RECOVERY_SCORE', 'STRAIN_SCORE', 'READINESS_SCORE', 'BODY_FAT_PERCENTAGE', 'MUSCLE_MASS', 'BONE_MASS', 'WATER_PERCENTAGE', 'SKIN_TEMPERATURE', 'BLOOD_GLUCOSE', 'STRESS_LEVEL');

-- CreateEnum
CREATE TYPE "ActivityType" AS ENUM ('RUNNING', 'CYCLING', 'SWIMMING', 'WALKING', 'HIKING', 'ROWING', 'ELLIPTICAL', 'WEIGHT_TRAINING', 'BODYWEIGHT', 'CROSSFIT', 'POWERLIFTING', 'BASKETBALL', 'SOCCER', 'TENNIS', 'GOLF', 'YOGA', 'PILATES', 'STRETCHING', 'MARTIAL_ARTS', 'DANCE', 'OTHER');

-- CreateEnum
CREATE TYPE "ActivityIntensity" AS ENUM ('LOW', 'MODERATE', 'HIGH', 'MAXIMUM');

-- CreateEnum
CREATE TYPE "MLFeatureCategory" AS ENUM ('NUTRITION_DAILY', 'NUTRITION_WEEKLY', 'NUTRITION_TEMPORAL', 'ACTIVITY_DAILY', 'ACTIVITY_WEEKLY', 'HEALTH_DAILY', 'HEALTH_WEEKLY', 'COMBINED_FEATURES');

-- CreateEnum
CREATE TYPE "MLInsightType" AS ENUM ('CORRELATION', 'PREDICTION', 'ANOMALY', 'RECOMMENDATION', 'GOAL_PROGRESS', 'PATTERN_DETECTED');

-- CreateEnum
CREATE TYPE "InsightPriority" AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');

-- CreateEnum
CREATE TYPE "AdminRole" AS ENUM ('SUPER_ADMIN', 'SUPPORT', 'ANALYST', 'VIEWER');

-- CreateEnum
CREATE TYPE "FeatureFlagType" AS ENUM ('BOOLEAN', 'STRING', 'NUMBER', 'JSON');

-- CreateEnum
CREATE TYPE "WebhookEventStatus" AS ENUM ('PENDING', 'SUCCESS', 'FAILED');

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT,
    "name" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "appleId" TEXT,
    "resetToken" TEXT,
    "resetTokenExpiresAt" TIMESTAMP(3),
    "profilePicture" TEXT,
    "goalCalories" INTEGER NOT NULL DEFAULT 2000,
    "goalProtein" DOUBLE PRECISION NOT NULL DEFAULT 150,
    "goalCarbs" DOUBLE PRECISION NOT NULL DEFAULT 200,
    "goalFat" DOUBLE PRECISION NOT NULL DEFAULT 65,
    "currentWeight" DOUBLE PRECISION,
    "goalWeight" DOUBLE PRECISION,
    "height" DOUBLE PRECISION,
    "activityLevel" TEXT NOT NULL DEFAULT 'moderate',
    "subscriptionTier" "SubscriptionTier" NOT NULL DEFAULT 'FREE',
    "subscriptionBillingCycle" "BillingCycle",
    "subscriptionStartDate" TIMESTAMP(3),
    "subscriptionEndDate" TIMESTAMP(3),
    "subscriptionPrice" DOUBLE PRECISION,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Meal" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "mealType" TEXT NOT NULL,
    "calories" DOUBLE PRECISION NOT NULL,
    "protein" DOUBLE PRECISION NOT NULL,
    "carbs" DOUBLE PRECISION NOT NULL,
    "fat" DOUBLE PRECISION NOT NULL,
    "fiber" DOUBLE PRECISION,
    "sugar" DOUBLE PRECISION,
    "saturatedFat" DOUBLE PRECISION,
    "transFat" DOUBLE PRECISION,
    "cholesterol" DOUBLE PRECISION,
    "sodium" DOUBLE PRECISION,
    "potassium" DOUBLE PRECISION,
    "calcium" DOUBLE PRECISION,
    "iron" DOUBLE PRECISION,
    "magnesium" DOUBLE PRECISION,
    "zinc" DOUBLE PRECISION,
    "phosphorus" DOUBLE PRECISION,
    "vitaminA" DOUBLE PRECISION,
    "vitaminC" DOUBLE PRECISION,
    "vitaminD" DOUBLE PRECISION,
    "vitaminE" DOUBLE PRECISION,
    "vitaminK" DOUBLE PRECISION,
    "vitaminB6" DOUBLE PRECISION,
    "vitaminB12" DOUBLE PRECISION,
    "folate" DOUBLE PRECISION,
    "thiamin" DOUBLE PRECISION,
    "riboflavin" DOUBLE PRECISION,
    "niacin" DOUBLE PRECISION,
    "servingSize" TEXT,
    "notes" TEXT,
    "imageUrl" TEXT,
    "consumedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Meal_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WaterIntake" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "amount" DOUBLE PRECISION NOT NULL,
    "recordedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "WaterIntake_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "WeightRecord" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "weight" DOUBLE PRECISION NOT NULL,
    "recordedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "WeightRecord_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "HealthMetric" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "recordedAt" TIMESTAMP(3) NOT NULL,
    "metricType" "HealthMetricType" NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "sourceId" TEXT,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "HealthMetric_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Activity" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "startedAt" TIMESTAMP(3) NOT NULL,
    "endedAt" TIMESTAMP(3) NOT NULL,
    "duration" INTEGER NOT NULL,
    "activityType" "ActivityType" NOT NULL,
    "intensity" "ActivityIntensity" NOT NULL,
    "caloriesBurned" DOUBLE PRECISION,
    "averageHeartRate" DOUBLE PRECISION,
    "maxHeartRate" DOUBLE PRECISION,
    "distance" DOUBLE PRECISION,
    "steps" INTEGER,
    "source" TEXT NOT NULL,
    "sourceId" TEXT,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Activity_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MLFeature" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "category" "MLFeatureCategory" NOT NULL,
    "features" JSONB NOT NULL,
    "version" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MLFeature_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MLPrediction" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "targetMetric" "HealthMetricType" NOT NULL,
    "targetDate" TIMESTAMP(3) NOT NULL,
    "predictedValue" DOUBLE PRECISION NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "predictionRange" JSONB NOT NULL,
    "modelId" TEXT NOT NULL,
    "modelVersion" TEXT NOT NULL,
    "featureImportance" JSONB NOT NULL,
    "actualValue" DOUBLE PRECISION,
    "predictionError" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "MLPrediction_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MLInsight" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "insightType" "MLInsightType" NOT NULL,
    "priority" "InsightPriority" NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "recommendation" TEXT NOT NULL,
    "correlation" DOUBLE PRECISION,
    "confidence" DOUBLE PRECISION NOT NULL,
    "dataPoints" INTEGER NOT NULL,
    "metadata" JSONB,
    "viewed" BOOLEAN NOT NULL DEFAULT false,
    "viewedAt" TIMESTAMP(3),
    "dismissed" BOOLEAN NOT NULL DEFAULT false,
    "dismissedAt" TIMESTAMP(3),
    "helpful" BOOLEAN,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3),

    CONSTRAINT "MLInsight_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserMLProfile" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "totalDataPoints" INTEGER NOT NULL DEFAULT 0,
    "dataQualityScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "hasMinimumNutritionData" BOOLEAN NOT NULL DEFAULT false,
    "hasMinimumHealthData" BOOLEAN NOT NULL DEFAULT false,
    "hasMinimumActivityData" BOOLEAN NOT NULL DEFAULT false,
    "modelsAvailable" JSONB NOT NULL,
    "lastTrainingDate" TIMESTAMP(3),
    "enablePredictions" BOOLEAN NOT NULL DEFAULT true,
    "enableInsights" BOOLEAN NOT NULL DEFAULT true,
    "insightFrequency" TEXT NOT NULL DEFAULT 'daily',
    "shareDataForResearch" BOOLEAN NOT NULL DEFAULT false,
    "dataRetentionDays" INTEGER NOT NULL DEFAULT 365,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserMLProfile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FoodFeedback" (
    "id" TEXT NOT NULL,
    "userId" TEXT,
    "imageHash" TEXT NOT NULL,
    "classificationId" TEXT,
    "originalPrediction" TEXT NOT NULL,
    "originalConfidence" DOUBLE PRECISION NOT NULL,
    "originalCategory" TEXT,
    "selectedFdcId" INTEGER NOT NULL,
    "selectedFoodName" TEXT NOT NULL,
    "wasCorrect" BOOLEAN NOT NULL,
    "classificationHints" JSONB,
    "userDescription" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FoodFeedback_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FoodFeedbackAggregation" (
    "id" TEXT NOT NULL,
    "originalPrediction" TEXT NOT NULL,
    "correctedFood" TEXT NOT NULL,
    "correctionCount" INTEGER NOT NULL DEFAULT 0,
    "avgConfidence" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "needsReview" BOOLEAN NOT NULL DEFAULT false,
    "reviewedAt" TIMESTAMP(3),
    "firstOccurrence" TIMESTAMP(3) NOT NULL,
    "lastOccurrence" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "FoodFeedbackAggregation_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Supplement" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "brand" TEXT,
    "dosageAmount" DOUBLE PRECISION NOT NULL,
    "dosageUnit" TEXT NOT NULL,
    "frequency" "SupplementFrequency" NOT NULL DEFAULT 'DAILY',
    "timesPerDay" INTEGER NOT NULL DEFAULT 1,
    "timeOfDay" "SupplementTimeOfDay"[],
    "withFood" BOOLEAN NOT NULL DEFAULT false,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "startDate" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endDate" TIMESTAMP(3),
    "notes" TEXT,
    "color" TEXT,
    "vitaminA" DOUBLE PRECISION,
    "vitaminC" DOUBLE PRECISION,
    "vitaminD" DOUBLE PRECISION,
    "vitaminE" DOUBLE PRECISION,
    "vitaminK" DOUBLE PRECISION,
    "vitaminB6" DOUBLE PRECISION,
    "vitaminB12" DOUBLE PRECISION,
    "folate" DOUBLE PRECISION,
    "thiamin" DOUBLE PRECISION,
    "riboflavin" DOUBLE PRECISION,
    "niacin" DOUBLE PRECISION,
    "calcium" DOUBLE PRECISION,
    "iron" DOUBLE PRECISION,
    "magnesium" DOUBLE PRECISION,
    "zinc" DOUBLE PRECISION,
    "potassium" DOUBLE PRECISION,
    "sodium" DOUBLE PRECISION,
    "phosphorus" DOUBLE PRECISION,
    "omega3" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Supplement_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SupplementLog" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "supplementId" TEXT NOT NULL,
    "takenAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "dosageAmount" DOUBLE PRECISION,
    "notes" TEXT,
    "skipped" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SupplementLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AdminUser" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "passwordHash" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "role" "AdminRole" NOT NULL DEFAULT 'SUPPORT',
    "mfaSecret" TEXT,
    "mfaEnabled" BOOLEAN NOT NULL DEFAULT false,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "lastLoginAt" TIMESTAMP(3),
    "lastLoginIp" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "AdminUser_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AdminAuditLog" (
    "id" TEXT NOT NULL,
    "adminUserId" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "targetType" TEXT,
    "targetId" TEXT,
    "details" JSONB,
    "ipAddress" TEXT NOT NULL,
    "userAgent" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AdminAuditLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AppStoreWebhookEvent" (
    "id" TEXT NOT NULL,
    "notificationType" TEXT NOT NULL,
    "subtype" TEXT,
    "originalTransactionId" TEXT,
    "transactionId" TEXT,
    "bundleId" TEXT,
    "payload" JSONB NOT NULL,
    "status" "WebhookEventStatus" NOT NULL DEFAULT 'PENDING',
    "errorMessage" TEXT,
    "retryCount" INTEGER NOT NULL DEFAULT 0,
    "lastRetryAt" TIMESTAMP(3),
    "userId" TEXT,
    "receivedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "processedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AppStoreWebhookEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FeatureFlag" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "type" "FeatureFlagType" NOT NULL DEFAULT 'BOOLEAN',
    "value" JSONB NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT false,
    "targeting" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "FeatureFlag_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- CreateIndex
CREATE UNIQUE INDEX "User_appleId_key" ON "User"("appleId");

-- CreateIndex
CREATE INDEX "User_email_idx" ON "User"("email");

-- CreateIndex
CREATE INDEX "User_appleId_idx" ON "User"("appleId");

-- CreateIndex
CREATE INDEX "User_resetToken_idx" ON "User"("resetToken");

-- CreateIndex
CREATE INDEX "Meal_userId_consumedAt_idx" ON "Meal"("userId", "consumedAt");

-- CreateIndex
CREATE INDEX "Meal_userId_mealType_idx" ON "Meal"("userId", "mealType");

-- CreateIndex
CREATE INDEX "WaterIntake_userId_recordedAt_idx" ON "WaterIntake"("userId", "recordedAt");

-- CreateIndex
CREATE INDEX "WeightRecord_userId_recordedAt_idx" ON "WeightRecord"("userId", "recordedAt");

-- CreateIndex
CREATE INDEX "HealthMetric_userId_recordedAt_idx" ON "HealthMetric"("userId", "recordedAt");

-- CreateIndex
CREATE INDEX "HealthMetric_userId_metricType_recordedAt_idx" ON "HealthMetric"("userId", "metricType", "recordedAt");

-- CreateIndex
CREATE UNIQUE INDEX "HealthMetric_userId_metricType_recordedAt_source_key" ON "HealthMetric"("userId", "metricType", "recordedAt", "source");

-- CreateIndex
CREATE INDEX "Activity_userId_startedAt_idx" ON "Activity"("userId", "startedAt");

-- CreateIndex
CREATE INDEX "Activity_userId_activityType_startedAt_idx" ON "Activity"("userId", "activityType", "startedAt");

-- CreateIndex
CREATE INDEX "MLFeature_userId_date_category_idx" ON "MLFeature"("userId", "date", "category");

-- CreateIndex
CREATE UNIQUE INDEX "MLFeature_userId_date_category_version_key" ON "MLFeature"("userId", "date", "category", "version");

-- CreateIndex
CREATE INDEX "MLPrediction_userId_targetDate_targetMetric_idx" ON "MLPrediction"("userId", "targetDate", "targetMetric");

-- CreateIndex
CREATE INDEX "MLPrediction_userId_modelId_createdAt_idx" ON "MLPrediction"("userId", "modelId", "createdAt");

-- CreateIndex
CREATE INDEX "MLInsight_userId_createdAt_idx" ON "MLInsight"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "MLInsight_userId_insightType_priority_idx" ON "MLInsight"("userId", "insightType", "priority");

-- CreateIndex
CREATE UNIQUE INDEX "UserMLProfile_userId_key" ON "UserMLProfile"("userId");

-- CreateIndex
CREATE INDEX "FoodFeedback_userId_idx" ON "FoodFeedback"("userId");

-- CreateIndex
CREATE INDEX "FoodFeedback_status_idx" ON "FoodFeedback"("status");

-- CreateIndex
CREATE INDEX "FoodFeedback_originalPrediction_idx" ON "FoodFeedback"("originalPrediction");

-- CreateIndex
CREATE UNIQUE INDEX "FoodFeedback_imageHash_originalPrediction_selectedFdcId_key" ON "FoodFeedback"("imageHash", "originalPrediction", "selectedFdcId");

-- CreateIndex
CREATE INDEX "FoodFeedbackAggregation_needsReview_reviewedAt_idx" ON "FoodFeedbackAggregation"("needsReview", "reviewedAt");

-- CreateIndex
CREATE INDEX "FoodFeedbackAggregation_correctionCount_idx" ON "FoodFeedbackAggregation"("correctionCount");

-- CreateIndex
CREATE UNIQUE INDEX "FoodFeedbackAggregation_originalPrediction_correctedFood_key" ON "FoodFeedbackAggregation"("originalPrediction", "correctedFood");

-- CreateIndex
CREATE INDEX "Supplement_userId_isActive_idx" ON "Supplement"("userId", "isActive");

-- CreateIndex
CREATE INDEX "Supplement_userId_createdAt_idx" ON "Supplement"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "SupplementLog_userId_takenAt_idx" ON "SupplementLog"("userId", "takenAt");

-- CreateIndex
CREATE INDEX "SupplementLog_supplementId_takenAt_idx" ON "SupplementLog"("supplementId", "takenAt");

-- CreateIndex
CREATE INDEX "SupplementLog_userId_supplementId_takenAt_idx" ON "SupplementLog"("userId", "supplementId", "takenAt");

-- CreateIndex
CREATE UNIQUE INDEX "AdminUser_email_key" ON "AdminUser"("email");

-- CreateIndex
CREATE INDEX "AdminUser_email_idx" ON "AdminUser"("email");

-- CreateIndex
CREATE INDEX "AdminUser_role_isActive_idx" ON "AdminUser"("role", "isActive");

-- CreateIndex
CREATE INDEX "AdminAuditLog_adminUserId_idx" ON "AdminAuditLog"("adminUserId");

-- CreateIndex
CREATE INDEX "AdminAuditLog_action_idx" ON "AdminAuditLog"("action");

-- CreateIndex
CREATE INDEX "AdminAuditLog_targetType_targetId_idx" ON "AdminAuditLog"("targetType", "targetId");

-- CreateIndex
CREATE INDEX "AdminAuditLog_createdAt_idx" ON "AdminAuditLog"("createdAt");

-- CreateIndex
CREATE INDEX "AppStoreWebhookEvent_notificationType_idx" ON "AppStoreWebhookEvent"("notificationType");

-- CreateIndex
CREATE INDEX "AppStoreWebhookEvent_originalTransactionId_idx" ON "AppStoreWebhookEvent"("originalTransactionId");

-- CreateIndex
CREATE INDEX "AppStoreWebhookEvent_userId_idx" ON "AppStoreWebhookEvent"("userId");

-- CreateIndex
CREATE INDEX "AppStoreWebhookEvent_status_idx" ON "AppStoreWebhookEvent"("status");

-- CreateIndex
CREATE INDEX "AppStoreWebhookEvent_receivedAt_idx" ON "AppStoreWebhookEvent"("receivedAt");

-- CreateIndex
CREATE UNIQUE INDEX "FeatureFlag_key_key" ON "FeatureFlag"("key");

-- CreateIndex
CREATE INDEX "FeatureFlag_key_idx" ON "FeatureFlag"("key");

-- CreateIndex
CREATE INDEX "FeatureFlag_isEnabled_idx" ON "FeatureFlag"("isEnabled");

-- AddForeignKey
ALTER TABLE "Meal" ADD CONSTRAINT "Meal_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "WaterIntake" ADD CONSTRAINT "WaterIntake_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "WeightRecord" ADD CONSTRAINT "WeightRecord_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "HealthMetric" ADD CONSTRAINT "HealthMetric_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Activity" ADD CONSTRAINT "Activity_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MLFeature" ADD CONSTRAINT "MLFeature_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MLPrediction" ADD CONSTRAINT "MLPrediction_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MLInsight" ADD CONSTRAINT "MLInsight_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserMLProfile" ADD CONSTRAINT "UserMLProfile_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Supplement" ADD CONSTRAINT "Supplement_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SupplementLog" ADD CONSTRAINT "SupplementLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SupplementLog" ADD CONSTRAINT "SupplementLog_supplementId_fkey" FOREIGN KEY ("supplementId") REFERENCES "Supplement"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AdminAuditLog" ADD CONSTRAINT "AdminAuditLog_adminUserId_fkey" FOREIGN KEY ("adminUserId") REFERENCES "AdminUser"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

