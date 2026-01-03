-- CreateEnum
CREATE TYPE "SubscriptionStatus" AS ENUM ('ACTIVE', 'EXPIRED', 'IN_GRACE_PERIOD', 'IN_BILLING_RETRY', 'REVOKED', 'REFUNDED');

-- CreateEnum
CREATE TYPE "SubscriptionEnvironment" AS ENUM ('SANDBOX', 'PRODUCTION');

-- CreateEnum
CREATE TYPE "GlucoseSource" AS ENUM ('DEXCOM', 'LIBRE', 'LEVELS', 'MANUAL');

-- CreateEnum
CREATE TYPE "GlucoseTrend" AS ENUM ('RISING_RAPIDLY', 'RISING', 'RISING_SLIGHTLY', 'STABLE', 'FALLING_SLIGHTLY', 'FALLING', 'FALLING_RAPIDLY', 'NOT_AVAILABLE');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "goalWater" INTEGER NOT NULL DEFAULT 2000,
ADD COLUMN     "isActive" BOOLEAN NOT NULL DEFAULT true;

-- CreateTable
CREATE TABLE "Subscription" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "productId" TEXT NOT NULL,
    "originalTransactionId" TEXT NOT NULL,
    "status" "SubscriptionStatus" NOT NULL DEFAULT 'ACTIVE',
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "cancelledAt" TIMESTAMP(3),
    "isTrialPeriod" BOOLEAN NOT NULL DEFAULT false,
    "isIntroOfferPeriod" BOOLEAN NOT NULL DEFAULT false,
    "trialEndsAt" TIMESTAMP(3),
    "autoRenewEnabled" BOOLEAN NOT NULL DEFAULT true,
    "autoRenewProductId" TEXT,
    "gracePeriodExpiresAt" TIMESTAMP(3),
    "billingRetryExpiresAt" TIMESTAMP(3),
    "priceLocale" TEXT,
    "priceCurrency" TEXT,
    "priceAmount" DECIMAL(10,2),
    "environment" "SubscriptionEnvironment" NOT NULL DEFAULT 'PRODUCTION',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Subscription_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SubscriptionEvent" (
    "id" TEXT NOT NULL,
    "subscriptionId" TEXT NOT NULL,
    "notificationType" TEXT NOT NULL,
    "subtype" TEXT,
    "transactionId" TEXT,
    "originalTransactionId" TEXT NOT NULL,
    "notificationUUID" TEXT,
    "eventData" JSONB,
    "processedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SubscriptionEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "GlucoseReading" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "value" DOUBLE PRECISION NOT NULL,
    "unit" TEXT NOT NULL DEFAULT 'mg/dL',
    "source" "GlucoseSource" NOT NULL,
    "sourceId" TEXT,
    "trendArrow" "GlucoseTrend",
    "trendRate" DOUBLE PRECISION,
    "recordedAt" TIMESTAMP(3) NOT NULL,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "GlucoseReading_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "MealGlucoseResponse" (
    "id" TEXT NOT NULL,
    "mealId" TEXT NOT NULL,
    "baselineGlucose" DOUBLE PRECISION NOT NULL,
    "baselineTime" TIMESTAMP(3) NOT NULL,
    "peakGlucose" DOUBLE PRECISION NOT NULL,
    "peakTime" INTEGER NOT NULL,
    "glucoseRise" DOUBLE PRECISION NOT NULL,
    "returnToBaseline" INTEGER,
    "twoHourGlucose" DOUBLE PRECISION,
    "areaUnderCurve" DOUBLE PRECISION NOT NULL,
    "glucoseScore" DOUBLE PRECISION NOT NULL,
    "readingCount" INTEGER NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "analyzedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "windowStart" TIMESTAMP(3) NOT NULL,
    "windowEnd" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "MealGlucoseResponse_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CGMConnection" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "provider" "GlucoseSource" NOT NULL,
    "accessToken" TEXT NOT NULL,
    "refreshToken" TEXT,
    "tokenType" TEXT NOT NULL DEFAULT 'Bearer',
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "scope" TEXT,
    "lastSyncAt" TIMESTAMP(3),
    "lastSyncStatus" TEXT,
    "lastSyncError" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "connectedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "disconnectedAt" TIMESTAMP(3),
    "externalUserId" TEXT,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "CGMConnection_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Subscription_userId_key" ON "Subscription"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "Subscription_originalTransactionId_key" ON "Subscription"("originalTransactionId");

-- CreateIndex
CREATE INDEX "Subscription_userId_idx" ON "Subscription"("userId");

-- CreateIndex
CREATE INDEX "Subscription_originalTransactionId_idx" ON "Subscription"("originalTransactionId");

-- CreateIndex
CREATE INDEX "Subscription_status_expiresAt_idx" ON "Subscription"("status", "expiresAt");

-- CreateIndex
CREATE INDEX "Subscription_productId_idx" ON "Subscription"("productId");

-- CreateIndex
CREATE UNIQUE INDEX "SubscriptionEvent_notificationUUID_key" ON "SubscriptionEvent"("notificationUUID");

-- CreateIndex
CREATE INDEX "SubscriptionEvent_subscriptionId_idx" ON "SubscriptionEvent"("subscriptionId");

-- CreateIndex
CREATE INDEX "SubscriptionEvent_notificationType_idx" ON "SubscriptionEvent"("notificationType");

-- CreateIndex
CREATE INDEX "SubscriptionEvent_originalTransactionId_idx" ON "SubscriptionEvent"("originalTransactionId");

-- CreateIndex
CREATE INDEX "SubscriptionEvent_createdAt_idx" ON "SubscriptionEvent"("createdAt");

-- CreateIndex
CREATE INDEX "GlucoseReading_userId_recordedAt_idx" ON "GlucoseReading"("userId", "recordedAt");

-- CreateIndex
CREATE INDEX "GlucoseReading_userId_source_recordedAt_idx" ON "GlucoseReading"("userId", "source", "recordedAt");

-- CreateIndex
CREATE UNIQUE INDEX "GlucoseReading_userId_source_recordedAt_key" ON "GlucoseReading"("userId", "source", "recordedAt");

-- CreateIndex
CREATE UNIQUE INDEX "MealGlucoseResponse_mealId_key" ON "MealGlucoseResponse"("mealId");

-- CreateIndex
CREATE INDEX "MealGlucoseResponse_mealId_idx" ON "MealGlucoseResponse"("mealId");

-- CreateIndex
CREATE INDEX "CGMConnection_userId_isActive_idx" ON "CGMConnection"("userId", "isActive");

-- CreateIndex
CREATE INDEX "CGMConnection_provider_isActive_idx" ON "CGMConnection"("provider", "isActive");

-- CreateIndex
CREATE UNIQUE INDEX "CGMConnection_userId_provider_key" ON "CGMConnection"("userId", "provider");

-- AddForeignKey
ALTER TABLE "Subscription" ADD CONSTRAINT "Subscription_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SubscriptionEvent" ADD CONSTRAINT "SubscriptionEvent_subscriptionId_fkey" FOREIGN KEY ("subscriptionId") REFERENCES "Subscription"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GlucoseReading" ADD CONSTRAINT "GlucoseReading_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "MealGlucoseResponse" ADD CONSTRAINT "MealGlucoseResponse_mealId_fkey" FOREIGN KEY ("mealId") REFERENCES "Meal"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CGMConnection" ADD CONSTRAINT "CGMConnection_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
