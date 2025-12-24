-- CreateEnum
CREATE TYPE "BiologicalSex" AS ENUM ('MALE', 'FEMALE', 'OTHER', 'PREFER_NOT_TO_SAY');

-- CreateEnum
CREATE TYPE "PrimaryGoal" AS ENUM ('WEIGHT_LOSS', 'MUSCLE_GAIN', 'MAINTENANCE', 'GENERAL_HEALTH', 'ATHLETIC_PERFORMANCE', 'BETTER_SLEEP', 'STRESS_REDUCTION');

-- CreateEnum
CREATE TYPE "NicotineUseLevel" AS ENUM ('NONE', 'OCCASIONAL', 'DAILY', 'HEAVY');

-- CreateEnum
CREATE TYPE "AlcoholUseLevel" AS ENUM ('NONE', 'OCCASIONAL', 'MODERATE', 'FREQUENT');

-- CreateEnum
CREATE TYPE "DevicePlatform" AS ENUM ('IOS', 'ANDROID');

-- CreateEnum
CREATE TYPE "NotificationCategory" AS ENUM ('MEAL_REMINDER', 'GOAL_PROGRESS', 'HEALTH_INSIGHT', 'SUPPLEMENT_REMINDER', 'STREAK_ALERT', 'WEEKLY_SUMMARY', 'MARKETING', 'SYSTEM');

-- CreateEnum
CREATE TYPE "NotificationStatus" AS ENUM ('PENDING', 'SENT', 'DELIVERED', 'OPENED', 'FAILED');

-- CreateEnum
CREATE TYPE "CampaignStatus" AS ENUM ('DRAFT', 'SCHEDULED', 'SENDING', 'COMPLETED', 'CANCELLED', 'FAILED');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "biologicalSex" "BiologicalSex",
ADD COLUMN     "dateOfBirth" TIMESTAMP(3),
ADD COLUMN     "dietaryPreferences" JSONB,
ADD COLUMN     "primaryGoal" "PrimaryGoal",
ADD COLUMN     "timezone" TEXT;

-- CreateTable
CREATE TABLE "LearnedPrompt" (
    "id" SERIAL NOT NULL,
    "foodKey" VARCHAR(100) NOT NULL,
    "prompt" TEXT NOT NULL,
    "source" VARCHAR(50) NOT NULL,
    "timesUsed" INTEGER NOT NULL DEFAULT 0,
    "successCount" INTEGER NOT NULL DEFAULT 0,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "LearnedPrompt_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserOnboarding" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "currentStep" INTEGER NOT NULL DEFAULT 1,
    "totalSteps" INTEGER NOT NULL DEFAULT 6,
    "completedAt" TIMESTAMP(3),
    "skippedSteps" INTEGER[] DEFAULT ARRAY[]::INTEGER[],
    "version" TEXT NOT NULL DEFAULT '1.0',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserOnboarding_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserHealthBackground" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "chronicConditions" JSONB,
    "medications" JSONB,
    "supplements" JSONB,
    "allergies" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserHealthBackground_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserLifestyle" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "nicotineUse" "NicotineUseLevel",
    "nicotineType" TEXT,
    "alcoholUse" "AlcoholUseLevel",
    "caffeineDaily" INTEGER,
    "typicalBedtime" TEXT,
    "typicalWakeTime" TEXT,
    "sleepQuality" INTEGER,
    "stressLevel" INTEGER,
    "workSchedule" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserLifestyle_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserPermissions" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "notificationsEnabled" BOOLEAN NOT NULL DEFAULT false,
    "notificationTypes" JSONB,
    "healthKitEnabled" BOOLEAN NOT NULL DEFAULT false,
    "healthKitScopes" JSONB,
    "healthConnectEnabled" BOOLEAN NOT NULL DEFAULT false,
    "healthConnectScopes" JSONB,
    "shareAnonymousData" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserPermissions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DeviceToken" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "platform" "DevicePlatform" NOT NULL,
    "expoPushToken" TEXT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "lastActiveAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deviceModel" TEXT,
    "osVersion" TEXT,
    "appVersion" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "DeviceToken_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NotificationPreference" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "enabledCategories" "NotificationCategory"[],
    "quietHoursEnabled" BOOLEAN NOT NULL DEFAULT false,
    "quietHoursStart" TEXT,
    "quietHoursEnd" TEXT,
    "mealReminderTimes" JSONB,
    "settings" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "NotificationPreference_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NotificationLog" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "category" "NotificationCategory" NOT NULL,
    "title" TEXT NOT NULL,
    "body" TEXT NOT NULL,
    "data" JSONB,
    "platform" "DevicePlatform" NOT NULL,
    "deviceToken" TEXT,
    "expoPushToken" TEXT,
    "interruptionLevel" TEXT,
    "threadId" TEXT,
    "relevanceScore" DOUBLE PRECISION,
    "campaignId" TEXT,
    "sentAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deliveredAt" TIMESTAMP(3),
    "openedAt" TIMESTAMP(3),
    "actionTaken" TEXT,
    "error" TEXT,
    "status" "NotificationStatus" NOT NULL DEFAULT 'SENT',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NotificationLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NotificationCampaign" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "notificationTitle" TEXT NOT NULL,
    "notificationBody" TEXT NOT NULL,
    "category" "NotificationCategory" NOT NULL,
    "data" JSONB,
    "targetSegment" JSONB NOT NULL,
    "estimatedRecipients" INTEGER,
    "isAbTest" BOOLEAN NOT NULL DEFAULT false,
    "variants" JSONB,
    "winningVariant" TEXT,
    "scheduledAt" TIMESTAMP(3),
    "sentAt" TIMESTAMP(3),
    "status" "CampaignStatus" NOT NULL DEFAULT 'DRAFT',
    "deliveryCount" INTEGER NOT NULL DEFAULT 0,
    "openCount" INTEGER NOT NULL DEFAULT 0,
    "actionCount" INTEGER NOT NULL DEFAULT 0,
    "failureCount" INTEGER NOT NULL DEFAULT 0,
    "createdByAdminId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "NotificationCampaign_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NotificationTemplate" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "category" "NotificationCategory" NOT NULL,
    "title" TEXT NOT NULL,
    "body" TEXT NOT NULL,
    "data" JSONB,
    "variables" TEXT[],
    "usageCount" INTEGER NOT NULL DEFAULT 0,
    "locale" TEXT NOT NULL DEFAULT 'en',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "NotificationTemplate_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LearnedPrompt_foodKey_isActive_idx" ON "LearnedPrompt"("foodKey", "isActive");

-- CreateIndex
CREATE UNIQUE INDEX "UserOnboarding_userId_key" ON "UserOnboarding"("userId");

-- CreateIndex
CREATE INDEX "UserOnboarding_userId_idx" ON "UserOnboarding"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "UserHealthBackground_userId_key" ON "UserHealthBackground"("userId");

-- CreateIndex
CREATE INDEX "UserHealthBackground_userId_idx" ON "UserHealthBackground"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "UserLifestyle_userId_key" ON "UserLifestyle"("userId");

-- CreateIndex
CREATE INDEX "UserLifestyle_userId_idx" ON "UserLifestyle"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "UserPermissions_userId_key" ON "UserPermissions"("userId");

-- CreateIndex
CREATE INDEX "UserPermissions_userId_idx" ON "UserPermissions"("userId");

-- CreateIndex
CREATE INDEX "DeviceToken_userId_isActive_idx" ON "DeviceToken"("userId", "isActive");

-- CreateIndex
CREATE INDEX "DeviceToken_token_idx" ON "DeviceToken"("token");

-- CreateIndex
CREATE INDEX "DeviceToken_userId_platform_idx" ON "DeviceToken"("userId", "platform");

-- CreateIndex
CREATE UNIQUE INDEX "DeviceToken_userId_token_platform_key" ON "DeviceToken"("userId", "token", "platform");

-- CreateIndex
CREATE UNIQUE INDEX "NotificationPreference_userId_key" ON "NotificationPreference"("userId");

-- CreateIndex
CREATE INDEX "NotificationPreference_userId_idx" ON "NotificationPreference"("userId");

-- CreateIndex
CREATE INDEX "NotificationLog_userId_sentAt_idx" ON "NotificationLog"("userId", "sentAt");

-- CreateIndex
CREATE INDEX "NotificationLog_category_idx" ON "NotificationLog"("category");

-- CreateIndex
CREATE INDEX "NotificationLog_campaignId_idx" ON "NotificationLog"("campaignId");

-- CreateIndex
CREATE INDEX "NotificationLog_status_idx" ON "NotificationLog"("status");

-- CreateIndex
CREATE INDEX "NotificationLog_userId_category_sentAt_idx" ON "NotificationLog"("userId", "category", "sentAt");

-- CreateIndex
CREATE INDEX "NotificationCampaign_status_idx" ON "NotificationCampaign"("status");

-- CreateIndex
CREATE INDEX "NotificationCampaign_scheduledAt_idx" ON "NotificationCampaign"("scheduledAt");

-- CreateIndex
CREATE INDEX "NotificationCampaign_category_idx" ON "NotificationCampaign"("category");

-- CreateIndex
CREATE INDEX "NotificationCampaign_createdByAdminId_idx" ON "NotificationCampaign"("createdByAdminId");

-- CreateIndex
CREATE INDEX "NotificationTemplate_category_idx" ON "NotificationTemplate"("category");

-- CreateIndex
CREATE INDEX "NotificationTemplate_locale_idx" ON "NotificationTemplate"("locale");

-- CreateIndex
CREATE UNIQUE INDEX "NotificationTemplate_name_locale_key" ON "NotificationTemplate"("name", "locale");

-- AddForeignKey
ALTER TABLE "UserOnboarding" ADD CONSTRAINT "UserOnboarding_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserHealthBackground" ADD CONSTRAINT "UserHealthBackground_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserLifestyle" ADD CONSTRAINT "UserLifestyle_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserPermissions" ADD CONSTRAINT "UserPermissions_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "DeviceToken" ADD CONSTRAINT "DeviceToken_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NotificationPreference" ADD CONSTRAINT "NotificationPreference_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NotificationLog" ADD CONSTRAINT "NotificationLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NotificationLog" ADD CONSTRAINT "NotificationLog_campaignId_fkey" FOREIGN KEY ("campaignId") REFERENCES "NotificationCampaign"("id") ON DELETE SET NULL ON UPDATE CASCADE;
