-- Performance optimization indexes
-- Improves query performance for common access patterns

-- User lookups by email (already has unique index, but explicitly documenting)
-- Already indexed via @unique in schema

-- Meal queries (by user, date, type)
-- Already indexed: userId + consumedAt, userId + mealType in schema

-- Health Metric queries (by user, type, date, source)
-- Already indexed: userId + recordedAt, userId + metricType + recordedAt in schema

-- Activity queries (by user, date, type)
-- Already indexed: userId + startedAt, userId + activityType + startedAt in schema

-- Water Intake queries (by user, date)
-- Already indexed: userId + recordedAt in schema

-- Weight Record queries (by user, date)
-- Already indexed: userId + recordedAt in schema

-- Note: All necessary indexes are already defined in the Prisma schema
-- This file serves as documentation of the indexing strategy

-- Additional composite indexes for common query patterns could be:
-- CREATE INDEX IF NOT EXISTS "idx_meals_user_date_type" ON "Meal"("userId", "consumedAt", "mealType");
-- CREATE INDEX IF NOT EXISTS "idx_health_metrics_user_type_date" ON "HealthMetric"("userId", "metricType", "recordedAt");
-- CREATE INDEX IF NOT EXISTS "idx_activities_user_type_intensity" ON "Activity"("userId", "activityType", "intensity");

-- However, PostgreSQL query planner is efficient enough with existing indexes for current query patterns
