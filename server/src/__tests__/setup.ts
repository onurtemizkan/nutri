/**
 * Jest Test Setup
 *
 * Configures the test environment:
 * - Sets up environment variables
 * - Initializes test database
 * - Provides cleanup utilities
 * - Configures global test lifecycle hooks
 */

import { PrismaClient, Activity, HealthMetric, ActivityType, ActivityIntensity, HealthMetricType } from '@prisma/client';

// Set test environment variables before any imports
(process.env as { NODE_ENV: string }).NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-secret-key-min-32-chars-long-for-testing!!!';
process.env.PORT = '3001'; // Different port to avoid conflicts

// Use PostgreSQL test database
process.env.DATABASE_URL = 'postgresql://postgres:postgres@localhost:5432/nutri_test_db';

// Global Prisma client for tests
export const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL,
    },
  },
  log: [], // Disable logging during tests for cleaner output
});

/**
 * Clean database by deleting all records
 * Preserves schema, only removes data
 */
export async function cleanDatabase() {
  // Delete in reverse order of dependencies to avoid foreign key constraints
  // Using $transaction to ensure atomicity
  await prisma.$transaction([
    prisma.healthMetric.deleteMany(),
    prisma.activity.deleteMany(),
    prisma.meal.deleteMany(),
    prisma.waterIntake.deleteMany(),
    prisma.weightRecord.deleteMany(),
    prisma.user.deleteMany(),
  ]);
}

/**
 * Reset database to pristine state
 * Useful for integration tests
 */
export async function resetDatabase() {
  try {
    // Push the schema to the test database
    const { execSync } = require('child_process');
    execSync('npx prisma db push --force-reset --skip-generate', {
      env: { ...process.env, DATABASE_URL: process.env.DATABASE_URL },
      stdio: 'ignore', // Suppress output
    });
  } catch (error) {
    console.error('Error resetting database:', error);
    throw error;
  }
}

// ============================================================================
// Global Test Lifecycle Hooks
// ============================================================================

/**
 * Before all tests: Initialize database
 * Note: Gracefully handles missing database for unit tests
 */
beforeAll(async () => {
  try {
    // Connect to database
    await prisma.$connect();
    // Clean database to ensure fresh start
    await cleanDatabase();
  } catch (error) {
    // Database may not exist for unit tests - this is OK
    const errorMessage = error instanceof Error ? error.message : String(error);
    if (errorMessage.includes('does not exist') || errorMessage.includes('Connection refused')) {
      console.warn('Database not available - skipping database setup for unit tests');
    } else {
      throw error;
    }
  }
});

/**
 * Before each test: Clean database for test isolation
 * This ensures each test starts with a clean slate
 * Note: Gracefully handles missing database for unit tests
 */
beforeEach(async () => {
  try {
    await cleanDatabase();
  } catch {
    // Database may not exist for unit tests - this is OK
  }
});

/**
 * After all tests: Disconnect and cleanup
 * Note: Gracefully handles missing database for unit tests
 */
afterAll(async () => {
  try {
    await prisma.$disconnect();
  } catch {
    // Database may not exist for unit tests - this is OK
  }

  // Clean up test database file (SQLite)
  if (process.env.DATABASE_URL?.includes('file:')) {
    try {
      const fs = require('fs');
      const path = require('path');
      const dbPath = path.join(__dirname, '../../prisma/test.db');
      if (fs.existsSync(dbPath)) {
        fs.unlinkSync(dbPath);
      }
    } catch {
      // Ignore cleanup errors
    }
  }
});

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Create a test user with default values
 */
export async function createTestUser(overrides?: Partial<{
  email: string;
  password: string;
  name: string;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
}>) {
  const bcrypt = require('bcryptjs');

  const defaultUser = {
    email: 'test@example.com',
    password: await bcrypt.hash('password123', 10),
    name: 'Test User',
    goalCalories: 2000,
    goalProtein: 150,
    goalCarbs: 250,
    goalFat: 65,
    ...overrides,
  };

  return prisma.user.create({
    data: defaultUser,
  });
}

/**
 * Create a test JWT token for a user
 */
export function createTestToken(userId: string): string {
  const jwt = require('jsonwebtoken');
  return jwt.sign(
    { userId },
    process.env.JWT_SECRET,
    { expiresIn: '1h' }
  );
}

/**
 * Create a test meal
 */
export async function createTestMeal(userId: string, overrides?: Partial<{
  name: string;
  mealType: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  consumedAt: Date;
}>) {
  const defaultMeal = {
    userId,
    name: 'Test Meal',
    mealType: 'lunch',
    calories: 500,
    protein: 30,
    carbs: 60,
    fat: 15,
    consumedAt: new Date(),
    ...overrides,
  };

  return prisma.meal.create({
    data: defaultMeal,
  });
}

/**
 * Create a test activity
 */
export async function createTestActivity(userId: string, overrides?: Partial<{
  startedAt: Date;
  endedAt: Date;
  duration: number;
  activityType: ActivityType;
  intensity: ActivityIntensity;
  caloriesBurned: number;
  source: string;
}>): Promise<Activity> {
  const now = new Date();
  const defaultActivity = {
    userId,
    startedAt: now,
    endedAt: new Date(now.getTime() + 30 * 60 * 1000), // 30 minutes later
    duration: 30,
    activityType: ActivityType.RUNNING,
    intensity: ActivityIntensity.MODERATE,
    caloriesBurned: 250,
    source: 'manual',
    ...overrides,
  };

  return prisma.activity.create({
    data: defaultActivity,
  });
}

/**
 * Create a test health metric
 */
export async function createTestHealthMetric(userId: string, overrides?: Partial<{
  recordedAt: Date;
  metricType: HealthMetricType;
  value: number;
  unit: string;
  source: string;
}>): Promise<HealthMetric> {
  const defaultMetric = {
    userId,
    recordedAt: new Date(),
    metricType: HealthMetricType.RESTING_HEART_RATE,
    value: 60.0,
    unit: 'bpm',
    source: 'manual',
    ...overrides,
  };

  return prisma.healthMetric.create({
    data: defaultMetric,
  });
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/**
 * Assert that response is a valid JWT token
 */
export function assertValidToken(token: string) {
  expect(token).toBeDefined();
  expect(typeof token).toBe('string');
  expect(token.split('.')).toHaveLength(3); // JWT has 3 parts
}

/**
 * Assert that response has user data structure
 */
export function assertUserStructure(user: unknown): void {
  expect(user).toHaveProperty('id');
  expect(user).toHaveProperty('email');
  expect(user).toHaveProperty('name');
  expect(user).not.toHaveProperty('password'); // Password should never be returned
}

/**
 * Assert that response has meal data structure
 */
export function assertMealStructure(meal: unknown): void {
  expect(meal).toHaveProperty('id');
  expect(meal).toHaveProperty('userId');
  expect(meal).toHaveProperty('name');
  expect(meal).toHaveProperty('calories');
  expect(meal).toHaveProperty('protein');
  expect(meal).toHaveProperty('carbs');
  expect(meal).toHaveProperty('fat');
}

/**
 * Assert that response has activity data structure
 */
export function assertActivityStructure(activity: unknown): void {
  expect(activity).toHaveProperty('id');
  expect(activity).toHaveProperty('userId');
  expect(activity).toHaveProperty('startedAt');
  expect(activity).toHaveProperty('endedAt');
  expect(activity).toHaveProperty('duration');
  expect(activity).toHaveProperty('activityType');
  expect(activity).toHaveProperty('intensity');
}

/**
 * Assert that response has health metric data structure
 */
export function assertHealthMetricStructure(metric: unknown): void {
  expect(metric).toHaveProperty('id');
  expect(metric).toHaveProperty('userId');
  expect(metric).toHaveProperty('metricType');
  expect(metric).toHaveProperty('value');
  expect(metric).toHaveProperty('recordedAt');
  expect(metric).toHaveProperty('unit');
  expect(metric).toHaveProperty('source');
}
