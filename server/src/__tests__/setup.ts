/**
 * Jest Test Setup
 *
 * Configures the test environment:
 * - Sets up environment variables
 * - Initializes test database
 * - Provides cleanup utilities
 * - Configures global test lifecycle hooks
 */

import { PrismaClient } from '@prisma/client';

// Set test environment variables before any imports
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-secret-key-min-32-chars-long-for-testing!!!';
process.env.PORT = '3001'; // Different port to avoid conflicts

// Use in-memory SQLite for fast, isolated tests
// In production tests, you might want to use a separate PostgreSQL test database
process.env.DATABASE_URL = 'file:./test.db';

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
  try {
    // Delete in reverse order of dependencies to avoid foreign key constraints
    await prisma.healthMetric.deleteMany();
    await prisma.activity.deleteMany();
    await prisma.meal.deleteMany();
    await prisma.waterIntake.deleteMany();
    await prisma.weightRecord.deleteMany();
    await prisma.user.deleteMany();
  } catch (error) {
    console.error('Error cleaning database:', error);
    // Don't throw - some tables might not exist yet, which is okay for tests
  }
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
 */
beforeAll(async () => {
  // Connect to database
  // Note: Schema is auto-created by Prisma for SQLite
  await prisma.$connect();
});

/**
 * After each test: Clean up test data
 * Ensures test isolation
 */
afterEach(async () => {
  await cleanDatabase();
});

/**
 * After all tests: Disconnect and cleanup
 */
afterAll(async () => {
  await prisma.$disconnect();

  // Clean up test database file (SQLite)
  if (process.env.DATABASE_URL?.includes('file:')) {
    try {
      const fs = require('fs');
      const path = require('path');
      const dbPath = path.join(__dirname, '../../prisma/test.db');
      if (fs.existsSync(dbPath)) {
        fs.unlinkSync(dbPath);
      }
    } catch (error) {
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
  activityType: any;
  intensity: any;
  caloriesBurned: number;
  source: string;
}>) {
  const now = new Date();
  const defaultActivity = {
    userId,
    startedAt: now,
    endedAt: new Date(now.getTime() + 30 * 60 * 1000), // 30 minutes later
    duration: 30,
    activityType: 'RUNNING' as any,
    intensity: 'MODERATE' as any,
    caloriesBurned: 250,
    source: 'manual',
    ...overrides,
  };

  return prisma.activity.create({
    data: defaultActivity as any,
  });
}

/**
 * Create a test health metric
 */
export async function createTestHealthMetric(userId: string, overrides?: Partial<{
  recordedAt: Date;
  metricType: any;
  value: number;
  unit: string;
  source: string;
}>) {
  const defaultMetric = {
    userId,
    recordedAt: new Date(),
    metricType: 'RESTING_HEART_RATE' as any,
    value: 60.0,
    unit: 'bpm',
    source: 'manual',
    ...overrides,
  };

  return prisma.healthMetric.create({
    data: defaultMetric as any,
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
export function assertUserStructure(user: any) {
  expect(user).toHaveProperty('id');
  expect(user).toHaveProperty('email');
  expect(user).toHaveProperty('name');
  expect(user).not.toHaveProperty('password'); // Password should never be returned
}

/**
 * Assert that response has meal data structure
 */
export function assertMealStructure(meal: any) {
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
export function assertActivityStructure(activity: any) {
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
export function assertHealthMetricStructure(metric: any) {
  expect(metric).toHaveProperty('id');
  expect(metric).toHaveProperty('userId');
  expect(metric).toHaveProperty('metricType');
  expect(metric).toHaveProperty('value');
  expect(metric).toHaveProperty('recordedAt');
  expect(metric).toHaveProperty('unit');
  expect(metric).toHaveProperty('source');
}
