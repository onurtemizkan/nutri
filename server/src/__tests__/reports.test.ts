/**
 * Reports API Endpoints Tests
 *
 * Tests all report-related endpoints:
 * - GET /api/reports/weekly - Weekly nutrition report
 * - GET /api/reports/monthly - Monthly nutrition report
 * - GET /api/reports/weekly/export - Export weekly report
 * - GET /api/reports/monthly/export - Export monthly report
 */

import request from 'supertest';
import app from '../index';
import {
  createTestUser,
  createTestToken,
  createTestMeal,
  createTestHealthMetric,
  createTestActivity,
} from './setup';
import { HealthMetricType, ActivityType, ActivityIntensity } from '@prisma/client';

describe('Reports API Endpoints', () => {
  let userId: string;
  let authToken: string;

  // Create a test user before each test
  beforeEach(async () => {
    const user = await createTestUser({ email: 'reports-test@example.com' });
    userId = user.id;
    authToken = createTestToken(userId);
  });

  // ============================================================================
  // GET /api/reports/weekly
  // ============================================================================

  describe('GET /api/reports/weekly', () => {
    it('should return a weekly report for current week', async () => {
      // Create some test meals for the current week
      const today = new Date();
      await createTestMeal(userId, {
        name: 'Breakfast',
        mealType: 'breakfast',
        calories: 400,
        protein: 25,
        carbs: 40,
        fat: 15,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Lunch',
        mealType: 'lunch',
        calories: 600,
        protein: 35,
        carbs: 60,
        fat: 20,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // Verify report structure
      expect(response.body).toHaveProperty('userId', userId);
      expect(response.body).toHaveProperty('generatedAt');
      expect(response.body).toHaveProperty('periodStart');
      expect(response.body).toHaveProperty('periodEnd');
      expect(response.body).toHaveProperty('dailyBreakdowns');
      expect(response.body).toHaveProperty('totals');
      expect(response.body).toHaveProperty('averages');
      expect(response.body).toHaveProperty('goalProgress');
      expect(response.body).toHaveProperty('trends');
      expect(response.body).toHaveProperty('streak');
      expect(response.body).toHaveProperty('macroDistribution');
      expect(response.body).toHaveProperty('mealTypeBreakdown');
      expect(response.body).toHaveProperty('insights');
      expect(response.body).toHaveProperty('achievements');
      expect(response.body).toHaveProperty('topFoods');

      // Check that dailyBreakdowns is an array
      expect(Array.isArray(response.body.dailyBreakdowns)).toBe(true);

      // Verify totals reflect the meals we created
      expect(response.body.totals).toHaveProperty('calories');
      expect(response.body.totals).toHaveProperty('protein');
      expect(response.body.totals).toHaveProperty('carbs');
      expect(response.body.totals).toHaveProperty('fat');
    });

    it('should return a weekly report for a specific date', async () => {
      // Use a date in the past
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 14); // 2 weeks ago
      const dateStr = pastDate.toISOString().split('T')[0];

      const response = await request(app)
        .get(`/api/reports/weekly?date=${dateStr}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // Verify report structure
      expect(response.body).toHaveProperty('periodStart');
      expect(response.body).toHaveProperty('periodEnd');

      // The period should contain the specified date
      const periodStart = new Date(response.body.periodStart);
      const periodEnd = new Date(response.body.periodEnd);
      expect(pastDate >= periodStart).toBe(true);
      expect(pastDate <= periodEnd).toBe(true);
    });

    it('should return empty data for a week with no meals', async () => {
      // Request a week far in the past with no data
      const response = await request(app)
        .get('/api/reports/weekly?date=2020-01-01')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.totals.calories).toBe(0);
      expect(response.body.totals.protein).toBe(0);
      expect(response.body.totals.carbs).toBe(0);
      expect(response.body.totals.fat).toBe(0);
      expect(response.body.daysGoalsMet).toBe(0);
    });

    it('should return 401 without authentication', async () => {
      await request(app).get('/api/reports/weekly').expect(401);
    });

    it('should return 400 for invalid date format', async () => {
      const response = await request(app)
        .get('/api/reports/weekly?date=invalid-date')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/reports/monthly
  // ============================================================================

  describe('GET /api/reports/monthly', () => {
    it('should return a monthly report for current month', async () => {
      // Create some test meals for the current month
      const today = new Date();
      await createTestMeal(userId, {
        name: 'Breakfast',
        mealType: 'breakfast',
        calories: 400,
        protein: 25,
        carbs: 40,
        fat: 15,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/monthly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // Verify report structure
      expect(response.body).toHaveProperty('userId', userId);
      expect(response.body).toHaveProperty('generatedAt');
      expect(response.body).toHaveProperty('month');
      expect(response.body).toHaveProperty('periodStart');
      expect(response.body).toHaveProperty('periodEnd');
      expect(response.body).toHaveProperty('weeklyBreakdowns');
      expect(response.body).toHaveProperty('dailyBreakdowns');
      expect(response.body).toHaveProperty('totals');
      expect(response.body).toHaveProperty('averages');
      expect(response.body).toHaveProperty('goalProgress');
      expect(response.body).toHaveProperty('totalDaysTracked');
      expect(response.body).toHaveProperty('trends');
      expect(response.body).toHaveProperty('bestDays');
      expect(response.body).toHaveProperty('worstDays');
      expect(response.body).toHaveProperty('streak');
      expect(response.body).toHaveProperty('weeklyTrends');
      expect(response.body).toHaveProperty('yearOverYear');

      // Check arrays
      expect(Array.isArray(response.body.weeklyBreakdowns)).toBe(true);
      expect(Array.isArray(response.body.dailyBreakdowns)).toBe(true);
      expect(Array.isArray(response.body.bestDays)).toBe(true);
    });

    it('should return a monthly report for a specific month', async () => {
      const response = await request(app)
        .get('/api/reports/monthly?month=2024-06')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.month).toBe('2024-06');

      // Verify period is correct for June 2024
      const periodStart = new Date(response.body.periodStart);
      const periodEnd = new Date(response.body.periodEnd);
      expect(periodStart.getMonth()).toBe(5); // June is 5 (0-indexed)
      expect(periodStart.getFullYear()).toBe(2024);
      expect(periodEnd.getMonth()).toBe(5);
      expect(periodEnd.getFullYear()).toBe(2024);
    });

    it('should return empty data for a month with no meals', async () => {
      const response = await request(app)
        .get('/api/reports/monthly?month=2020-01')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.totals.calories).toBe(0);
      expect(response.body.totalDaysTracked).toBe(0);
    });

    it('should return 401 without authentication', async () => {
      await request(app).get('/api/reports/monthly').expect(401);
    });

    it('should return 400 for invalid month format', async () => {
      const response = await request(app)
        .get('/api/reports/monthly?month=invalid')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for month format YYYY-M (single digit month)', async () => {
      const response = await request(app)
        .get('/api/reports/monthly?month=2024-1')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/reports/weekly/export
  // ============================================================================

  describe('GET /api/reports/weekly/export', () => {
    it('should export weekly report as JSON', async () => {
      // Create a test meal
      await createTestMeal(userId, {
        name: 'Test Meal',
        mealType: 'lunch',
        calories: 500,
        protein: 30,
        carbs: 50,
        fat: 20,
      });

      const response = await request(app)
        .get('/api/reports/weekly/export?format=json')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('format', 'json');
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('filename');
      expect(response.body).toHaveProperty('mimeType', 'application/json');

      // Verify the data contains the full report
      expect(response.body.data).toHaveProperty('userId');
      expect(response.body.data).toHaveProperty('totals');
      expect(response.body.data).toHaveProperty('dailyBreakdowns');
    });

    it('should return placeholder for PDF format', async () => {
      const response = await request(app)
        .get('/api/reports/weekly/export?format=pdf')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('format', 'pdf');
      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('mimeType', 'application/pdf');
    });

    it('should return placeholder for image format', async () => {
      const response = await request(app)
        .get('/api/reports/weekly/export?format=image')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('format', 'image');
      expect(response.body).toHaveProperty('message');
      expect(response.body).toHaveProperty('mimeType', 'image/png');
    });

    it('should return 400 for missing format parameter', async () => {
      const response = await request(app)
        .get('/api/reports/weekly/export')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 400 for invalid format', async () => {
      const response = await request(app)
        .get('/api/reports/weekly/export?format=invalid')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 401 without authentication', async () => {
      await request(app).get('/api/reports/weekly/export?format=json').expect(401);
    });
  });

  // ============================================================================
  // GET /api/reports/monthly/export
  // ============================================================================

  describe('GET /api/reports/monthly/export', () => {
    it('should export monthly report as JSON', async () => {
      // Create a test meal
      await createTestMeal(userId, {
        name: 'Test Meal',
        mealType: 'dinner',
        calories: 700,
        protein: 40,
        carbs: 60,
        fat: 25,
      });

      const response = await request(app)
        .get('/api/reports/monthly/export?format=json')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('format', 'json');
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('filename');
      expect(response.body).toHaveProperty('mimeType', 'application/json');

      // Verify the data contains the full report
      expect(response.body.data).toHaveProperty('userId');
      expect(response.body.data).toHaveProperty('month');
      expect(response.body.data).toHaveProperty('weeklyBreakdowns');
    });

    it('should export specific month as JSON', async () => {
      const response = await request(app)
        .get('/api/reports/monthly/export?format=json&month=2024-06')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.data.month).toBe('2024-06');
    });

    it('should return placeholder for PDF format', async () => {
      const response = await request(app)
        .get('/api/reports/monthly/export?format=pdf')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('format', 'pdf');
    });

    it('should return 400 for missing format parameter', async () => {
      const response = await request(app)
        .get('/api/reports/monthly/export')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should return 401 without authentication', async () => {
      await request(app).get('/api/reports/monthly/export?format=json').expect(401);
    });
  });

  // ============================================================================
  // Report Content Tests
  // ============================================================================

  describe('Report Content Accuracy', () => {
    it('should calculate daily totals correctly', async () => {
      const today = new Date();

      // Create multiple meals for today
      await createTestMeal(userId, {
        name: 'Breakfast',
        mealType: 'breakfast',
        calories: 300,
        protein: 20,
        carbs: 30,
        fat: 10,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Lunch',
        mealType: 'lunch',
        calories: 500,
        protein: 30,
        carbs: 50,
        fat: 15,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Dinner',
        mealType: 'dinner',
        calories: 600,
        protein: 35,
        carbs: 60,
        fat: 20,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // Find today's breakdown
      const todayStr = today.toISOString().split('T')[0];
      const todayBreakdown = response.body.dailyBreakdowns.find((d: { date: string }) =>
        d.date.startsWith(todayStr)
      );

      expect(todayBreakdown).toBeDefined();
      expect(todayBreakdown.calories).toBe(1400); // 300 + 500 + 600
      expect(todayBreakdown.protein).toBe(85); // 20 + 30 + 35
      expect(todayBreakdown.carbs).toBe(140); // 30 + 50 + 60
      expect(todayBreakdown.fat).toBe(45); // 10 + 15 + 20
      expect(todayBreakdown.mealCount).toBe(3);
    });

    it('should include top foods ranked by count', async () => {
      const today = new Date();

      // Create multiple meals with the same food name
      for (let i = 0; i < 3; i++) {
        await createTestMeal(userId, {
          name: 'Chicken Breast',
          mealType: 'lunch',
          calories: 200,
          protein: 40,
          carbs: 0,
          fat: 5,
          consumedAt: today,
        });
      }

      await createTestMeal(userId, {
        name: 'Rice Bowl',
        mealType: 'dinner',
        calories: 300,
        protein: 10,
        carbs: 60,
        fat: 5,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.topFoods.length).toBeGreaterThan(0);

      // Chicken Breast should be at the top (count: 3)
      const chickenEntry = response.body.topFoods.find(
        (f: { name: string }) => f.name === 'Chicken Breast'
      );
      expect(chickenEntry).toBeDefined();
      expect(chickenEntry.count).toBe(3);
    });

    it('should calculate meal type breakdown correctly', async () => {
      const today = new Date();

      await createTestMeal(userId, {
        name: 'Oatmeal',
        mealType: 'breakfast',
        calories: 300,
        protein: 10,
        carbs: 50,
        fat: 8,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Salad',
        mealType: 'lunch',
        calories: 400,
        protein: 20,
        carbs: 30,
        fat: 15,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Pasta',
        mealType: 'dinner',
        calories: 600,
        protein: 25,
        carbs: 80,
        fat: 18,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Yogurt',
        mealType: 'snack',
        calories: 150,
        protein: 10,
        carbs: 15,
        fat: 5,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.mealTypeBreakdown).toBeDefined();
      expect(response.body.mealTypeBreakdown.breakfast.count).toBe(1);
      expect(response.body.mealTypeBreakdown.lunch.count).toBe(1);
      expect(response.body.mealTypeBreakdown.dinner.count).toBe(1);
      expect(response.body.mealTypeBreakdown.snack.count).toBe(1);
    });

    it('should calculate macro distribution percentages', async () => {
      const today = new Date();

      // Create a meal with known macro values
      // 100g protein (400 cal) + 200g carbs (800 cal) + 50g fat (450 cal) = 1650 total cal
      await createTestMeal(userId, {
        name: 'Test Meal',
        mealType: 'lunch',
        calories: 1650,
        protein: 100,
        carbs: 200,
        fat: 50,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // Verify macro distribution percentages add up to approximately 100
      const totalPercent =
        response.body.macroDistribution.proteinPercent +
        response.body.macroDistribution.carbsPercent +
        response.body.macroDistribution.fatPercent;

      expect(totalPercent).toBeCloseTo(100, 0);
    });
  });

  // ============================================================================
  // With Health Metrics and Activities
  // ============================================================================

  describe('Report with Health Metrics and Activities', () => {
    it('should include health metrics summary in weekly report', async () => {
      const today = new Date();

      // Create test meal
      await createTestMeal(userId, {
        name: 'Test Meal',
        mealType: 'lunch',
        calories: 500,
        protein: 30,
        carbs: 50,
        fat: 20,
        consumedAt: today,
      });

      // Create test health metrics
      await createTestHealthMetric(userId, {
        metricType: HealthMetricType.RESTING_HEART_RATE,
        value: 62,
        recordedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('healthMetricsSummary');
      expect(Array.isArray(response.body.healthMetricsSummary)).toBe(true);
    });

    it('should include activity summary in weekly report', async () => {
      const today = new Date();

      // Create test meal
      await createTestMeal(userId, {
        name: 'Test Meal',
        mealType: 'lunch',
        calories: 500,
        protein: 30,
        carbs: 50,
        fat: 20,
        consumedAt: today,
      });

      // Create test activity
      await createTestActivity(userId, {
        activityType: ActivityType.RUNNING,
        intensity: ActivityIntensity.MODERATE,
        duration: 30,
        caloriesBurned: 300,
        startedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('activitySummary');
    });
  });

  // ============================================================================
  // Edge Cases
  // ============================================================================

  describe('Edge Cases', () => {
    it('should handle user with no data gracefully', async () => {
      // Create a new user with no meals, activities, or health metrics
      const emptyUser = await createTestUser({ email: 'empty@example.com' });
      const emptyToken = createTestToken(emptyUser.id);

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${emptyToken}`)
        .expect(200);

      expect(response.body.totals.calories).toBe(0);
      expect(response.body.daysGoalsMet).toBe(0);
      expect(response.body.goalCompletionRate).toBe(0);
      expect(response.body.topFoods).toHaveLength(0);
    });

    it('should handle extremely large calorie values', async () => {
      const today = new Date();

      await createTestMeal(userId, {
        name: 'Large Meal',
        mealType: 'dinner',
        calories: 99999,
        protein: 1000,
        carbs: 5000,
        fat: 3000,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/reports/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.totals.calories).toBe(99999);
    });

    it('should handle boundary dates (first/last day of week)', async () => {
      // Test first day of current week (Sunday)
      const today = new Date();
      const dayOfWeek = today.getDay();
      const firstDayOfWeek = new Date(today);
      firstDayOfWeek.setDate(today.getDate() - dayOfWeek);
      const dateStr = firstDayOfWeek.toISOString().split('T')[0];

      const response = await request(app)
        .get(`/api/reports/weekly?date=${dateStr}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.periodStart).toBeDefined();
      expect(response.body.periodEnd).toBeDefined();
    });

    it('should handle boundary dates (first/last day of month)', async () => {
      // Test first day of a month
      const response = await request(app)
        .get('/api/reports/monthly?month=2024-01')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      const periodStart = new Date(response.body.periodStart);
      const periodEnd = new Date(response.body.periodEnd);

      expect(periodStart.getDate()).toBe(1);
      expect(periodEnd.getDate()).toBe(31); // January has 31 days
    });
  });
});
