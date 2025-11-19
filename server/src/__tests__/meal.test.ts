/**
 * Meal Endpoints Tests
 *
 * Tests all meal-related endpoints:
 * - Create meal
 * - Get meals (with date filtering)
 * - Get meal by ID
 * - Update meal
 * - Delete meal
 * - Get daily summary
 * - Get weekly summary
 */

import request from 'supertest';
import app from '../index';
import {
  prisma,
  createTestUser,
  createTestToken,
  createTestMeal,
  assertMealStructure,
} from './setup';

describe('Meal API Endpoints', () => {
  let userId: string;
  let authToken: string;

  // Create a test user and get auth token before each test
  beforeEach(async () => {
    const user = await createTestUser({ email: 'meal-test@example.com' });
    userId = user.id;
    authToken = createTestToken(userId);
  });

  // ============================================================================
  // POST /api/meals
  // ============================================================================

  describe('POST /api/meals', () => {
    it('should create a meal successfully', async () => {
      const mealData = {
        name: 'Chicken Salad',
        mealType: 'lunch',
        calories: 450,
        protein: 35,
        carbs: 30,
        fat: 20,
        fiber: 8,
        sugar: 5,
        servingSize: '1 bowl',
        notes: 'Grilled chicken with mixed greens',
      };

      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(mealData)
        .expect(201);

      assertMealStructure(response.body);
      expect(response.body.name).toBe(mealData.name);
      expect(response.body.mealType).toBe(mealData.mealType);
      expect(response.body.calories).toBe(mealData.calories);
      expect(response.body.protein).toBe(mealData.protein);
      expect(response.body.carbs).toBe(mealData.carbs);
      expect(response.body.fat).toBe(mealData.fat);
      expect(response.body.userId).toBe(userId);
    });

    it('should create meal with minimal required fields', async () => {
      const mealData = {
        name: 'Quick Snack',
        mealType: 'snack',
        calories: 150,
        protein: 5,
        carbs: 20,
        fat: 6,
      };

      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(mealData)
        .expect(201);

      assertMealStructure(response.body);
      expect(response.body.name).toBe(mealData.name);
    });

    it('should create meal with custom consumedAt timestamp', async () => {
      const customDate = new Date('2024-01-15T12:00:00Z');
      const mealData = {
        name: 'Breakfast',
        mealType: 'breakfast',
        calories: 350,
        protein: 20,
        carbs: 40,
        fat: 12,
        consumedAt: customDate.toISOString(),
      };

      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(mealData)
        .expect(201);

      expect(new Date(response.body.consumedAt).getTime()).toBe(customDate.getTime());
    });

    it('should reject meal without authentication', async () => {
      const response = await request(app)
        .post('/api/meals')
        .send({
          name: 'Test Meal',
          mealType: 'dinner',
          calories: 500,
          protein: 30,
          carbs: 50,
          fat: 15,
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject meal with missing required fields', async () => {
      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Incomplete Meal',
          // Missing mealType, calories, protein, carbs, fat
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject meal with negative calories', async () => {
      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Invalid Meal',
          mealType: 'lunch',
          calories: -100,
          protein: 20,
          carbs: 30,
          fat: 10,
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('positive');
    });

    it('should reject meal with invalid mealType', async () => {
      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Invalid Type Meal',
          mealType: 'brunch', // Not a valid enum value
          calories: 400,
          protein: 25,
          carbs: 35,
          fat: 15,
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject meal with empty name', async () => {
      const response = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: '',
          mealType: 'lunch',
          calories: 400,
          protein: 25,
          carbs: 35,
          fat: 15,
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('required');
    });
  });

  // ============================================================================
  // GET /api/meals
  // ============================================================================

  describe('GET /api/meals', () => {
    it('should get all meals for user', async () => {
      // Create test meals
      await createTestMeal(userId, { name: 'Meal 1' });
      await createTestMeal(userId, { name: 'Meal 2' });
      await createTestMeal(userId, { name: 'Meal 3' });

      const response = await request(app)
        .get('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body).toHaveLength(3);
      response.body.forEach((meal: any) => assertMealStructure(meal));
    });

    it('should get meals filtered by date', async () => {
      const targetDate = new Date('2024-01-15');
      const otherDate = new Date('2024-01-20');

      // Create meals on different dates
      await createTestMeal(userId, {
        name: 'Target Date Meal 1',
        consumedAt: targetDate,
      });
      await createTestMeal(userId, {
        name: 'Target Date Meal 2',
        consumedAt: targetDate,
      });
      await createTestMeal(userId, {
        name: 'Other Date Meal',
        consumedAt: otherDate,
      });

      const response = await request(app)
        .get('/api/meals')
        .query({ date: targetDate.toISOString() })
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body).toHaveLength(2);
    });

    it('should return empty array when no meals exist', async () => {
      const response = await request(app)
        .get('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body).toHaveLength(0);
    });

    it('should only return meals for authenticated user', async () => {
      // Create another user with meals
      const otherUser = await createTestUser({ email: 'other@example.com' });
      await createTestMeal(otherUser.id, { name: 'Other User Meal' });

      // Create meals for current user
      await createTestMeal(userId, { name: 'My Meal' });

      const response = await request(app)
        .get('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveLength(1);
      expect(response.body[0].name).toBe('My Meal');
    });

    it('should reject request without authentication', async () => {
      const response = await request(app)
        .get('/api/meals')
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/meals/:id
  // ============================================================================

  describe('GET /api/meals/:id', () => {
    it('should get meal by ID', async () => {
      const meal = await createTestMeal(userId, { name: 'Test Meal' });

      const response = await request(app)
        .get(`/api/meals/${meal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      assertMealStructure(response.body);
      expect(response.body.id).toBe(meal.id);
      expect(response.body.name).toBe('Test Meal');
    });

    it('should reject request for non-existent meal', async () => {
      const response = await request(app)
        .get('/api/meals/non-existent-id')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject request for another user\'s meal', async () => {
      // Create another user with a meal
      const otherUser = await createTestUser({ email: 'other2@example.com' });
      const otherMeal = await createTestMeal(otherUser.id, { name: 'Other Meal' });

      const response = await request(app)
        .get(`/api/meals/${otherMeal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject request without authentication', async () => {
      const meal = await createTestMeal(userId, { name: 'Test Meal' });

      const response = await request(app)
        .get(`/api/meals/${meal.id}`)
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // PUT /api/meals/:id
  // ============================================================================

  describe('PUT /api/meals/:id', () => {
    it('should update meal successfully', async () => {
      const meal = await createTestMeal(userId, {
        name: 'Original Name',
        calories: 400,
      });

      const updates = {
        name: 'Updated Name',
        calories: 500,
        protein: 40,
      };

      const response = await request(app)
        .put(`/api/meals/${meal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(updates)
        .expect(200);

      assertMealStructure(response.body);
      expect(response.body.name).toBe(updates.name);
      expect(response.body.calories).toBe(updates.calories);
      expect(response.body.protein).toBe(updates.protein);
    });

    it('should update only provided fields', async () => {
      const meal = await createTestMeal(userId, {
        name: 'Original Name',
        calories: 400,
        protein: 30,
      });

      const updates = {
        calories: 450, // Only update calories
      };

      const response = await request(app)
        .put(`/api/meals/${meal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(updates)
        .expect(200);

      expect(response.body.name).toBe('Original Name'); // Unchanged
      expect(response.body.calories).toBe(450); // Updated
      expect(response.body.protein).toBe(30); // Unchanged
    });

    it('should reject update with invalid data', async () => {
      const meal = await createTestMeal(userId);

      const response = await request(app)
        .put(`/api/meals/${meal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          calories: -100, // Negative calories
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('positive');
    });

    it('should reject update for non-existent meal', async () => {
      const response = await request(app)
        .put('/api/meals/non-existent-id')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Updated Name',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject update for another user\'s meal', async () => {
      // Create another user with a meal
      const otherUser = await createTestUser({ email: 'other3@example.com' });
      const otherMeal = await createTestMeal(otherUser.id);

      const response = await request(app)
        .put(`/api/meals/${otherMeal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Hacked Name',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject update without authentication', async () => {
      const meal = await createTestMeal(userId);

      const response = await request(app)
        .put(`/api/meals/${meal.id}`)
        .send({
          name: 'Updated Name',
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // DELETE /api/meals/:id
  // ============================================================================

  describe('DELETE /api/meals/:id', () => {
    it('should delete meal successfully', async () => {
      const meal = await createTestMeal(userId, { name: 'To Delete' });

      const response = await request(app)
        .delete(`/api/meals/${meal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('message');
      expect(response.body.message).toContain('deleted');

      // Verify meal is actually deleted
      const deletedMeal = await prisma.meal.findUnique({
        where: { id: meal.id },
      });
      expect(deletedMeal).toBeNull();
    });

    it('should reject delete for non-existent meal', async () => {
      const response = await request(app)
        .delete('/api/meals/non-existent-id')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject delete for another user\'s meal', async () => {
      // Create another user with a meal
      const otherUser = await createTestUser({ email: 'other4@example.com' });
      const otherMeal = await createTestMeal(otherUser.id);

      const response = await request(app)
        .delete(`/api/meals/${otherMeal.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body).toHaveProperty('error');

      // Verify meal still exists
      const meal = await prisma.meal.findUnique({
        where: { id: otherMeal.id },
      });
      expect(meal).not.toBeNull();
    });

    it('should reject delete without authentication', async () => {
      const meal = await createTestMeal(userId);

      const response = await request(app)
        .delete(`/api/meals/${meal.id}`)
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/meals/summary/daily
  // ============================================================================

  describe('GET /api/meals/summary/daily', () => {
    it('should get daily summary for today', async () => {
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      // Create meals for today
      await createTestMeal(userId, {
        name: 'Breakfast',
        calories: 300,
        protein: 20,
        carbs: 40,
        fat: 10,
        consumedAt: today,
      });
      await createTestMeal(userId, {
        name: 'Lunch',
        calories: 500,
        protein: 35,
        carbs: 50,
        fat: 20,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/meals/summary/daily')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('totalCalories', 800);
      expect(response.body).toHaveProperty('totalProtein', 55);
      expect(response.body).toHaveProperty('totalCarbs', 90);
      expect(response.body).toHaveProperty('totalFat', 30);
      expect(response.body).toHaveProperty('mealsCount', 2);
    });

    it('should get daily summary for specific date', async () => {
      const targetDate = new Date('2024-01-15');

      await createTestMeal(userId, {
        name: 'Historic Meal',
        calories: 400,
        protein: 30,
        carbs: 45,
        fat: 15,
        consumedAt: targetDate,
      });

      const response = await request(app)
        .get('/api/meals/summary/daily')
        .query({ date: targetDate.toISOString() })
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('totalCalories', 400);
      expect(response.body).toHaveProperty('mealsCount', 1);
    });

    it('should return zero summary when no meals exist', async () => {
      const response = await request(app)
        .get('/api/meals/summary/daily')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('totalCalories', 0);
      expect(response.body).toHaveProperty('totalProtein', 0);
      expect(response.body).toHaveProperty('mealsCount', 0);
    });

    it('should reject request without authentication', async () => {
      const response = await request(app)
        .get('/api/meals/summary/daily')
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/meals/summary/weekly
  // ============================================================================

  describe('GET /api/meals/summary/weekly', () => {
    it('should get weekly summary', async () => {
      const today = new Date();
      const weekAgo = new Date(today);
      weekAgo.setDate(weekAgo.getDate() - 6);

      // Create meals across the week
      await createTestMeal(userId, {
        name: 'Day 1 Meal',
        calories: 400,
        consumedAt: weekAgo,
      });
      await createTestMeal(userId, {
        name: 'Day 7 Meal',
        calories: 500,
        consumedAt: today,
      });

      const response = await request(app)
        .get('/api/meals/summary/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('totalCalories', 900);
      expect(response.body).toHaveProperty('mealsCount', 2);
      expect(response.body).toHaveProperty('averageDailyCalories');
      expect(response.body.averageDailyCalories).toBeGreaterThan(0);
    });

    it('should return zero summary when no meals exist', async () => {
      const response = await request(app)
        .get('/api/meals/summary/weekly')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('totalCalories', 0);
      expect(response.body).toHaveProperty('mealsCount', 0);
    });

    it('should reject request without authentication', async () => {
      const response = await request(app)
        .get('/api/meals/summary/weekly')
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // Complete CRUD Flow
  // ============================================================================

  describe('Complete CRUD Flow', () => {
    it('should complete full CRUD lifecycle', async () => {
      // 1. CREATE: Create a new meal
      const createResponse = await request(app)
        .post('/api/meals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'CRUD Test Meal',
          mealType: 'lunch',
          calories: 600,
          protein: 40,
          carbs: 60,
          fat: 20,
        })
        .expect(201);

      const mealId = createResponse.body.id;
      expect(createResponse.body.name).toBe('CRUD Test Meal');

      // 2. READ: Get the created meal
      const readResponse = await request(app)
        .get(`/api/meals/${mealId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(readResponse.body.id).toBe(mealId);
      expect(readResponse.body.name).toBe('CRUD Test Meal');

      // 3. UPDATE: Update the meal
      const updateResponse = await request(app)
        .put(`/api/meals/${mealId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Updated CRUD Meal',
          calories: 650,
        })
        .expect(200);

      expect(updateResponse.body.name).toBe('Updated CRUD Meal');
      expect(updateResponse.body.calories).toBe(650);
      expect(updateResponse.body.protein).toBe(40); // Unchanged

      // 4. DELETE: Delete the meal
      await request(app)
        .delete(`/api/meals/${mealId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      // 5. VERIFY: Confirm deletion
      await request(app)
        .get(`/api/meals/${mealId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });
  });
});
