import { Response } from 'express';
import { z } from 'zod';
import { mealService } from '../services/mealService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { createMealSchema, updateMealSchema } from '../validation/schemas';

export class MealController {
  async createMeal(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = createMealSchema.parse(req.body);

      const mealData = {
        ...validatedData,
        consumedAt: validatedData.consumedAt
          ? new Date(validatedData.consumedAt)
          : undefined,
      };

      const meal = await mealService.createMeal(userId, mealData);

      res.status(201).json(meal);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors[0].message });
        return;
      }

      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getMeals(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const date = req.query.date ? new Date(req.query.date as string) : undefined;

      const meals = await mealService.getMeals(userId, date);

      res.status(200).json(meals);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getMealById(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const meal = await mealService.getMealById(userId, req.params.id);

      res.status(200).json(meal);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async updateMeal(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = updateMealSchema.parse(req.body);

      const mealData = {
        ...validatedData,
        consumedAt: validatedData.consumedAt
          ? new Date(validatedData.consumedAt)
          : undefined,
      };

      const meal = await mealService.updateMeal(userId, req.params.id, mealData);

      res.status(200).json(meal);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors[0].message });
        return;
      }

      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async deleteMeal(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const result = await mealService.deleteMeal(userId, req.params.id);

      res.status(200).json(result);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getDailySummary(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const date = req.query.date ? new Date(req.query.date as string) : undefined;

      const summary = await mealService.getDailySummary(userId, date);

      res.status(200).json(summary);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getWeeklySummary(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const summary = await mealService.getWeeklySummary(userId);

      res.status(200).json(summary);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }
}

export const mealController = new MealController();
