import { mealService } from '../services/mealService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { createMealSchema, updateMealSchema } from '../validation/schemas';
import {
  withErrorHandling,
  ErrorHandlers,
} from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class MealController {
  createMeal = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
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

    res.status(HTTP_STATUS.CREATED).json(meal);
  });

  getMeals = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const meals = await mealService.getMeals(userId, date);

    res.status(HTTP_STATUS.OK).json(meals);
  });

  getMealById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const meal = await mealService.getMealById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(meal);
  });

  updateMeal = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
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

    res.status(HTTP_STATUS.OK).json(meal);
  });

  deleteMeal = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await mealService.deleteMeal(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  getDailySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const summary = await mealService.getDailySummary(userId, date);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  getWeeklySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await mealService.getWeeklySummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });
}

export const mealController = new MealController();
