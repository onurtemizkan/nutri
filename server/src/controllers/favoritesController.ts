import { favoritesService } from '../services/favoritesService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  createFavoriteMealSchema,
  updateFavoriteMealSchema,
  createMealTemplateSchema,
  updateMealTemplateSchema,
  createQuickAddPresetSchema,
  updateQuickAddPresetSchema,
  getFavoritesQuerySchema,
  getRecentFoodsQuerySchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { z } from 'zod';

// Schema for creating favorite from meal
const createFavoriteFromMealSchema = z.object({
  mealId: z.string().min(1),
  customName: z.string().optional(),
});

// Schema for using a favorite
const useFavoriteSchema = z.object({
  mealType: z.enum(['breakfast', 'lunch', 'dinner', 'snack']).optional(),
});

// Schema for reordering favorites
const reorderFavoritesSchema = z.object({
  items: z.array(
    z.object({
      id: z.string().min(1),
      sortOrder: z.number().int().min(0),
    })
  ),
});

// Schema for using a template
const useTemplateSchema = z.object({
  mealType: z.enum(['breakfast', 'lunch', 'dinner', 'snack']).optional(),
});

// Schema for executing a preset
const executePresetSchema = z.object({
  mealType: z.enum(['breakfast', 'lunch', 'dinner', 'snack']).optional(),
  consumedAt: z.string().datetime().optional(),
});

// Schema for using recent food
const useRecentFoodSchema = z.object({
  mealType: z.enum(['breakfast', 'lunch', 'dinner', 'snack']),
});

export class FavoritesController {
  // ============================================================================
  // FAVORITE MEALS
  // ============================================================================

  createFavorite = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createFavoriteMealSchema.parse(req.body);
    const favorite = await favoritesService.createFavorite(userId, validatedData);

    res.status(HTTP_STATUS.CREATED).json(favorite);
  });

  createFavoriteFromMeal = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { mealId, customName } = createFavoriteFromMealSchema.parse(req.body);
    const favorite = await favoritesService.createFavoriteFromMeal(userId, mealId, customName);

    if (!favorite) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Source meal not found' });
      return;
    }

    res.status(HTTP_STATUS.CREATED).json(favorite);
  });

  getFavorites = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const query = getFavoritesQuerySchema.parse(req.query);
    const favorites = await favoritesService.getFavorites(userId, query);

    res.status(HTTP_STATUS.OK).json(favorites);
  });

  getFavoriteById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const favorite = await favoritesService.getFavoriteById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(favorite);
  });

  updateFavorite = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateFavoriteMealSchema.parse(req.body);
    const favorite = await favoritesService.updateFavorite(userId, req.params.id, validatedData);

    res.status(HTTP_STATUS.OK).json(favorite);
  });

  deleteFavorite = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    await favoritesService.deleteFavorite(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json({ message: 'Favorite deleted successfully' });
  });

  useFavorite = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { mealType } = useFavoriteSchema.parse(req.body);
    const meal = await favoritesService.useFavorite(userId, req.params.id, mealType);

    res.status(HTTP_STATUS.CREATED).json(meal);
  });

  reorderFavorites = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { items } = reorderFavoritesSchema.parse(req.body);
    await favoritesService.reorderFavorites(userId, items);

    res.status(HTTP_STATUS.OK).json({ message: 'Favorites reordered successfully' });
  });

  // ============================================================================
  // MEAL TEMPLATES
  // ============================================================================

  createTemplate = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createMealTemplateSchema.parse(req.body);
    const template = await favoritesService.createTemplate(userId, validatedData);

    res.status(HTTP_STATUS.CREATED).json(template);
  });

  getTemplates = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 50;
    const offset = req.query.offset ? parseInt(req.query.offset as string, 10) : 0;
    const templates = await favoritesService.getTemplates(userId, limit, offset);

    res.status(HTTP_STATUS.OK).json(templates);
  });

  getTemplateById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const template = await favoritesService.getTemplateById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(template);
  });

  updateTemplate = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateMealTemplateSchema.parse(req.body);
    const template = await favoritesService.updateTemplate(userId, req.params.id, validatedData);

    res.status(HTTP_STATUS.OK).json(template);
  });

  deleteTemplate = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    await favoritesService.deleteTemplate(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json({ message: 'Template deleted successfully' });
  });

  useTemplate = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { mealType } = useTemplateSchema.parse(req.body);
    const meal = await favoritesService.useTemplate(userId, req.params.id, mealType);

    res.status(HTTP_STATUS.CREATED).json(meal);
  });

  // ============================================================================
  // QUICK ADD PRESETS
  // ============================================================================

  createPreset = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createQuickAddPresetSchema.parse(req.body);
    const preset = await favoritesService.createPreset(userId, validatedData);

    res.status(HTTP_STATUS.CREATED).json(preset);
  });

  getPresets = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const activeOnly = req.query.activeOnly !== 'false';
    const presets = await favoritesService.getPresets(userId, activeOnly);

    res.status(HTTP_STATUS.OK).json(presets);
  });

  getPresetById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const preset = await favoritesService.getPresetById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(preset);
  });

  updatePreset = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateQuickAddPresetSchema.parse(req.body);
    const preset = await favoritesService.updatePreset(userId, req.params.id, validatedData);

    res.status(HTTP_STATUS.OK).json(preset);
  });

  deletePreset = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    await favoritesService.deletePreset(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json({ message: 'Preset deleted successfully' });
  });

  executePreset = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { mealType, consumedAt } = executePresetSchema.parse(req.body);
    const result = await favoritesService.executePreset(
      userId,
      req.params.id,
      mealType,
      consumedAt ? new Date(consumedAt) : undefined
    );

    res.status(HTTP_STATUS.CREATED).json(result);
  });

  // ============================================================================
  // RECENT FOODS
  // ============================================================================

  getRecentFoods = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { limit } = getRecentFoodsQuerySchema.parse(req.query);
    const recentFoods = await favoritesService.getRecentFoods(userId, limit);

    res.status(HTTP_STATUS.OK).json(recentFoods);
  });

  useRecentFood = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { mealType } = useRecentFoodSchema.parse(req.body);
    const meal = await favoritesService.useRecentFood(userId, req.params.id, mealType);

    res.status(HTTP_STATUS.CREATED).json(meal);
  });

  clearRecentFoods = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    await favoritesService.clearRecentFoods(userId);

    res.status(HTTP_STATUS.OK).json({ message: 'Recent foods cleared successfully' });
  });
}

export const favoritesController = new FavoritesController();
