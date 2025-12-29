import prisma from '../config/database';
import { logger } from '../config/logger';
import {
  CreateFavoriteMeal,
  UpdateFavoriteMeal,
  CreateMealTemplate,
  UpdateMealTemplate,
  MealTemplateItem,
  CreateQuickAddPreset,
  UpdateQuickAddPreset,
  GetFavoritesQuery,
} from '../validation/schemas';
import { FavoriteSourceType, QuickAddPresetType, Prisma } from '@prisma/client';

// Maximum number of recent foods to keep per user
const MAX_RECENT_FOODS = 50;

/**
 * Favorites Service
 * Handles favorite meals, meal templates, quick-add presets, and recent foods
 */
class FavoritesService {
  // ============================================================================
  // FAVORITE MEALS
  // ============================================================================

  async createFavorite(userId: string, data: CreateFavoriteMeal) {
    const favorite = await prisma.favoriteMeal.create({
      data: {
        userId,
        name: data.name,
        mealType: data.mealType,
        calories: data.calories,
        protein: data.protein,
        carbs: data.carbs,
        fat: data.fat,
        fiber: data.fiber,
        sugar: data.sugar,
        sodium: data.sodium,
        servingSize: data.servingSize,
        sourceType: (data.sourceType as FavoriteSourceType) || 'CUSTOM',
        sourceMealId: data.sourceMealId,
        fdcId: data.fdcId,
        barcode: data.barcode,
        customName: data.customName,
        notes: data.notes,
        imageUrl: data.imageUrl,
        sortOrder: data.sortOrder ?? 0,
      },
    });

    logger.info({ userId, favoriteId: favorite.id }, 'Created favorite meal');
    return favorite;
  }

  async createFavoriteFromMeal(userId: string, mealId: string, customName?: string) {
    // Find the meal
    const meal = await prisma.meal.findFirst({
      where: { id: mealId, userId },
    });

    if (!meal) {
      return null; // Let controller handle not found
    }

    // Create favorite from meal data
    const favorite = await prisma.favoriteMeal.create({
      data: {
        userId,
        name: customName || meal.name,
        mealType: meal.mealType,
        calories: meal.calories,
        protein: meal.protein,
        carbs: meal.carbs,
        fat: meal.fat,
        fiber: meal.fiber,
        sugar: meal.sugar,
        sodium: meal.sodium,
        servingSize: meal.servingSize,
        sourceType: 'MEAL',
        sourceMealId: meal.id,
        customName: customName,
        imageUrl: meal.imageUrl,
      },
    });

    logger.info({ userId, favoriteId: favorite.id, mealId }, 'Created favorite from meal');
    return favorite;
  }

  async getFavorites(userId: string, query: GetFavoritesQuery) {
    const { limit, offset, sortBy, sortOrder } = query;

    // Build orderBy clause
    const orderBy: Prisma.FavoriteMealOrderByWithRelationInput = {};
    const sortField = sortBy || 'usageCount';
    orderBy[sortField as keyof Prisma.FavoriteMealOrderByWithRelationInput] = sortOrder || 'desc';

    const [favorites, total] = await Promise.all([
      prisma.favoriteMeal.findMany({
        where: { userId },
        orderBy,
        skip: offset,
        take: limit,
      }),
      prisma.favoriteMeal.count({ where: { userId } }),
    ]);

    return {
      data: favorites,
      total,
      limit,
      offset,
      hasMore: offset + favorites.length < total,
    };
  }

  async getFavoriteById(userId: string, favoriteId: string) {
    return prisma.favoriteMeal.findFirst({
      where: { id: favoriteId, userId },
    });
  }

  async updateFavorite(userId: string, favoriteId: string, data: UpdateFavoriteMeal) {
    const favorite = await prisma.favoriteMeal.findFirst({
      where: { id: favoriteId, userId },
    });

    if (!favorite) {
      throw new Error('Favorite not found');
    }

    const updated = await prisma.favoriteMeal.update({
      where: { id: favoriteId },
      data: {
        ...data,
        sourceType: data.sourceType as FavoriteSourceType | undefined,
      },
    });

    logger.info({ userId, favoriteId }, 'Updated favorite meal');
    return updated;
  }

  async deleteFavorite(userId: string, favoriteId: string) {
    const favorite = await prisma.favoriteMeal.findFirst({
      where: { id: favoriteId, userId },
    });

    if (!favorite) {
      throw new Error('Favorite not found');
    }

    await prisma.favoriteMeal.delete({
      where: { id: favoriteId },
    });

    logger.info({ userId, favoriteId }, 'Deleted favorite meal');
  }

  async useFavorite(userId: string, favoriteId: string, mealType?: string) {
    // Get the favorite
    const favorite = await prisma.favoriteMeal.findFirst({
      where: { id: favoriteId, userId },
    });

    if (!favorite) {
      throw new Error('Favorite not found');
    }

    // Create a meal from the favorite
    const meal = await prisma.meal.create({
      data: {
        userId,
        name: favorite.customName || favorite.name,
        mealType: mealType || favorite.mealType || 'snack',
        calories: favorite.calories,
        protein: favorite.protein,
        carbs: favorite.carbs,
        fat: favorite.fat,
        fiber: favorite.fiber,
        sugar: favorite.sugar,
        sodium: favorite.sodium,
        servingSize: favorite.servingSize,
        imageUrl: favorite.imageUrl,
      },
    });

    // Update usage stats
    await prisma.favoriteMeal.update({
      where: { id: favoriteId },
      data: {
        usageCount: { increment: 1 },
        lastUsedAt: new Date(),
      },
    });

    // Also add to recent foods
    await this.addRecentFood(userId, {
      name: favorite.name,
      calories: favorite.calories,
      protein: favorite.protein,
      carbs: favorite.carbs,
      fat: favorite.fat,
      fiber: favorite.fiber,
      sugar: favorite.sugar,
      servingSize: favorite.servingSize,
      fdcId: favorite.fdcId,
      barcode: favorite.barcode,
    });

    logger.info({ userId, favoriteId, mealId: meal.id }, 'Used favorite meal');
    return meal;
  }

  async reorderFavorites(userId: string, items: Array<{ id: string; sortOrder: number }>) {
    // Use transaction for atomicity - ensures all updates succeed or none do
    await prisma.$transaction(
      items.map((item) =>
        prisma.favoriteMeal.updateMany({
          where: { id: item.id, userId },
          data: { sortOrder: item.sortOrder },
        })
      )
    );

    logger.info({ userId, itemCount: items.length }, 'Reordered favorites');
  }

  // ============================================================================
  // MEAL TEMPLATES
  // ============================================================================

  async createTemplate(userId: string, data: CreateMealTemplate) {
    // Calculate total nutritional values
    const totals = this.calculateTemplateTotals(data.items);

    const template = await prisma.mealTemplate.create({
      data: {
        userId,
        name: data.name,
        description: data.description,
        mealType: data.mealType,
        imageUrl: data.imageUrl,
        items: data.items as Prisma.InputJsonValue,
        totalCalories: totals.calories,
        totalProtein: totals.protein,
        totalCarbs: totals.carbs,
        totalFat: totals.fat,
        totalFiber: totals.fiber,
        sortOrder: data.sortOrder ?? 0,
      },
    });

    logger.info({ userId, templateId: template.id }, 'Created meal template');
    return template;
  }

  private calculateTemplateTotals(items: MealTemplateItem[]) {
    return items.reduce(
      (acc, item) => {
        const quantity = item.quantity || 1;
        return {
          calories: acc.calories + item.calories * quantity,
          protein: acc.protein + item.protein * quantity,
          carbs: acc.carbs + item.carbs * quantity,
          fat: acc.fat + item.fat * quantity,
          fiber: acc.fiber + (item.fiber || 0) * quantity,
        };
      },
      { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0 }
    );
  }

  async getTemplates(userId: string, limit = 50, offset = 0) {
    const [templates, total] = await Promise.all([
      prisma.mealTemplate.findMany({
        where: { userId },
        orderBy: { usageCount: 'desc' },
        skip: offset,
        take: limit,
      }),
      prisma.mealTemplate.count({ where: { userId } }),
    ]);

    return {
      data: templates,
      total,
      limit,
      offset,
      hasMore: offset + templates.length < total,
    };
  }

  async getTemplateById(userId: string, templateId: string) {
    return prisma.mealTemplate.findFirst({
      where: { id: templateId, userId },
    });
  }

  async updateTemplate(userId: string, templateId: string, data: UpdateMealTemplate) {
    const template = await prisma.mealTemplate.findFirst({
      where: { id: templateId, userId },
    });

    if (!template) {
      throw new Error('Template not found');
    }

    // Recalculate totals if items changed
    const totals = data.items ? this.calculateTemplateTotals(data.items) : null;

    const updated = await prisma.mealTemplate.update({
      where: { id: templateId },
      data: {
        ...data,
        items: data.items as Prisma.InputJsonValue | undefined,
        ...(totals && {
          totalCalories: totals.calories,
          totalProtein: totals.protein,
          totalCarbs: totals.carbs,
          totalFat: totals.fat,
          totalFiber: totals.fiber,
        }),
      },
    });

    logger.info({ userId, templateId }, 'Updated meal template');
    return updated;
  }

  async deleteTemplate(userId: string, templateId: string) {
    const template = await prisma.mealTemplate.findFirst({
      where: { id: templateId, userId },
    });

    if (!template) {
      throw new Error('Template not found');
    }

    await prisma.mealTemplate.delete({
      where: { id: templateId },
    });

    logger.info({ userId, templateId }, 'Deleted meal template');
  }

  async useTemplate(userId: string, templateId: string, mealType?: string) {
    const template = await prisma.mealTemplate.findFirst({
      where: { id: templateId, userId },
    });

    if (!template) {
      throw new Error('Template not found');
    }

    // Create a single meal from the template
    const meal = await prisma.meal.create({
      data: {
        userId,
        name: template.name,
        mealType: mealType || template.mealType || 'snack',
        calories: template.totalCalories,
        protein: template.totalProtein,
        carbs: template.totalCarbs,
        fat: template.totalFat,
        fiber: template.totalFiber,
        notes: template.description,
        imageUrl: template.imageUrl,
      },
    });

    // Update usage stats
    await prisma.mealTemplate.update({
      where: { id: templateId },
      data: {
        usageCount: { increment: 1 },
        lastUsedAt: new Date(),
      },
    });

    logger.info({ userId, templateId, mealId: meal.id }, 'Used meal template');
    return meal;
  }

  // ============================================================================
  // QUICK ADD PRESETS
  // ============================================================================

  async createPreset(userId: string, data: CreateQuickAddPreset) {
    const preset = await prisma.quickAddPreset.create({
      data: {
        userId,
        name: data.name,
        icon: data.icon,
        color: data.color,
        presetType: data.presetType as QuickAddPresetType,
        mealName: data.mealName,
        mealType: data.mealType,
        calories: data.calories,
        protein: data.protein,
        carbs: data.carbs,
        fat: data.fat,
        fiber: data.fiber,
        servingSize: data.servingSize,
        waterAmount: data.waterAmount,
        sortOrder: data.sortOrder ?? 0,
        isActive: data.isActive ?? true,
      },
    });

    logger.info({ userId, presetId: preset.id }, 'Created quick add preset');
    return preset;
  }

  async getPresets(userId: string, activeOnly = true) {
    return prisma.quickAddPreset.findMany({
      where: {
        userId,
        ...(activeOnly && { isActive: true }),
      },
      orderBy: [{ sortOrder: 'asc' }, { usageCount: 'desc' }],
    });
  }

  async getPresetById(userId: string, presetId: string) {
    return prisma.quickAddPreset.findFirst({
      where: { id: presetId, userId },
    });
  }

  async updatePreset(userId: string, presetId: string, data: UpdateQuickAddPreset) {
    const preset = await prisma.quickAddPreset.findFirst({
      where: { id: presetId, userId },
    });

    if (!preset) {
      throw new Error('Preset not found');
    }

    const updated = await prisma.quickAddPreset.update({
      where: { id: presetId },
      data: {
        ...data,
        presetType: data.presetType as QuickAddPresetType | undefined,
      },
    });

    logger.info({ userId, presetId }, 'Updated quick add preset');
    return updated;
  }

  async deletePreset(userId: string, presetId: string) {
    const preset = await prisma.quickAddPreset.findFirst({
      where: { id: presetId, userId },
    });

    if (!preset) {
      throw new Error('Preset not found');
    }

    await prisma.quickAddPreset.delete({
      where: { id: presetId },
    });

    logger.info({ userId, presetId }, 'Deleted quick add preset');
  }

  async executePreset(userId: string, presetId: string, mealType?: string, consumedAt?: Date) {
    const preset = await prisma.quickAddPreset.findFirst({
      where: { id: presetId, userId },
    });

    if (!preset) {
      throw new Error('Preset not found');
    }

    let result: { type: string; data: unknown };

    if (preset.presetType === 'MEAL' && preset.mealName && preset.calories !== null) {
      // Create a meal
      const meal = await prisma.meal.create({
        data: {
          userId,
          name: preset.mealName,
          mealType: mealType || preset.mealType || 'snack',
          calories: preset.calories,
          protein: preset.protein || 0,
          carbs: preset.carbs || 0,
          fat: preset.fat || 0,
          fiber: preset.fiber,
          servingSize: preset.servingSize,
          consumedAt: consumedAt || new Date(),
        },
      });
      result = { type: 'meal', data: meal };
    } else if (preset.presetType === 'WATER' && preset.waterAmount) {
      // Create water intake
      const water = await prisma.waterIntake.create({
        data: {
          userId,
          amount: preset.waterAmount,
          recordedAt: consumedAt || new Date(),
        },
      });
      result = { type: 'water', data: water };
    } else {
      throw new Error('Invalid preset configuration');
    }

    // Update usage stats
    await prisma.quickAddPreset.update({
      where: { id: presetId },
      data: {
        usageCount: { increment: 1 },
        lastUsedAt: new Date(),
      },
    });

    logger.info({ userId, presetId, type: result.type }, 'Executed quick add preset');
    return result;
  }

  // ============================================================================
  // RECENT FOODS
  // ============================================================================

  async addRecentFood(
    userId: string,
    food: {
      name: string;
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
      fiber?: number | null;
      sugar?: number | null;
      servingSize?: string | null;
      fdcId?: number | null;
      barcode?: string | null;
    }
  ) {
    // Upsert recent food entry
    await prisma.recentFood.upsert({
      where: {
        userId_name: { userId, name: food.name },
      },
      update: {
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        fiber: food.fiber,
        sugar: food.sugar,
        servingSize: food.servingSize,
        fdcId: food.fdcId,
        barcode: food.barcode,
        logCount: { increment: 1 },
        lastLoggedAt: new Date(),
      },
      create: {
        userId,
        name: food.name,
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        fiber: food.fiber,
        sugar: food.sugar,
        servingSize: food.servingSize,
        fdcId: food.fdcId,
        barcode: food.barcode,
      },
    });

    // Prune old entries if exceeding limit
    await this.pruneRecentFoods(userId);
  }

  private async pruneRecentFoods(userId: string) {
    const count = await prisma.recentFood.count({ where: { userId } });

    if (count > MAX_RECENT_FOODS) {
      // Get IDs of entries to delete (oldest by lastLoggedAt)
      const toDelete = await prisma.recentFood.findMany({
        where: { userId },
        orderBy: { lastLoggedAt: 'asc' },
        take: count - MAX_RECENT_FOODS,
        select: { id: true },
      });

      await prisma.recentFood.deleteMany({
        where: { id: { in: toDelete.map((r) => r.id) } },
      });
    }
  }

  async getRecentFoods(userId: string, limit = 20) {
    return prisma.recentFood.findMany({
      where: { userId },
      orderBy: { lastLoggedAt: 'desc' },
      take: limit,
    });
  }

  async useRecentFood(userId: string, recentFoodId: string, mealType: string) {
    const recentFood = await prisma.recentFood.findFirst({
      where: { id: recentFoodId, userId },
    });

    if (!recentFood) {
      throw new Error('Recent food not found');
    }

    // Create a meal from the recent food
    const meal = await prisma.meal.create({
      data: {
        userId,
        name: recentFood.name,
        mealType,
        calories: recentFood.calories,
        protein: recentFood.protein,
        carbs: recentFood.carbs,
        fat: recentFood.fat,
        fiber: recentFood.fiber,
        sugar: recentFood.sugar,
        servingSize: recentFood.servingSize,
      },
    });

    // Update usage stats
    await prisma.recentFood.update({
      where: { id: recentFoodId },
      data: {
        logCount: { increment: 1 },
        lastLoggedAt: new Date(),
      },
    });

    logger.info({ userId, recentFoodId, mealId: meal.id }, 'Used recent food');
    return meal;
  }

  async clearRecentFoods(userId: string) {
    await prisma.recentFood.deleteMany({
      where: { userId },
    });

    logger.info({ userId }, 'Cleared recent foods');
  }
}

export const favoritesService = new FavoritesService();
