import prisma from '../config/database';
import { logger } from '../config/logger';
import { AchievementCategory, AchievementRequirement, XPSource } from '@prisma/client';

// XP amounts for different actions
const XP_AMOUNTS = {
  MEAL_LOGGED: 10,
  WATER_LOGGED: 5,
  WATER_GOAL_MET: 25,
  EXERCISE_LOGGED: 15,
  WEIGHT_LOGGED: 10,
  STREAK_BONUS_BASE: 5, // Multiplied by streak length
  STREAK_BONUS_MAX: 50, // Cap for streak bonus
  DAILY_BONUS: 20,
  CALORIE_GOAL: 30,
  MACRO_GOAL: 20,
  LEVEL_UP_BONUS: 50,
};

// XP required per level (exponential scaling)
function xpForLevel(level: number): number {
  return Math.floor(100 * Math.pow(1.2, level - 1));
}

// Get ISO week number (1-53) for a date
function getISOWeek(date: Date): number {
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
}

// Get year-week string for comparison (e.g., "2025-W01")
function getYearWeek(date: Date): string {
  return `${date.getUTCFullYear()}-W${String(getISOWeek(date)).padStart(2, '0')}`;
}

// Get year-month string for comparison (e.g., "2025-01")
function getYearMonth(date: Date): string {
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
}

/**
 * Gamification Service
 * Handles streaks, XP, achievements, and leaderboards
 */
class GamificationService {
  // ============================================================================
  // STREAK MANAGEMENT
  // ============================================================================

  async getOrCreateStreak(userId: string) {
    let streak = await prisma.userStreak.findUnique({
      where: { userId },
    });

    if (!streak) {
      streak = await prisma.userStreak.create({
        data: { userId },
      });
    }

    return streak;
  }

  async updateStreak(userId: string, activityType: 'meal' | 'water' | 'exercise') {
    const streak = await this.getOrCreateStreak(userId);
    // Use UTC for consistent streak calculation across all users and servers
    const today = new Date();
    today.setUTCHours(0, 0, 0, 0);

    const lastActivityDate = streak.lastActivityDate ? new Date(streak.lastActivityDate) : null;

    if (lastActivityDate) {
      lastActivityDate.setUTCHours(0, 0, 0, 0);
    }

    // Check if already logged today
    if (lastActivityDate && lastActivityDate.getTime() === today.getTime()) {
      // Update specific streak type
      await this.updateSpecificStreak(userId, activityType);
      return streak;
    }

    // Check if this is a consecutive day
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    let newStreak = streak.currentStreak;
    let streakBroken = false;

    if (!lastActivityDate) {
      // First activity ever
      newStreak = 1;
    } else if (lastActivityDate.getTime() === yesterday.getTime()) {
      // Consecutive day
      newStreak = streak.currentStreak + 1;
    } else if (lastActivityDate.getTime() < yesterday.getTime()) {
      // Streak broken - check if we can use a freeze token
      if (streak.freezeTokens > 0) {
        // Use freeze token
        newStreak = streak.currentStreak + 1;
        await prisma.userStreak.update({
          where: { userId },
          data: {
            freezeTokens: { decrement: 1 },
            freezeUsedAt: new Date(),
          },
        });
        logger.info({ userId }, 'Used streak freeze token');
      } else {
        // Streak broken
        newStreak = 1;
        streakBroken = true;
      }
    }

    const longestStreak = Math.max(newStreak, streak.longestStreak);

    const updatedStreak = await prisma.userStreak.update({
      where: { userId },
      data: {
        currentStreak: newStreak,
        longestStreak,
        lastActivityDate: today,
      },
    });

    // Award streak bonus XP
    if (!streakBroken && newStreak > 1) {
      const streakBonus = Math.min(
        XP_AMOUNTS.STREAK_BONUS_BASE * newStreak,
        XP_AMOUNTS.STREAK_BONUS_MAX
      );
      await this.awardXP(userId, streakBonus, XPSource.STREAK_BONUS, {
        referenceType: 'streak',
        description: `${newStreak}-day streak bonus`,
      });
    }

    // Check streak-related achievements
    await this.checkStreakAchievements(userId, newStreak, longestStreak);

    // Update specific streak type
    await this.updateSpecificStreak(userId, activityType);

    logger.info({ userId, newStreak, longestStreak, streakBroken }, 'Updated user streak');

    return updatedStreak;
  }

  private async updateSpecificStreak(userId: string, activityType: 'meal' | 'water' | 'exercise') {
    const field =
      activityType === 'meal'
        ? 'mealStreak'
        : activityType === 'water'
          ? 'waterStreak'
          : 'exerciseStreak';

    await prisma.userStreak.update({
      where: { userId },
      data: {
        [field]: { increment: 1 },
      },
    });
  }

  async getStreakStats(userId: string) {
    const streak = await this.getOrCreateStreak(userId);

    // Check if streak is still active (using UTC for consistency)
    const today = new Date();
    today.setUTCHours(0, 0, 0, 0);

    const yesterday = new Date(today);
    yesterday.setUTCDate(yesterday.getUTCDate() - 1);

    let isActive = false;
    if (streak.lastActivityDate) {
      const lastDate = new Date(streak.lastActivityDate);
      lastDate.setUTCHours(0, 0, 0, 0);
      isActive =
        lastDate.getTime() === today.getTime() || lastDate.getTime() === yesterday.getTime();
    }

    return {
      currentStreak: isActive ? streak.currentStreak : 0,
      longestStreak: streak.longestStreak,
      mealStreak: streak.mealStreak,
      waterStreak: streak.waterStreak,
      exerciseStreak: streak.exerciseStreak,
      freezeTokens: streak.freezeTokens,
      isActive,
      lastActivityDate: streak.lastActivityDate,
    };
  }

  async addFreezeToken(userId: string, amount = 1) {
    const streak = await prisma.userStreak.update({
      where: { userId },
      data: {
        freezeTokens: { increment: amount },
      },
    });

    logger.info({ userId, amount, total: streak.freezeTokens }, 'Added freeze token');
    return streak;
  }

  // ============================================================================
  // XP AND LEVELING
  // ============================================================================

  async getOrCreateXP(userId: string) {
    let xp = await prisma.userXP.findUnique({
      where: { userId },
    });

    if (!xp) {
      xp = await prisma.userXP.create({
        data: {
          userId,
          xpToNextLevel: xpForLevel(2),
        },
      });
    }

    return xp;
  }

  async awardXP(
    userId: string,
    amount: number,
    source: XPSource,
    options: {
      referenceId?: string;
      referenceType?: string;
      description?: string;
    } = {}
  ) {
    const userXP = await this.getOrCreateXP(userId);

    // Reset weekly/monthly XP if calendar week/month has changed (using UTC)
    const now = new Date();
    const currentYearWeek = getYearWeek(now);
    const currentYearMonth = getYearMonth(now);
    const storedYearWeek = getYearWeek(new Date(userXP.weekStartDate));
    const storedYearMonth = getYearMonth(new Date(userXP.monthStartDate));

    const weeklyReset = currentYearWeek !== storedYearWeek;
    const monthlyReset = currentYearMonth !== storedYearMonth;

    // Update XP
    const newTotalXP = userXP.totalXP + amount;
    let newLevel = userXP.currentLevel;
    let leveledUp = false;

    // Check for level up
    while (newTotalXP >= this.getTotalXPForLevel(newLevel + 1)) {
      newLevel++;
      leveledUp = true;
    }

    const xpToNextLevel = this.getTotalXPForLevel(newLevel + 1) - newTotalXP;

    const updatedXP = await prisma.userXP.update({
      where: { userId },
      data: {
        totalXP: newTotalXP,
        currentLevel: newLevel,
        xpToNextLevel,
        weeklyXP: weeklyReset ? amount : { increment: amount },
        monthlyXP: monthlyReset ? amount : { increment: amount },
        ...(weeklyReset && { weekStartDate: now }),
        ...(monthlyReset && { monthStartDate: now }),
      },
    });

    // Log transaction
    await prisma.xPTransaction.create({
      data: {
        userId,
        amount,
        source,
        description: options.description,
        referenceId: options.referenceId,
        referenceType: options.referenceType,
      },
    });

    // Award level up bonus
    if (leveledUp) {
      await this.onLevelUp(userId, newLevel);
    }

    logger.info(
      {
        userId,
        amount,
        source,
        newTotalXP,
        newLevel,
        leveledUp,
      },
      'Awarded XP'
    );

    return {
      xp: updatedXP,
      leveledUp,
      newLevel: leveledUp ? newLevel : undefined,
    };
  }

  private getTotalXPForLevel(level: number): number {
    let total = 0;
    for (let i = 1; i < level; i++) {
      total += xpForLevel(i);
    }
    return total;
  }

  private async onLevelUp(userId: string, newLevel: number) {
    // Award level up bonus XP
    await prisma.xPTransaction.create({
      data: {
        userId,
        amount: XP_AMOUNTS.LEVEL_UP_BONUS,
        source: XPSource.LEVEL_UP,
        description: `Reached level ${newLevel}`,
        referenceType: 'level',
        referenceId: String(newLevel),
      },
    });

    // Award freeze token every 5 levels
    if (newLevel % 5 === 0) {
      await this.addFreezeToken(userId, 1);
    }

    // Check level-based achievements
    await this.checkLevelAchievements(userId, newLevel);

    logger.info({ userId, newLevel }, 'User leveled up');
  }

  async getXPStats(userId: string) {
    const xp = await this.getOrCreateXP(userId);
    const currentLevelXP = this.getTotalXPForLevel(xp.currentLevel);
    const nextLevelXP = this.getTotalXPForLevel(xp.currentLevel + 1);
    const progressXP = xp.totalXP - currentLevelXP;
    const neededXP = nextLevelXP - currentLevelXP;

    return {
      totalXP: xp.totalXP,
      currentLevel: xp.currentLevel,
      xpToNextLevel: xp.xpToNextLevel,
      progressXP,
      neededXP,
      progressPercent: Math.round((progressXP / neededXP) * 100),
      weeklyXP: xp.weeklyXP,
      monthlyXP: xp.monthlyXP,
    };
  }

  async getXPHistory(userId: string, limit = 20) {
    return prisma.xPTransaction.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
  }

  // ============================================================================
  // ACHIEVEMENTS
  // ============================================================================

  async getAchievements(
    userId: string,
    options: { category?: AchievementCategory; includeHidden?: boolean } = {}
  ) {
    const achievements = await prisma.achievement.findMany({
      where: {
        isActive: true,
        ...(options.category && { category: options.category }),
        ...(!options.includeHidden && { isHidden: false }),
      },
      include: {
        userAchievements: {
          where: { userId },
        },
        parent: true,
      },
      orderBy: [{ category: 'asc' }, { tier: 'asc' }, { targetValue: 'asc' }],
    });

    return achievements.map((achievement) => {
      const userAchievement = achievement.userAchievements[0];
      return {
        ...achievement,
        userAchievements: undefined,
        progress: userAchievement?.currentProgress || 0,
        isCompleted: userAchievement?.isCompleted || false,
        completedAt: userAchievement?.completedAt,
        hasBeenSeen: userAchievement?.hasBeenSeen ?? true,
      };
    });
  }

  async getUnseenAchievements(userId: string) {
    const userAchievements = await prisma.userAchievement.findMany({
      where: {
        userId,
        isCompleted: true,
        hasBeenSeen: false,
      },
      include: {
        achievement: true,
      },
    });

    return userAchievements;
  }

  async markAchievementsSeen(userId: string, achievementIds: string[]) {
    await prisma.userAchievement.updateMany({
      where: {
        userId,
        achievementId: { in: achievementIds },
      },
      data: {
        hasBeenSeen: true,
      },
    });
  }

  async updateAchievementProgress(
    userId: string,
    requirement: AchievementRequirement,
    currentValue: number
  ) {
    // Find all achievements with this requirement
    const achievements = await prisma.achievement.findMany({
      where: {
        requirement,
        isActive: true,
      },
    });

    for (const achievement of achievements) {
      await this.processAchievement(userId, achievement.id, currentValue);
    }
  }

  private async processAchievement(userId: string, achievementId: string, currentValue: number) {
    const achievement = await prisma.achievement.findUnique({
      where: { id: achievementId },
    });

    if (!achievement) return;

    // Get or create user achievement
    let userAchievement = await prisma.userAchievement.findUnique({
      where: {
        userId_achievementId: { userId, achievementId },
      },
    });

    if (!userAchievement) {
      userAchievement = await prisma.userAchievement.create({
        data: {
          userId,
          achievementId,
          currentProgress: currentValue,
        },
      });
    }

    // Already completed
    if (userAchievement.isCompleted) return;

    // Update progress
    const isCompleted = currentValue >= achievement.targetValue;

    await prisma.userAchievement.update({
      where: { id: userAchievement.id },
      data: {
        currentProgress: currentValue,
        isCompleted,
        ...(isCompleted && { completedAt: new Date(), hasBeenSeen: false }),
      },
    });

    // Award XP if completed
    if (isCompleted && achievement.xpReward > 0) {
      await this.awardXP(userId, achievement.xpReward, XPSource.ACHIEVEMENT, {
        referenceId: achievementId,
        referenceType: 'achievement',
        description: `Earned "${achievement.title}" achievement`,
      });

      logger.info({ userId, achievementId, title: achievement.title }, 'Achievement unlocked');
    }
  }

  private async checkStreakAchievements(
    userId: string,
    currentStreak: number,
    longestStreak: number
  ) {
    await this.updateAchievementProgress(
      userId,
      AchievementRequirement.STREAK_DAYS,
      Math.max(currentStreak, longestStreak)
    );
  }

  private async checkLevelAchievements(userId: string, level: number) {
    await this.updateAchievementProgress(userId, AchievementRequirement.LEVEL_REACHED, level);
  }

  // ============================================================================
  // LEADERBOARDS
  // ============================================================================

  async getLeaderboard(type: 'weekly' | 'monthly' | 'allTime', limit = 10) {
    const orderBy =
      type === 'weekly'
        ? { weeklyXP: 'desc' as const }
        : type === 'monthly'
          ? { monthlyXP: 'desc' as const }
          : { totalXP: 'desc' as const };

    const leaderboard = await prisma.userXP.findMany({
      orderBy,
      take: limit,
      select: {
        userId: true,
        totalXP: true,
        currentLevel: true,
        weeklyXP: true,
        monthlyXP: true,
      },
    });

    // Get user names
    const userIds = leaderboard.map((entry) => entry.userId);
    const users = await prisma.user.findMany({
      where: { id: { in: userIds } },
      select: { id: true, name: true },
    });

    const userMap = new Map(users.map((u) => [u.id, u.name]));

    return leaderboard.map((entry, index) => ({
      rank: index + 1,
      userId: entry.userId,
      name: userMap.get(entry.userId) || 'Anonymous',
      xp: type === 'weekly' ? entry.weeklyXP : type === 'monthly' ? entry.monthlyXP : entry.totalXP,
      level: entry.currentLevel,
    }));
  }

  async getUserRank(userId: string, type: 'weekly' | 'monthly' | 'allTime') {
    const userXP = await this.getOrCreateXP(userId);
    const xpValue =
      type === 'weekly' ? userXP.weeklyXP : type === 'monthly' ? userXP.monthlyXP : userXP.totalXP;

    const field = type === 'weekly' ? 'weeklyXP' : type === 'monthly' ? 'monthlyXP' : 'totalXP';

    const rank = await prisma.userXP.count({
      where: {
        [field]: { gt: xpValue },
      },
    });

    return {
      rank: rank + 1,
      xp: xpValue,
      level: userXP.currentLevel,
    };
  }

  // ============================================================================
  // EVENT HANDLERS (Called from other services)
  // ============================================================================

  async onMealLogged(userId: string, mealId: string) {
    // Update streak
    await this.updateStreak(userId, 'meal');

    // Award XP
    await this.awardXP(userId, XP_AMOUNTS.MEAL_LOGGED, XPSource.MEAL_LOGGED, {
      referenceId: mealId,
      referenceType: 'meal',
    });

    // Update meal count achievement
    const mealCount = await prisma.meal.count({ where: { userId } });
    await this.updateAchievementProgress(userId, AchievementRequirement.MEALS_LOGGED, mealCount);
  }

  async onWaterLogged(userId: string, waterIntakeId: string) {
    // Award XP for logging
    await this.awardXP(userId, XP_AMOUNTS.WATER_LOGGED, XPSource.WATER_LOGGED, {
      referenceId: waterIntakeId,
      referenceType: 'water',
    });
  }

  async onWaterGoalMet(userId: string) {
    // Update streak
    await this.updateStreak(userId, 'water');

    // Award XP
    await this.awardXP(userId, XP_AMOUNTS.WATER_GOAL_MET, XPSource.WATER_GOAL_MET);

    // Update water goal achievement
    // Count days where water goal was met
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { goalWater: true },
    });

    if (user) {
      // This is a simplified count - in production, you'd track this more accurately
      const waterGoalsMet = await prisma.userStreak.findUnique({
        where: { userId },
        select: { waterStreak: true },
      });

      await this.updateAchievementProgress(
        userId,
        AchievementRequirement.WATER_GOALS_MET,
        waterGoalsMet?.waterStreak || 0
      );
    }
  }

  async onExerciseLogged(userId: string, activityId: string) {
    // Update streak
    await this.updateStreak(userId, 'exercise');

    // Award XP
    await this.awardXP(userId, XP_AMOUNTS.EXERCISE_LOGGED, XPSource.EXERCISE_LOGGED, {
      referenceId: activityId,
      referenceType: 'activity',
    });

    // Update exercise achievement
    const exerciseCount = await prisma.activity.count({ where: { userId } });
    await this.updateAchievementProgress(
      userId,
      AchievementRequirement.EXERCISES_LOGGED,
      exerciseCount
    );
  }

  async onWeightLogged(userId: string, weightRecordId: string) {
    // Award XP
    await this.awardXP(userId, XP_AMOUNTS.WEIGHT_LOGGED, XPSource.WEIGHT_LOGGED, {
      referenceId: weightRecordId,
      referenceType: 'weight',
    });

    // Update weight logging achievement
    const weightCount = await prisma.weightRecord.count({ where: { userId } });
    await this.updateAchievementProgress(userId, AchievementRequirement.WEIGHT_LOGGED, weightCount);
  }

  async onCalorieGoalMet(userId: string) {
    await this.awardXP(userId, XP_AMOUNTS.CALORIE_GOAL, XPSource.CALORIE_GOAL);

    // Track calorie goal achievements
    const streak = await this.getOrCreateStreak(userId);
    await this.updateAchievementProgress(
      userId,
      AchievementRequirement.CALORIES_UNDER,
      streak.mealStreak // Using meal streak as a proxy
    );
  }

  async onMacroGoalMet(userId: string, macroType: 'protein' | 'carbs' | 'fat') {
    await this.awardXP(userId, XP_AMOUNTS.MACRO_GOAL, XPSource.MACRO_GOAL, {
      referenceType: 'macro',
      description: `Met ${macroType} goal`,
    });

    if (macroType === 'protein') {
      const streak = await this.getOrCreateStreak(userId);
      await this.updateAchievementProgress(
        userId,
        AchievementRequirement.PROTEIN_GOAL,
        streak.mealStreak
      );
    }
  }

  // ============================================================================
  // GAMIFICATION SUMMARY
  // ============================================================================

  async getGamificationSummary(userId: string) {
    const [streakStats, xpStats, recentAchievements, unseenAchievements] = await Promise.all([
      this.getStreakStats(userId),
      this.getXPStats(userId),
      prisma.userAchievement.findMany({
        where: { userId, isCompleted: true },
        include: { achievement: true },
        orderBy: { completedAt: 'desc' },
        take: 5,
      }),
      this.getUnseenAchievements(userId),
    ]);

    return {
      streak: streakStats,
      xp: xpStats,
      recentAchievements: recentAchievements.map((ua) => ({
        id: ua.achievement.id,
        title: ua.achievement.title,
        icon: ua.achievement.icon,
        tier: ua.achievement.tier,
        completedAt: ua.completedAt,
      })),
      unseenCount: unseenAchievements.length,
      hasUnseen: unseenAchievements.length > 0,
    };
  }
}

export const gamificationService = new GamificationService();
