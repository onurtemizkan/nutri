/**
 * Offline Food Cache with SQLite
 *
 * Provides offline-first food search using SQLite with FTS5
 * Caches top 10K popular foods and user's recent selections
 *
 * NOTE: Requires expo-sqlite to be installed:
 * npx expo install expo-sqlite
 */

import { Platform } from 'react-native';
import { foodsApi } from '../api/foods';
import type { USDAFood } from '../types/foods';

// SQLite types - expo-sqlite must be installed for runtime
// These stubs allow TypeScript to compile without the package
interface SQLiteDatabase {
  execAsync(sql: string): Promise<void>;
  runAsync(sql: string, params?: unknown[]): Promise<unknown>;
  getFirstAsync<T>(sql: string, params?: unknown[]): Promise<T | null>;
  getAllAsync<T>(sql: string, params?: unknown[]): Promise<T[]>;
  withTransactionAsync(fn: () => Promise<void>): Promise<void>;
  closeAsync(): Promise<void>;
}

// Dynamic import for expo-sqlite (lazy loaded)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let SQLiteModule: ExpoSQLiteModule | null = null;
let sqliteLoadError: Error | null = null;

interface ExpoSQLiteModule {
  openDatabaseAsync(name: string): Promise<SQLiteDatabase>;
}

async function loadSQLite(): Promise<ExpoSQLiteModule> {
  if (sqliteLoadError) {
    throw sqliteLoadError;
  }

  if (!SQLiteModule) {
    try {
      // Dynamic require to avoid bundling issues when expo-sqlite isn't installed
      // Install with: npx expo install expo-sqlite
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      SQLiteModule = require('expo-sqlite') as ExpoSQLiteModule;
    } catch (e) {
      sqliteLoadError = new Error(
        'expo-sqlite is not installed. Run: npx expo install expo-sqlite'
      );
      throw sqliteLoadError;
    }
  }
  return SQLiteModule;
}

// Types for cached foods
export interface CachedFood {
  fdcId: number;
  description: string;
  brandOwner?: string;
  dataType: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  servingSize?: number;
  servingSizeUnit?: string;
  lastSync: number;
  source: 'popular' | 'recent';
}

export interface FoodCacheStats {
  popularFoodsCount: number;
  recentFoodsCount: number;
  lastSyncDate: Date | null;
  cacheSize: number; // bytes estimate
}

export interface OfflineSearchResult {
  foods: CachedFood[];
  source: 'cache';
  searchTime: number; // ms
}

// Constants
const DB_NAME = 'food_cache.db';
// Weekly refresh balances data freshness with reduced background sync/network usage
// for largely stable food metadata.
const SYNC_INTERVAL_DAYS = 7;
// Limit popular foods cache to ~10K entries to keep SQLite database small enough
// for mobile storage constraints while covering the vast majority of searches.
const POPULAR_FOODS_LIMIT = 10000;
const SYNC_BATCH_SIZE = 1000;
const RECENT_FOODS_LIMIT = 100;

// Database instance (lazy loaded)
let db: SQLiteDatabase | null = null;

/**
 * Get or create database instance
 */
async function getDatabase(): Promise<SQLiteDatabase> {
  if (db) return db;

  // SQLite not supported on web
  if (Platform.OS === 'web') {
    throw new Error('SQLite is not supported on web platform');
  }

  const SQLite = await loadSQLite();
  db = await SQLite.openDatabaseAsync(DB_NAME);
  return db;
}

/**
 * Initialize the food cache database
 * Creates tables and FTS5 virtual table for full-text search
 */
export async function initFoodCache(): Promise<void> {
  try {
    const database = await getDatabase();

    // Create popular_foods table
    await database.execAsync(`
      CREATE TABLE IF NOT EXISTS popular_foods (
        fdcId INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        brandOwner TEXT,
        dataType TEXT NOT NULL,
        calories REAL NOT NULL,
        protein REAL NOT NULL,
        carbs REAL NOT NULL,
        fat REAL NOT NULL,
        fiber REAL,
        sugar REAL,
        servingSize REAL,
        servingSizeUnit TEXT,
        lastSync INTEGER NOT NULL
      );
    `);

    // Create recent_foods table
    await database.execAsync(`
      CREATE TABLE IF NOT EXISTS recent_foods (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fdcId INTEGER NOT NULL,
        userId TEXT,
        timestamp INTEGER NOT NULL,
        fullData TEXT NOT NULL,
        UNIQUE(fdcId, userId)
      );
    `);

    // Create FTS5 virtual table for fast text search
    await database.execAsync(`
      CREATE VIRTUAL TABLE IF NOT EXISTS food_search_fts USING fts5(
        fdcId,
        description,
        brandOwner,
        content='popular_foods',
        content_rowid='fdcId'
      );
    `);

    // Create triggers to keep FTS in sync with popular_foods
    await database.execAsync(`
      CREATE TRIGGER IF NOT EXISTS popular_foods_ai AFTER INSERT ON popular_foods BEGIN
        INSERT INTO food_search_fts(rowid, fdcId, description, brandOwner)
        VALUES (new.fdcId, new.fdcId, new.description, new.brandOwner);
      END;
    `);

    await database.execAsync(`
      CREATE TRIGGER IF NOT EXISTS popular_foods_ad AFTER DELETE ON popular_foods BEGIN
        INSERT INTO food_search_fts(food_search_fts, rowid, fdcId, description, brandOwner)
        VALUES ('delete', old.fdcId, old.fdcId, old.description, old.brandOwner);
      END;
    `);

    await database.execAsync(`
      CREATE TRIGGER IF NOT EXISTS popular_foods_au AFTER UPDATE ON popular_foods BEGIN
        INSERT INTO food_search_fts(food_search_fts, rowid, fdcId, description, brandOwner)
        VALUES ('delete', old.fdcId, old.fdcId, old.description, old.brandOwner);
        INSERT INTO food_search_fts(rowid, fdcId, description, brandOwner)
        VALUES (new.fdcId, new.fdcId, new.description, new.brandOwner);
      END;
    `);

    // Create sync metadata table
    await database.execAsync(`
      CREATE TABLE IF NOT EXISTS sync_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
    `);

    // Create indexes
    await database.execAsync(`
      CREATE INDEX IF NOT EXISTS idx_popular_foods_dataType ON popular_foods(dataType);
      CREATE INDEX IF NOT EXISTS idx_recent_foods_userId ON recent_foods(userId);
      CREATE INDEX IF NOT EXISTS idx_recent_foods_timestamp ON recent_foods(timestamp DESC);
    `);

    console.log('[FoodCache] Database initialized successfully');
  } catch (error) {
    console.error('[FoodCache] Failed to initialize database:', error);
    throw error;
  }
}

/**
 * Check if sync is needed (>7 days since last sync)
 */
export async function needsSync(): Promise<boolean> {
  try {
    const database = await getDatabase();
    const result = await database.getFirstAsync<{ value: string }>(
      'SELECT value FROM sync_metadata WHERE key = ?',
      ['lastSync']
    );

    if (!result) return true;

    const lastSync = parseInt(result.value, 10);
    const daysSinceSync = (Date.now() - lastSync) / (1000 * 60 * 60 * 24);

    return daysSinceSync > SYNC_INTERVAL_DAYS;
  } catch {
    return true;
  }
}

/**
 * Get the last sync date
 */
export async function getLastSyncDate(): Promise<Date | null> {
  try {
    const database = await getDatabase();
    const result = await database.getFirstAsync<{ value: string }>(
      'SELECT value FROM sync_metadata WHERE key = ?',
      ['lastSync']
    );

    if (!result) return null;
    return new Date(parseInt(result.value, 10));
  } catch {
    return null;
  }
}

/**
 * Sync popular foods from the server
 * Downloads incrementally in batches
 */
export async function syncPopularFoods(
  onProgress?: (progress: number) => void
): Promise<{ synced: number; errors: number }> {
  let synced = 0;
  let errors = 0;

  try {
    const database = await getDatabase();

    // Get total count from server
    const totalPages = Math.ceil(POPULAR_FOODS_LIMIT / SYNC_BATCH_SIZE);

    for (let page = 1; page <= totalPages; page++) {
      try {
        // Fetch batch from server
        const result = await foodsApi.searchFoods({
          query: '*', // Get all foods
          page,
          limit: SYNC_BATCH_SIZE,
          sortBy: 'dataType.keyword', // Prioritize Foundation/SR Legacy
          sortOrder: 'asc',
        });

        if (result.foods.length === 0) break;

        // Insert batch using transaction
        await database.withTransactionAsync(async () => {
          for (const food of result.foods) {
            try {
              await database.runAsync(
                `INSERT OR REPLACE INTO popular_foods
                 (fdcId, description, brandOwner, dataType, calories, protein, carbs, fat, fiber, sugar, servingSize, servingSizeUnit, lastSync)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                  food.fdcId,
                  food.description,
                  food.brandOwner || null,
                  food.dataType,
                  food.calories,
                  food.protein,
                  food.carbs,
                  food.fat,
                  food.fiber || null,
                  food.sugar || null,
                  food.servingSize || null,
                  food.servingSizeUnit || null,
                  Date.now(),
                ]
              );
              synced++;
            } catch (e) {
              errors++;
            }
          }
        });

        // Update progress
        if (onProgress) {
          onProgress(Math.min((page / totalPages) * 100, 100));
        }

        // Check if we've reached the end
        if (!result.pagination.hasNextPage || synced >= POPULAR_FOODS_LIMIT) {
          break;
        }
      } catch (batchError) {
        console.error(`[FoodCache] Batch ${page} sync failed:`, batchError);
        errors += SYNC_BATCH_SIZE;
      }
    }

    // Update last sync timestamp
    await database.runAsync(
      'INSERT OR REPLACE INTO sync_metadata (key, value) VALUES (?, ?)',
      ['lastSync', Date.now().toString()]
    );

    console.log(`[FoodCache] Sync complete: ${synced} synced, ${errors} errors`);
  } catch (error) {
    console.error('[FoodCache] Sync failed:', error);
  }

  return { synced, errors };
}

/**
 * Cache a recently selected food
 */
export async function cacheRecentFood(
  food: USDAFood,
  userId?: string
): Promise<void> {
  try {
    const database = await getDatabase();

    // Insert or update recent food
    await database.runAsync(
      `INSERT OR REPLACE INTO recent_foods (fdcId, userId, timestamp, fullData)
       VALUES (?, ?, ?, ?)`,
      [
        food.fdcId,
        userId || null,
        Date.now(),
        JSON.stringify(food),
      ]
    );

    // Also add to popular_foods if not exists
    await database.runAsync(
      `INSERT OR IGNORE INTO popular_foods
       (fdcId, description, brandOwner, dataType, calories, protein, carbs, fat, fiber, sugar, servingSize, servingSizeUnit, lastSync)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        food.fdcId,
        food.description,
        food.brandOwner || null,
        food.dataType,
        food.calories,
        food.protein,
        food.carbs,
        food.fat,
        food.fiber || null,
        food.sugar || null,
        food.servingSize || null,
        food.servingSizeUnit || null,
        Date.now(),
      ]
    );

    // Trim old recent foods to limit
    await database.runAsync(
      `DELETE FROM recent_foods WHERE id NOT IN (
         SELECT id FROM recent_foods
         WHERE userId IS ? OR userId IS NULL
         ORDER BY timestamp DESC
         LIMIT ?
       )`,
      [userId || null, RECENT_FOODS_LIMIT]
    );
  } catch (error) {
    console.error('[FoodCache] Failed to cache recent food:', error);
  }
}

/**
 * Search offline cache using FTS5
 */
export async function searchOffline(
  query: string,
  limit: number = 25
): Promise<OfflineSearchResult> {
  const startTime = Date.now();

  try {
    const database = await getDatabase();

    // Clean and prepare query for FTS5
    // Security: Removes FTS5 special chars (*, ", -, ^, etc.) via regex.
    // Word-based operators (OR, AND, NOT, NEAR) are neutralized by appending *
    // which makes them prefix searches (e.g., "OR" -> "OR*") rather than operators.
    // The parameterized query with ? prevents SQL injection.
    const cleanQuery = query
      .trim()
      .toLowerCase()
      .replace(/[^\w\s]/g, '') // Remove FTS5 special characters
      .split(/\s+/)
      .filter((word) => word.length >= 2)
      .map((word) => `${word}*`) // Add prefix matching (also neutralizes operators)
      .join(' ');

    if (!cleanQuery) {
      return { foods: [], source: 'cache', searchTime: Date.now() - startTime };
    }

    // Search using FTS5
    const results = await database.getAllAsync<{
      fdcId: number;
      description: string;
      brandOwner: string | null;
      dataType: string;
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
      fiber: number | null;
      sugar: number | null;
      servingSize: number | null;
      servingSizeUnit: string | null;
      lastSync: number;
    }>(
      `SELECT p.* FROM popular_foods p
       JOIN food_search_fts f ON p.fdcId = f.fdcId
       WHERE food_search_fts MATCH ?
       ORDER BY
         CASE p.dataType
           WHEN 'Foundation' THEN 1
           WHEN 'SR Legacy' THEN 2
           WHEN 'Survey (FNDDS)' THEN 3
           WHEN 'Branded' THEN 4
           ELSE 5
         END,
         rank
       LIMIT ?`,
      [cleanQuery, limit]
    );

    const foods: CachedFood[] = results.map((row) => ({
      fdcId: row.fdcId,
      description: row.description,
      brandOwner: row.brandOwner || undefined,
      dataType: row.dataType,
      calories: row.calories,
      protein: row.protein,
      carbs: row.carbs,
      fat: row.fat,
      fiber: row.fiber || undefined,
      sugar: row.sugar || undefined,
      servingSize: row.servingSize || undefined,
      servingSizeUnit: row.servingSizeUnit || undefined,
      lastSync: row.lastSync,
      source: 'popular',
    }));

    return {
      foods,
      source: 'cache',
      searchTime: Date.now() - startTime,
    };
  } catch (error) {
    console.error('[FoodCache] Offline search failed:', error);
    return { foods: [], source: 'cache', searchTime: Date.now() - startTime };
  }
}

/**
 * Get a single cached food by FDC ID
 */
export async function getCachedFood(fdcId: number): Promise<CachedFood | null> {
  try {
    const database = await getDatabase();

    // First check recent foods (has full data)
    const recent = await database.getFirstAsync<{
      fullData: string;
    }>(
      'SELECT fullData FROM recent_foods WHERE fdcId = ?',
      [fdcId]
    );

    if (recent) {
      const food = JSON.parse(recent.fullData) as USDAFood;
      return {
        fdcId: food.fdcId,
        description: food.description,
        brandOwner: food.brandOwner,
        dataType: food.dataType,
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        fiber: food.fiber,
        sugar: food.sugar,
        servingSize: food.servingSize,
        servingSizeUnit: food.servingSizeUnit,
        lastSync: Date.now(),
        source: 'recent',
      };
    }

    // Check popular foods
    const popular = await database.getFirstAsync<{
      fdcId: number;
      description: string;
      brandOwner: string | null;
      dataType: string;
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
      fiber: number | null;
      sugar: number | null;
      servingSize: number | null;
      servingSizeUnit: string | null;
      lastSync: number;
    }>(
      'SELECT * FROM popular_foods WHERE fdcId = ?',
      [fdcId]
    );

    if (popular) {
      return {
        fdcId: popular.fdcId,
        description: popular.description,
        brandOwner: popular.brandOwner || undefined,
        dataType: popular.dataType,
        calories: popular.calories,
        protein: popular.protein,
        carbs: popular.carbs,
        fat: popular.fat,
        fiber: popular.fiber || undefined,
        sugar: popular.sugar || undefined,
        servingSize: popular.servingSize || undefined,
        servingSizeUnit: popular.servingSizeUnit || undefined,
        lastSync: popular.lastSync,
        source: 'popular',
      };
    }

    return null;
  } catch (error) {
    console.error('[FoodCache] Failed to get cached food:', error);
    return null;
  }
}

/**
 * Get recent foods for a user
 */
export async function getRecentFoods(
  userId?: string,
  limit: number = 10
): Promise<CachedFood[]> {
  try {
    const database = await getDatabase();

    const results = await database.getAllAsync<{
      fdcId: number;
      fullData: string;
      timestamp: number;
    }>(
      `SELECT fdcId, fullData, timestamp FROM recent_foods
       WHERE userId IS ? OR userId IS NULL
       ORDER BY timestamp DESC
       LIMIT ?`,
      [userId || null, limit]
    );

    return results.map((row) => {
      const food = JSON.parse(row.fullData) as USDAFood;
      return {
        fdcId: food.fdcId,
        description: food.description,
        brandOwner: food.brandOwner,
        dataType: food.dataType,
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        fiber: food.fiber,
        sugar: food.sugar,
        servingSize: food.servingSize,
        servingSizeUnit: food.servingSizeUnit,
        lastSync: row.timestamp,
        source: 'recent',
      };
    });
  } catch (error) {
    console.error('[FoodCache] Failed to get recent foods:', error);
    return [];
  }
}

/**
 * Get cache statistics
 */
export async function getCacheStats(): Promise<FoodCacheStats> {
  try {
    const database = await getDatabase();

    const popularCount = await database.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM popular_foods'
    );

    const recentCount = await database.getFirstAsync<{ count: number }>(
      'SELECT COUNT(*) as count FROM recent_foods'
    );

    const lastSync = await getLastSyncDate();

    // Estimate size: ~200 bytes per food entry
    const estimatedSize =
      ((popularCount?.count || 0) + (recentCount?.count || 0)) * 200;

    return {
      popularFoodsCount: popularCount?.count || 0,
      recentFoodsCount: recentCount?.count || 0,
      lastSyncDate: lastSync,
      cacheSize: estimatedSize,
    };
  } catch (error) {
    console.error('[FoodCache] Failed to get cache stats:', error);
    return {
      popularFoodsCount: 0,
      recentFoodsCount: 0,
      lastSyncDate: null,
      cacheSize: 0,
    };
  }
}

/**
 * Clear all cached data
 */
export async function clearCache(): Promise<void> {
  try {
    const database = await getDatabase();

    await database.execAsync('DELETE FROM popular_foods');
    await database.execAsync('DELETE FROM recent_foods');
    await database.execAsync('DELETE FROM sync_metadata');
    await database.execAsync("INSERT INTO food_search_fts(food_search_fts) VALUES('rebuild')");

    console.log('[FoodCache] Cache cleared');
  } catch (error) {
    console.error('[FoodCache] Failed to clear cache:', error);
  }
}

/**
 * Close database connection
 */
export async function closeDatabase(): Promise<void> {
  if (db) {
    await db.closeAsync();
    db = null;
  }
}

// Export default object for convenience
export default {
  initFoodCache,
  needsSync,
  getLastSyncDate,
  syncPopularFoods,
  cacheRecentFood,
  searchOffline,
  getCachedFood,
  getRecentFoods,
  getCacheStats,
  clearCache,
  closeDatabase,
};
