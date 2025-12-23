/**
 * useFoodSearch Hook
 *
 * Provides offline-first food search with seamless online/offline integration
 * Searches local SQLite cache first, then appends online results
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { Platform } from 'react-native';
import { foodsApi } from '@/lib/api/foods';
import {
  searchOffline,
  cacheRecentFood,
  initFoodCache,
  needsSync,
  syncPopularFoods,
  getRecentFoods,
  getCacheStats,
  type CachedFood,
  type FoodCacheStats,
} from '@/lib/storage/foodCache';
import type { USDAFood, FoodSearchOptions, FoodFilterTab, DATA_TYPE_FILTERS } from '@/lib/types/foods';

// Search result with source indication
export interface FoodSearchResult {
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
  source: 'cache' | 'online';
}

export interface UseFoodSearchOptions {
  debounceMs?: number;
  defaultLimit?: number;
  enableOffline?: boolean;
}

export interface UseFoodSearchReturn {
  // Search state
  query: string;
  setQuery: (query: string) => void;
  results: FoodSearchResult[];
  isSearching: boolean;
  isSearchingOnline: boolean;
  error: string | null;

  // Pagination
  page: number;
  totalResults: number;
  hasMore: boolean;
  loadMore: () => void;

  // Filter
  filter: FoodFilterTab;
  setFilter: (filter: FoodFilterTab) => void;

  // Actions
  search: (query: string) => Promise<void>;
  selectFood: (food: FoodSearchResult) => Promise<void>;
  getRecent: () => Promise<FoodSearchResult[]>;
  clearResults: () => void;

  // Cache status
  cacheStats: FoodCacheStats | null;
  isCacheReady: boolean;
  isSyncing: boolean;
  syncProgress: number;
}

const DEFAULT_DEBOUNCE = 300;
const DEFAULT_LIMIT = 25;

export function useFoodSearch(
  options: UseFoodSearchOptions = {}
): UseFoodSearchReturn {
  const {
    debounceMs = DEFAULT_DEBOUNCE,
    defaultLimit = DEFAULT_LIMIT,
    enableOffline = Platform.OS !== 'web',
  } = options;

  // State
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<FoodFilterTab>('all');
  const [results, setResults] = useState<FoodSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isSearchingOnline, setIsSearchingOnline] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalResults, setTotalResults] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  // Cache state
  const [cacheStats, setCacheStats] = useState<FoodCacheStats | null>(null);
  const [isCacheReady, setIsCacheReady] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncProgress, setSyncProgress] = useState(0);

  // Refs
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastQueryRef = useRef<string>('');
  const lastFilterRef = useRef<FoodFilterTab>('all');
  const searchIdRef = useRef(0);
  const resultsRef = useRef<FoodSearchResult[]>([]);

  // Initialize cache on mount
  useEffect(() => {
    if (!enableOffline) {
      setIsCacheReady(true);
      return;
    }

    async function init() {
      try {
        await initFoodCache();
        setIsCacheReady(true);

        // Get cache stats
        const stats = await getCacheStats();
        setCacheStats(stats);

        // Check if sync is needed
        const shouldSync = await needsSync();
        if (shouldSync && stats.popularFoodsCount < 1000) {
          // Only auto-sync if cache is mostly empty
          setIsSyncing(true);
          await syncPopularFoods((progress) => setSyncProgress(progress));
          setIsSyncing(false);

          // Update stats after sync
          const newStats = await getCacheStats();
          setCacheStats(newStats);
        }
      } catch (err) {
        console.error('[useFoodSearch] Cache init failed:', err);
        // Still mark as ready - will work online-only
        setIsCacheReady(true);
      }
    }

    init();
  }, [enableOffline]);

  // Keep resultsRef synchronized with results state
  useEffect(() => {
    resultsRef.current = results;
  }, [results]);

  // Convert cached food to search result
  const cachedToResult = useCallback((food: CachedFood): FoodSearchResult => ({
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
    source: 'cache',
  }), []);

  // Convert USDA food to search result
  const usdaToResult = useCallback((food: USDAFood): FoodSearchResult => ({
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
    source: 'online',
  }), []);

  // Main search function
  const search = useCallback(async (searchQuery: string) => {
    const searchId = ++searchIdRef.current;
    const trimmedQuery = searchQuery.trim();

    if (!trimmedQuery) {
      setResults([]);
      setTotalResults(0);
      setHasMore(false);
      setError(null);
      return;
    }

    setIsSearching(true);
    setError(null);

    // Step 1: Search offline cache first (instant results)
    if (enableOffline && isCacheReady) {
      try {
        const offlineResults = await searchOffline(trimmedQuery, defaultLimit);

        // Check if this search is still current
        if (searchId !== searchIdRef.current) return;

        if (offlineResults.foods.length > 0) {
          setResults(offlineResults.foods.map(cachedToResult));
          setTotalResults(offlineResults.foods.length);
        }
      } catch (err) {
        console.error('[useFoodSearch] Offline search failed:', err);
      }
    }

    // Step 2: Search online (appends/updates results)
    setIsSearchingOnline(true);

    try {
      // Import DATA_TYPE_FILTERS dynamically to avoid circular dependencies
      const { DATA_TYPE_FILTERS } = await import('@/lib/types/foods');

      const onlineResults = await foodsApi.searchFoods({
        query: trimmedQuery,
        page: 1,
        limit: defaultLimit,
        dataType: DATA_TYPE_FILTERS[filter],
      });

      // Check if this search is still current
      if (searchId !== searchIdRef.current) return;

      // Merge results: keep cached results, add online results that aren't duplicates
      // Use ref to avoid stale closure and unnecessary re-renders
      const currentResults = resultsRef.current;
      const cachedIds = new Set(currentResults.map((r) => r.fdcId));
      const onlineFoods = onlineResults.foods.map(usdaToResult);
      const newOnlineFoods = onlineFoods.filter((f) => !cachedIds.has(f.fdcId));

      // Replace cached with online versions (more complete data)
      const mergedResults = [
        ...onlineFoods.slice(0, defaultLimit), // Prefer online versions
        ...currentResults.filter((r) => r.source === 'cache' && !onlineFoods.find((o) => o.fdcId === r.fdcId)),
      ].slice(0, defaultLimit);

      setResults(mergedResults);
      setTotalResults(onlineResults.pagination.total);
      setHasMore(onlineResults.pagination.hasNextPage);
      setPage(1);
    } catch (err) {
      console.error('[useFoodSearch] Online search failed:', err);
      // Keep cached results if online fails
      if (resultsRef.current.length === 0) {
        setError('Search failed. Please check your connection.');
      }
    } finally {
      if (searchId === searchIdRef.current) {
        setIsSearching(false);
        setIsSearchingOnline(false);
      }
    }
  }, [enableOffline, isCacheReady, filter, defaultLimit, cachedToResult, usdaToResult]);

  // Debounced search
  const debouncedSearch = useCallback(
    (searchQuery: string) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }

      debounceRef.current = setTimeout(() => {
        if (searchQuery !== lastQueryRef.current || filter !== lastFilterRef.current) {
          lastQueryRef.current = searchQuery;
          lastFilterRef.current = filter;
          search(searchQuery);
        }
      }, debounceMs);
    },
    [search, filter, debounceMs]
  );

  // Handle query change
  const handleQueryChange = useCallback(
    (newQuery: string) => {
      setQuery(newQuery);
      debouncedSearch(newQuery);
    },
    [debouncedSearch]
  );

  // Handle filter change
  const handleFilterChange = useCallback(
    (newFilter: FoodFilterTab) => {
      setFilter(newFilter);
      if (query.trim()) {
        search(query);
      }
    },
    [query, search]
  );

  // Load more results
  const loadMore = useCallback(async () => {
    if (isSearchingOnline || !hasMore || !query.trim()) return;

    setIsSearchingOnline(true);

    try {
      const { DATA_TYPE_FILTERS } = await import('@/lib/types/foods');

      const onlineResults = await foodsApi.searchFoods({
        query: query.trim(),
        page: page + 1,
        limit: defaultLimit,
        dataType: DATA_TYPE_FILTERS[filter],
      });

      const newFoods = onlineResults.foods.map(usdaToResult);
      const existingIds = new Set(results.map((r) => r.fdcId));
      const uniqueNewFoods = newFoods.filter((f) => !existingIds.has(f.fdcId));

      setResults((prev) => [...prev, ...uniqueNewFoods]);
      setHasMore(onlineResults.pagination.hasNextPage);
      setPage((p) => p + 1);
    } catch (err) {
      console.error('[useFoodSearch] Load more failed:', err);
    } finally {
      setIsSearchingOnline(false);
    }
  }, [isSearchingOnline, hasMore, query, page, filter, defaultLimit, results, usdaToResult]);

  // Select a food (caches it for offline access)
  const selectFood = useCallback(async (food: FoodSearchResult) => {
    if (!enableOffline) return;

    try {
      // Cache the selected food
      await cacheRecentFood({
        fdcId: food.fdcId,
        description: food.description,
        brandOwner: food.brandOwner,
        dataType: food.dataType as USDAFood['dataType'],
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        fiber: food.fiber,
        sugar: food.sugar,
        servingSize: food.servingSize,
        servingSizeUnit: food.servingSizeUnit,
      });
    } catch (err) {
      console.error('[useFoodSearch] Failed to cache selection:', err);
    }
  }, [enableOffline]);

  // Get recent foods
  const getRecent = useCallback(async (): Promise<FoodSearchResult[]> => {
    if (!enableOffline || !isCacheReady) return [];

    try {
      const recent = await getRecentFoods(undefined, 10);
      return recent.map(cachedToResult);
    } catch (err) {
      console.error('[useFoodSearch] Failed to get recent foods:', err);
      return [];
    }
  }, [enableOffline, isCacheReady, cachedToResult]);

  // Clear results
  const clearResults = useCallback(() => {
    setQuery('');
    setResults([]);
    setTotalResults(0);
    setHasMore(false);
    setError(null);
    setPage(1);
    lastQueryRef.current = '';
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return {
    query,
    setQuery: handleQueryChange,
    results,
    isSearching,
    isSearchingOnline,
    error,
    page,
    totalResults,
    hasMore,
    loadMore,
    filter,
    setFilter: handleFilterChange,
    search,
    selectFood,
    getRecent,
    clearResults,
    cacheStats,
    isCacheReady,
    isSyncing,
    syncProgress,
  };
}

export default useFoodSearch;
