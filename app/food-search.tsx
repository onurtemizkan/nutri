/**
 * Food Search Screen
 *
 * Search USDA FoodData Central database for foods
 * Features: debounced search, data type filters, result cards with nutrition preview
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Keyboard,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { foodsApi } from '@/lib/api/foods';
import { FoodSearchCard } from '@/lib/components/FoodSearchCard';
import { USDAFood, FoodFilterTab, DATA_TYPE_FILTERS } from '@/lib/types/foods';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

// Debounce delay in ms
const SEARCH_DEBOUNCE = 300;
const DEFAULT_LIMIT = 25;

// Filter tabs
const FILTER_TABS: Array<{ key: FoodFilterTab; label: string }> = [
  { key: 'all', label: 'All' },
  { key: 'whole', label: 'Whole Foods' },
  { key: 'branded', label: 'Branded' },
  { key: 'meals', label: 'Meals' },
];

export default function FoodSearchScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ initialQuery?: string; fromClassification?: string }>();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // State
  const [query, setQuery] = useState(params.initialQuery || '');
  const [activeFilter, setActiveFilter] = useState<FoodFilterTab>('all');
  const [foods, setFoods] = useState<USDAFood[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [totalResults, setTotalResults] = useState(0);

  // Refs
  const searchInputRef = useRef<TextInput>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastQueryRef = useRef<string>('');
  const lastFilterRef = useRef<FoodFilterTab>('all');

  // Search function
  const performSearch = useCallback(
    async (searchQuery: string, filter: FoodFilterTab, pageNum: number, append: boolean = false) => {
      if (!searchQuery.trim()) {
        setFoods([]);
        setTotalResults(0);
        setHasMore(false);
        return;
      }

      if (pageNum === 1) {
        setIsLoading(true);
      } else {
        setIsLoadingMore(true);
      }
      setError(null);

      try {
        const result = await foodsApi.searchFoods({
          query: searchQuery.trim(),
          page: pageNum,
          limit: DEFAULT_LIMIT,
          dataType: DATA_TYPE_FILTERS[filter],
        });

        if (append) {
          setFoods((prev) => [...prev, ...result.foods]);
        } else {
          setFoods(result.foods);
        }
        setTotalResults(result.pagination.total);
        setHasMore(result.pagination.hasNextPage);
        setPage(pageNum);
      } catch (err) {
        setError(getErrorMessage(err, 'Failed to search foods'));
        if (!append) {
          setFoods([]);
        }
      } finally {
        setIsLoading(false);
        setIsLoadingMore(false);
      }
    },
    []
  );

  // Debounced search
  const debouncedSearch = useCallback(
    (searchQuery: string, filter: FoodFilterTab) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = setTimeout(() => {
        // Only search if query or filter changed
        if (searchQuery !== lastQueryRef.current || filter !== lastFilterRef.current) {
          lastQueryRef.current = searchQuery;
          lastFilterRef.current = filter;
          performSearch(searchQuery, filter, 1, false);
        }
      }, SEARCH_DEBOUNCE);
    },
    [performSearch]
  );

  // Handle query change
  const handleQueryChange = useCallback(
    (text: string) => {
      setQuery(text);
      debouncedSearch(text, activeFilter);
    },
    [activeFilter, debouncedSearch]
  );

  // Handle filter change
  const handleFilterChange = useCallback(
    (filter: FoodFilterTab) => {
      setActiveFilter(filter);
      if (query.trim()) {
        debouncedSearch(query, filter);
      }
    },
    [query, debouncedSearch]
  );

  // Load more
  const loadMore = useCallback(() => {
    if (!isLoadingMore && hasMore && query.trim()) {
      performSearch(query, activeFilter, page + 1, true);
    }
  }, [isLoadingMore, hasMore, query, activeFilter, page, performSearch]);

  // Clear search
  const clearSearch = useCallback(() => {
    setQuery('');
    setFoods([]);
    setTotalResults(0);
    setHasMore(false);
    lastQueryRef.current = '';
    searchInputRef.current?.focus();
  }, []);

  // Handle food selection
  const handleFoodSelect = useCallback(
    (food: USDAFood) => {
      // Record the selection for recent foods
      foodsApi.recordFoodSelection(food.fdcId).catch(() => {
        // Ignore errors - this is a non-critical operation
      });

      // Navigate back with selected food data
      // The parent screen should handle this via params
      router.back();
      // Emit event or use context to pass selected food
      // For now, we'll use a simple approach with params
      router.setParams({
        selectedFoodId: food.fdcId.toString(),
        selectedFoodName: food.description,
        selectedFoodCalories: food.calories.toString(),
        selectedFoodProtein: food.protein.toString(),
        selectedFoodCarbs: food.carbs.toString(),
        selectedFoodFat: food.fat.toString(),
      });
    },
    [router]
  );

  // Initial search if coming from classification
  useEffect(() => {
    if (params.initialQuery) {
      performSearch(params.initialQuery, 'all', 1, false);
      lastQueryRef.current = params.initialQuery;
    }
  }, [params.initialQuery, performSearch]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // Render empty state
  const renderEmptyState = () => {
    if (isLoading) return null;

    if (error) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="alert-circle-outline" size={48} color={colors.status.error} />
          <Text style={styles.emptyTitle}>Search Error</Text>
          <Text style={styles.emptyText}>{error}</Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => performSearch(query, activeFilter, 1, false)}
          >
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      );
    }

    if (!query.trim()) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="search-outline" size={48} color={colors.text.tertiary} />
          <Text style={styles.emptyTitle}>Search for Foods</Text>
          <Text style={styles.emptyText}>
            Type to search over 500,000 foods from the USDA database
          </Text>
        </View>
      );
    }

    if (foods.length === 0) {
      return (
        <View style={styles.emptyState}>
          <Ionicons name="fast-food-outline" size={48} color={colors.text.tertiary} />
          <Text style={styles.emptyTitle}>No Results</Text>
          <Text style={styles.emptyText}>
            Try a different search term or change the filter
          </Text>
        </View>
      );
    }

    return null;
  };

  // Render footer
  const renderFooter = () => {
    if (isLoadingMore) {
      return (
        <View style={styles.footer}>
          <ActivityIndicator size="small" color={colors.primary.main} />
        </View>
      );
    }
    return null;
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={[styles.header, { paddingHorizontal: responsiveSpacing.horizontal }]}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Search Foods</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Search Input */}
      <View
        style={[
          styles.searchContainer,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && { maxWidth: FORM_MAX_WIDTH, alignSelf: 'center', width: '100%' },
        ]}
      >
        <View style={styles.searchInputContainer}>
          <Ionicons
            name="search"
            size={20}
            color={colors.text.tertiary}
            style={styles.searchIcon}
          />
          <TextInput
            ref={searchInputRef}
            style={styles.searchInput}
            placeholder="Search foods..."
            placeholderTextColor={colors.text.disabled}
            value={query}
            onChangeText={handleQueryChange}
            returnKeyType="search"
            autoCapitalize="none"
            autoCorrect={false}
            autoFocus={!params.initialQuery}
          />
          {query.length > 0 && (
            <TouchableOpacity
              onPress={clearSearch}
              style={styles.clearButton}
              accessibilityLabel="Clear search"
            >
              <Ionicons name="close-circle" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      {/* Filter Tabs */}
      <View
        style={[
          styles.filterContainer,
          isTablet && { maxWidth: FORM_MAX_WIDTH, alignSelf: 'center', width: '100%' },
        ]}
      >
        {FILTER_TABS.map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[
              styles.filterTab,
              activeFilter === tab.key && styles.filterTabActive,
            ]}
            onPress={() => handleFilterChange(tab.key)}
            accessibilityLabel={`Filter by ${tab.label}`}
            accessibilityState={{ selected: activeFilter === tab.key }}
          >
            {activeFilter === tab.key ? (
              <LinearGradient
                colors={gradients.primary as [string, string]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.filterTabGradient}
              >
                <Text style={styles.filterTabTextActive}>{tab.label}</Text>
              </LinearGradient>
            ) : (
              <Text style={styles.filterTabText}>{tab.label}</Text>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {/* Results count */}
      {query.trim() && !isLoading && foods.length > 0 && (
        <View
          style={[
            styles.resultsCount,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && { maxWidth: FORM_MAX_WIDTH, alignSelf: 'center', width: '100%' },
          ]}
        >
          <Text style={styles.resultsCountText}>
            {totalResults.toLocaleString()} results
          </Text>
        </View>
      )}

      {/* Loading indicator */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Searching...</Text>
        </View>
      )}

      {/* Results or Empty State */}
      {!isLoading && (
        <FlatList
          data={foods}
          keyExtractor={(item) => item.fdcId.toString()}
          renderItem={({ item }) => (
            <FoodSearchCard food={item} onPress={handleFoodSelect} />
          )}
          contentContainerStyle={[
            styles.listContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && { maxWidth: FORM_MAX_WIDTH, alignSelf: 'center', width: '100%' },
            foods.length === 0 && styles.listContentEmpty,
          ]}
          ListEmptyComponent={renderEmptyState}
          ListFooterComponent={renderFooter}
          onEndReached={loadMore}
          onEndReachedThreshold={0.3}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          onScrollBeginDrag={Keyboard.dismiss}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  backButton: {
    padding: spacing.xs,
  },
  headerTitle: {
    flex: 1,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    textAlign: 'center',
  },
  headerSpacer: {
    width: 32,
  },
  searchContainer: {
    paddingBottom: spacing.md,
  },
  searchInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.md,
    ...shadows.sm,
  },
  searchIcon: {
    marginRight: spacing.sm,
  },
  searchInput: {
    flex: 1,
    height: 48,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  clearButton: {
    padding: spacing.xs,
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
    gap: spacing.sm,
  },
  filterTab: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  filterTabActive: {
    backgroundColor: 'transparent',
  },
  filterTabGradient: {
    flex: 1,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
  },
  filterTabText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium as '500',
  },
  filterTabTextActive: {
    fontSize: typography.fontSize.sm,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold as '600',
  },
  resultsCount: {
    paddingBottom: spacing.sm,
  },
  resultsCountText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  listContent: {
    paddingBottom: spacing.xl,
  },
  listContentEmpty: {
    flex: 1,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
  },
  emptyTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: typography.fontSize.md * 1.5,
  },
  retryButton: {
    marginTop: spacing.lg,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  footer: {
    paddingVertical: spacing.lg,
    alignItems: 'center',
  },
});
