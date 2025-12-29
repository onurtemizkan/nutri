/**
 * Insights Feed Screen
 * Displays ML-generated personalized nutrition insights and recommendations
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '@/lib/context/AuthContext';
import { insightsApi } from '@/lib/api/insights';
import { InsightCard, InsightSummaryCard } from '@/lib/components/InsightCard';
import {
  MLInsight,
  InsightSummary,
  MLInsightType,
  getInsightTypeLabel,
} from '@/lib/types/insights';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';

// ============================================================================
// TYPES
// ============================================================================

type FilterType = 'all' | MLInsightType;
type SortType = 'priority' | 'recent' | 'unread';

interface FilterOption {
  value: FilterType;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
}

const FILTER_OPTIONS: FilterOption[] = [
  { value: 'all', label: 'All', icon: 'apps' },
  { value: 'CORRELATION', label: 'Correlations', icon: 'analytics' },
  { value: 'RECOMMENDATION', label: 'Tips', icon: 'bulb' },
  { value: 'GOAL_PROGRESS', label: 'Goals', icon: 'flag' },
  { value: 'PATTERN_DETECTED', label: 'Patterns', icon: 'git-branch' },
];

// ============================================================================
// COMPONENT
// ============================================================================

export default function InsightsScreen() {
  const [insights, setInsights] = useState<MLInsight[]>([]);
  const [summary, setSummary] = useState<InsightSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<FilterType>('all');
  // Sorting is handled server-side, state kept for future client-side sorting
  const [_sortBy] = useState<SortType>('priority');

  const { user } = useAuth();
  const router = useRouter();
  const { getResponsiveValue } = useResponsive();

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  // Load insights
  const loadInsights = useCallback(
    async (reset = false) => {
      if (!user) {
        setIsLoading(false);
        return;
      }

      try {
        setError(null);
        const newOffset = reset ? 0 : offset;

        const params = {
          limit: 20,
          offset: newOffset,
          dismissed: false,
          ...(filter !== 'all' && { insightType: filter as MLInsightType }),
        };

        const [feedData, summaryData] = await Promise.all([
          insightsApi.getInsights(params),
          reset || !summary ? insightsApi.getSummary() : Promise.resolve(summary),
        ]);

        if (reset) {
          setInsights(feedData.insights);
        } else {
          setInsights((prev) => [...prev, ...feedData.insights]);
        }

        setSummary(summaryData);
        setHasMore(feedData.hasMore);
        setOffset(newOffset + feedData.insights.length);
      } catch (err) {
        console.error('Failed to load insights:', err);
        setError('Failed to load insights');
      } finally {
        setIsLoading(false);
        setRefreshing(false);
      }
    },
    [user, offset, filter, summary]
  );

  // Initial load and refresh on focus
  useFocusEffect(
    useCallback(() => {
      setIsLoading(true);
      loadInsights(true);
    }, [loadInsights])
  );

  // Handle pull-to-refresh
  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    loadInsights(true);
  }, [loadInsights]);

  // Load more on scroll
  const handleLoadMore = useCallback(() => {
    if (hasMore && !isLoading) {
      loadInsights(false);
    }
  }, [hasMore, isLoading, loadInsights]);

  // Generate new insights
  const handleGenerate = useCallback(async () => {
    if (isGenerating) return;

    setIsGenerating(true);
    try {
      const result = await insightsApi.generate({ regenerate: false });

      if (result.errors.length > 0) {
        Alert.alert(
          'Partial Success',
          `Generated ${result.generated} insights with ${result.errors.length} warnings.`,
          [{ text: 'OK' }]
        );
      } else if (result.generated === 0) {
        Alert.alert(
          'No New Insights',
          'Not enough data to generate insights yet. Keep logging your meals and health metrics!',
          [{ text: 'OK' }]
        );
      } else {
        Alert.alert('Success', `Generated ${result.generated} new insights!`, [{ text: 'View' }]);
      }

      // Refresh the list
      loadInsights(true);
    } catch (err) {
      console.error('Failed to generate insights:', err);
      Alert.alert('Error', 'Failed to generate insights. Please try again later.', [
        { text: 'OK' },
      ]);
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating, loadInsights]);

  // Handle insight actions
  const handleInsightPress = useCallback(async (insight: MLInsight) => {
    // Mark as viewed only after successful API call
    if (!insight.viewed) {
      try {
        await insightsApi.markAsViewed(insight.id);
        setInsights((prev) => prev.map((i) => (i.id === insight.id ? { ...i, viewed: true } : i)));
      } catch (err) {
        console.error('Failed to mark insight as viewed:', err);
        // State is not updated if API fails
      }
    }
    // Could navigate to detail view if needed
  }, []);

  const handleDismiss = useCallback(
    async (insightId: string) => {
      try {
        await insightsApi.dismiss(insightId);
        setInsights((prev) => prev.filter((i) => i.id !== insightId));
        if (summary) {
          setSummary({
            ...summary,
            totalInsights: summary.totalInsights - 1,
          });
        }
      } catch (err) {
        console.error('Failed to dismiss insight:', err);
      }
    },
    [summary]
  );

  const handleFeedback = useCallback(async (insightId: string, helpful: boolean) => {
    try {
      await insightsApi.provideFeedback(insightId, { helpful });
      setInsights((prev) => prev.map((i) => (i.id === insightId ? { ...i, helpful } : i)));
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    }
  }, []);

  // Render filter chips
  const renderFilterChips = () => (
    <View style={styles.filterContainer}>
      <FlatList
        horizontal
        showsHorizontalScrollIndicator={false}
        data={FILTER_OPTIONS}
        keyExtractor={(item) => item.value}
        contentContainerStyle={styles.filterList}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={[styles.filterChip, filter === item.value && styles.filterChipActive]}
            onPress={() => {
              setFilter(item.value);
              setOffset(0);
            }}
          >
            <Ionicons
              name={item.icon}
              size={14}
              color={filter === item.value ? colors.text.primary : colors.text.tertiary}
            />
            <Text
              style={[styles.filterChipText, filter === item.value && styles.filterChipTextActive]}
            >
              {item.label}
            </Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );

  // Render header with summary
  const renderHeader = () => (
    <View style={styles.headerContainer}>
      {/* Summary card */}
      {summary && (summary.unviewedCount > 0 || summary.highPriorityCount > 0) && (
        <InsightSummaryCard
          totalInsights={summary.totalInsights}
          unviewedCount={summary.unviewedCount}
          highPriorityCount={summary.highPriorityCount}
          onPress={() => setFilter('all')}
        />
      )}

      {/* Generate button */}
      <TouchableOpacity
        style={styles.generateButton}
        onPress={handleGenerate}
        disabled={isGenerating}
      >
        {isGenerating ? (
          <ActivityIndicator size="small" color={colors.primary.main} />
        ) : (
          <Ionicons name="sparkles" size={18} color={colors.primary.main} />
        )}
        <Text style={styles.generateButtonText}>
          {isGenerating ? 'Generating...' : 'Generate New Insights'}
        </Text>
      </TouchableOpacity>

      {/* Filter chips */}
      {renderFilterChips()}

      {/* Results count */}
      {!isLoading && (
        <Text style={styles.resultsCount}>
          {insights.length} insight{insights.length !== 1 ? 's' : ''}{' '}
          {filter !== 'all' && `(${getInsightTypeLabel(filter as MLInsightType)})`}
        </Text>
      )}
    </View>
  );

  // Render empty state
  const renderEmptyState = () => (
    <View style={styles.emptyContainer}>
      <LinearGradient
        colors={gradients.primary}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.emptyIconContainer}
      >
        <Ionicons name="bulb" size={48} color={colors.text.primary} />
      </LinearGradient>
      <Text style={styles.emptyTitle}>No Insights Yet</Text>
      <Text style={styles.emptyDescription}>
        We need more data to generate personalized insights. Keep logging your meals and syncing
        your health metrics!
      </Text>
      <TouchableOpacity style={styles.emptyButton} onPress={handleGenerate} disabled={isGenerating}>
        <Text style={styles.emptyButtonText}>
          {isGenerating ? 'Generating...' : 'Try Generating Insights'}
        </Text>
      </TouchableOpacity>
    </View>
  );

  // Render error state
  if (error && !refreshing) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.errorContainer}>
          <Ionicons name="cloud-offline" size={48} color={colors.semantic.error} />
          <Text style={styles.errorTitle}>Unable to Load Insights</Text>
          <Text style={styles.errorDescription}>{error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={() => loadInsights(true)}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // Render insight item
  const renderInsight = ({ item }: { item: MLInsight }) => (
    <InsightCard
      insight={item}
      onPress={() => handleInsightPress(item)}
      onDismiss={() => handleDismiss(item.id)}
      onFeedback={(helpful) => handleFeedback(item.id, helpful)}
      onMarkViewed={() =>
        setInsights((prev) => prev.map((i) => (i.id === item.id ? { ...i, viewed: true } : i)))
      }
    />
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={[styles.header, { paddingHorizontal: contentPadding }]}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.title}>Insights</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity
            onPress={handleRefresh}
            style={styles.refreshButton}
            disabled={refreshing}
          >
            <Ionicons
              name="refresh"
              size={22}
              color={refreshing ? colors.text.disabled : colors.text.secondary}
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* Content */}
      {isLoading && insights.length === 0 ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading insights...</Text>
        </View>
      ) : (
        <FlatList
          data={insights}
          keyExtractor={(item) => item.id}
          renderItem={renderInsight}
          contentContainerStyle={[
            styles.listContent,
            { paddingHorizontal: contentPadding },
            insights.length === 0 && styles.listContentEmpty,
          ]}
          ListHeaderComponent={renderHeader}
          ListEmptyComponent={renderEmptyState}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={handleRefresh}
              tintColor={colors.primary.main}
              colors={[colors.primary.main]}
            />
          }
          onEndReached={handleLoadMore}
          onEndReachedThreshold={0.3}
          ListFooterComponent={
            hasMore ? (
              <View style={styles.loadMoreContainer}>
                <ActivityIndicator size="small" color={colors.primary.main} />
              </View>
            ) : null
          }
        />
      )}
    </SafeAreaView>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    padding: spacing.xs,
    marginLeft: -spacing.xs,
  },
  title: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  refreshButton: {
    padding: spacing.xs,
  },

  // Header content
  headerContainer: {
    marginBottom: spacing.md,
  },

  // Generate button
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    borderStyle: 'dashed',
  },
  generateButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Filters
  filterContainer: {
    marginBottom: spacing.md,
  },
  filterList: {
    gap: spacing.sm,
  },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  filterChipActive: {
    backgroundColor: colors.primary.main,
    borderColor: colors.primary.main,
  },
  filterChipText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
  },
  filterChipTextActive: {
    color: colors.text.primary,
  },

  // Results count
  resultsCount: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
  },

  // List
  listContent: {
    paddingTop: spacing.md,
    paddingBottom: spacing.xl * 2,
  },
  listContentEmpty: {
    flexGrow: 1,
  },

  // Loading
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
  },
  loadingText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  loadMoreContainer: {
    padding: spacing.lg,
    alignItems: 'center',
  },

  // Empty state
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.xl * 2,
  },
  emptyIconContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  emptyTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
    textAlign: 'center',
  },
  emptyDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: typography.fontSize.sm * 1.5,
    marginBottom: spacing.lg,
  },
  emptyButton: {
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
  },
  emptyButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Error state
  errorContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.md,
  },
  errorTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  errorDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    marginTop: spacing.sm,
  },
  retryButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
});
