import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { userSupplementsApi } from '@/lib/api/supplements';
import {
  UserSupplement,
  SUPPLEMENT_CATEGORIES,
  SCHEDULE_TYPES,
} from '@/lib/types/supplements';
import { useAuth } from '@/lib/context/AuthContext';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

export default function MySupplementsScreen() {
  const [supplements, setSupplements] = useState<UserSupplement[]>([]);
  const [showInactive, setShowInactive] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { user } = useAuth();
  const router = useRouter();

  const loadSupplements = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      const data = await userSupplementsApi.getAll(showInactive);
      setSupplements(data);
    } catch (error) {
      console.error('Failed to load supplements:', error);
      showAlert('Error', getErrorMessage(error, 'Failed to load supplements'));
    } finally {
      setIsLoading(false);
    }
  }, [user, showInactive]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadSupplements();
    setRefreshing(false);
  }, [loadSupplements]);

  useEffect(() => {
    loadSupplements();
  }, [loadSupplements]);

  const getCategoryLabel = (category: string) => {
    return SUPPLEMENT_CATEGORIES.find((c) => c.value === category)?.label || category;
  };

  const getScheduleLabel = (scheduleType: string) => {
    return SCHEDULE_TYPES.find((s) => s.value === scheduleType)?.label || scheduleType;
  };

  const activeSupplements = supplements.filter((s) => s.isActive);
  const inactiveSupplements = supplements.filter((s) => !s.isActive);

  const renderSupplementItem = ({ item }: { item: UserSupplement }) => (
    <TouchableOpacity
      style={[styles.supplementCard, !item.isActive && styles.supplementCardInactive]}
      onPress={() => router.push(`/edit-supplement/${item.id}` as never)}
    >
      <View style={styles.supplementInfo}>
        <View style={styles.supplementHeader}>
          <Text style={styles.supplementName}>{item.supplement.name}</Text>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: item.isActive ? colors.status.success : colors.text.disabled },
            ]}
          />
        </View>
        <Text style={styles.supplementCategory}>
          {getCategoryLabel(item.supplement.category)}
        </Text>
        <Text style={styles.supplementDosage}>
          {item.dosage} {item.unit} • {getScheduleLabel(item.scheduleType)}
        </Text>
        {item.scheduleTimes && Array.isArray(item.scheduleTimes) && item.scheduleTimes.length > 0 && (
          <Text style={styles.supplementTimes}>
            {(item.scheduleTimes as string[]).join(', ')}
          </Text>
        )}
        {item.notes && (
          <Text style={styles.supplementNotes} numberOfLines={1}>
            {item.notes}
          </Text>
        )}
      </View>
      <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
    </TouchableOpacity>
  );

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={24} color={colors.text.secondary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>My Supplements</Text>
          <TouchableOpacity onPress={() => router.push('/add-supplement' as never)}>
            <Ionicons name="add" size={24} color={colors.primary.main} />
          </TouchableOpacity>
        </View>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color={colors.text.secondary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Supplements</Text>
        <TouchableOpacity onPress={() => router.push('/add-supplement' as never)}>
          <Ionicons name="add" size={24} color={colors.primary.main} />
        </TouchableOpacity>
      </View>

      {supplements.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="medical-outline" size={64} color={colors.text.disabled} />
          <Text style={styles.emptyTitle}>No Supplements</Text>
          <Text style={styles.emptyText}>
            Add supplements to track your daily intake
          </Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={() => router.push('/add-supplement' as never)}
          >
            <Text style={styles.emptyButtonText}>Add Supplement</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={showInactive ? supplements : activeSupplements}
          keyExtractor={(item) => item.id}
          renderItem={renderSupplementItem}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={colors.primary.main}
              colors={[colors.primary.main]}
            />
          }
          ListHeaderComponent={
            <View style={styles.listHeader}>
              <Text style={styles.countText}>
                {activeSupplements.length} active supplement{activeSupplements.length !== 1 ? 's' : ''}
              </Text>
              {inactiveSupplements.length > 0 && (
                <TouchableOpacity
                  style={styles.toggleInactive}
                  onPress={() => setShowInactive(!showInactive)}
                >
                  <Text style={styles.toggleInactiveText}>
                    {showInactive ? 'Hide' : 'Show'} inactive ({inactiveSupplements.length})
                  </Text>
                </TouchableOpacity>
              )}
            </View>
          }
          ListFooterComponent={
            showInactive && inactiveSupplements.length > 0 ? (
              <View style={styles.inactiveSection}>
                <Text style={styles.inactiveTitle}>Inactive</Text>
                {inactiveSupplements.map((supp) => (
                  <View key={supp.id}>{renderSupplementItem({ item: supp })}</View>
                ))}
              </View>
            ) : null
          }
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
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContent: {
    padding: spacing.lg,
    paddingBottom: spacing['3xl'],
  },
  listHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  countText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
  },
  toggleInactive: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  toggleInactiveText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Supplement Card
  supplementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  supplementCardInactive: {
    opacity: 0.6,
  },
  supplementInfo: {
    flex: 1,
  },
  supplementHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  supplementName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  supplementCategory: {
    fontSize: typography.fontSize.xs,
    color: colors.primary.main,
    marginTop: 2,
    fontWeight: typography.fontWeight.medium,
  },
  supplementDosage: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: spacing.xs,
  },
  supplementTimes: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  supplementNotes: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    fontStyle: 'italic',
    marginTop: spacing.xs,
  },

  // Inactive Section
  inactiveSection: {
    marginTop: spacing.lg,
    paddingTop: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  inactiveTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },

  // Empty State
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  emptyTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  emptyButton: {
    backgroundColor: colors.primary.main,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
  },
  emptyButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
});
