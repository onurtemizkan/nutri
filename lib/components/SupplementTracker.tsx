import { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { supplementsApi } from '@/lib/api/supplements';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import type { TodaySupplementStatus, SupplementStatus, SupplementTimeOfDay, Supplement } from '@/lib/types';

const TIME_LABELS: Record<SupplementTimeOfDay, string> = {
  MORNING: 'Morning',
  AFTERNOON: 'Afternoon',
  EVENING: 'Evening',
  BEFORE_BED: 'Before Bed',
  WITH_BREAKFAST: 'Breakfast',
  WITH_LUNCH: 'Lunch',
  WITH_DINNER: 'Dinner',
  EMPTY_STOMACH: 'Empty Stomach',
};

const TIME_ICONS: Record<SupplementTimeOfDay, string> = {
  MORNING: 'sunny-outline',
  AFTERNOON: 'partly-sunny-outline',
  EVENING: 'moon-outline',
  BEFORE_BED: 'bed-outline',
  WITH_BREAKFAST: 'cafe-outline',
  WITH_LUNCH: 'restaurant-outline',
  WITH_DINNER: 'pizza-outline',
  EMPTY_STOMACH: 'water-outline',
};

// Canonical time order for sorting
const TIME_ORDER: SupplementTimeOfDay[] = [
  'MORNING', 'WITH_BREAKFAST', 'EMPTY_STOMACH',
  'AFTERNOON', 'WITH_LUNCH',
  'EVENING', 'WITH_DINNER',
  'BEFORE_BED',
];

// Determine current time slot based on hour
function getCurrentTimeSlot(): SupplementTimeOfDay {
  const hour = new Date().getHours();
  if (hour < 10) return 'MORNING';
  if (hour < 14) return 'AFTERNOON';
  if (hour < 18) return 'EVENING';
  return 'BEFORE_BED';
}

// Get priority for sorting time slots (current time first, then upcoming, past at end)
function getTimePriority(time: SupplementTimeOfDay): number {
  const currentSlot = getCurrentTimeSlot();
  const currentIndex = TIME_ORDER.findIndex(t => t === currentSlot);
  const timeIndex = TIME_ORDER.findIndex(t => t === time);

  if (timeIndex >= currentIndex) {
    return timeIndex - currentIndex;
  }
  return TIME_ORDER.length + timeIndex;
}

// Represents a single dose slot for a supplement
interface DoseSlot {
  supplement: Supplement;
  supplementId: string;
  timeSlot: SupplementTimeOfDay;
  slotIndex: number;        // Position in timeOfDay array (0, 1, 2)
  totalSlots: number;       // Total number of doses per day
  takenCount: number;       // How many doses taken today
  isSlotComplete: boolean;  // Is THIS specific slot marked as taken
  isNextToTake: boolean;    // Is this the next slot user should take
  isAllComplete: boolean;   // Are ALL doses for this supplement complete
}

interface Props {
  onRefreshNeeded?: () => void;
}

export function SupplementTracker({ onRefreshNeeded }: Props) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [todayStatus, setTodayStatus] = useState<TodaySupplementStatus | null>(null);

  const loadData = useCallback(async () => {
    try {
      const data = await supplementsApi.getTodayStatus();
      setTodayStatus(data);
    } catch (error) {
      console.error('Failed to load supplement status:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleLogIntake = async (supplementId: string, skipped: boolean = false) => {
    try {
      await supplementsApi.logIntake({
        supplementId,
        skipped,
      });
      await loadData();
      onRefreshNeeded?.();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to log intake'));
    }
  };

  // Create dose slots for each time slot of each supplement
  // Multi-daily supplements get multiple entries (one per time slot)
  const createDoseSlots = useCallback((supplements: SupplementStatus[]) => {
    const groups: Record<string, DoseSlot[]> = {};
    const asNeeded: SupplementStatus[] = [];

    supplements.forEach(status => {
      const { supplement, takenCount, targetCount, isComplete } = status;

      // AS_NEEDED supplements go to their own section
      if (supplement.frequency === 'AS_NEEDED') {
        asNeeded.push(status);
        return;
      }

      // Get the time slots for this supplement
      const timeSlots = supplement.timeOfDay.length > 0
        ? supplement.timeOfDay
        : ['MORNING' as SupplementTimeOfDay]; // Default to morning if no time specified

      // For multi-daily supplements, create a dose slot for EACH time slot
      timeSlots.forEach((timeSlot, slotIndex) => {
        // Determine if this specific slot is complete
        // We assume doses are taken in order: slot 0 first, then slot 1, etc.
        const isSlotComplete = takenCount > slotIndex;

        // This slot is "next to take" if all previous slots are done and this one isn't
        const isNextToTake = takenCount === slotIndex;

        const doseSlot: DoseSlot = {
          supplement,
          supplementId: supplement.id,
          timeSlot,
          slotIndex,
          totalSlots: timeSlots.length,
          takenCount,
          isSlotComplete,
          isNextToTake,
          isAllComplete: isComplete,
        };

        if (!groups[timeSlot]) {
          groups[timeSlot] = [];
        }
        groups[timeSlot].push(doseSlot);
      });
    });

    // Sort groups by time priority (current/upcoming first)
    const sortedGroups = Object.entries(groups)
      .sort(([a], [b]) => getTimePriority(a as SupplementTimeOfDay) - getTimePriority(b as SupplementTimeOfDay));

    return { sortedGroups, asNeeded };
  }, []);

  if (isLoading) {
    return (
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Supplements</Text>
        </View>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color={colors.primary.main} />
        </View>
      </View>
    );
  }

  // No supplements at all
  if (!todayStatus || todayStatus.supplements.length === 0) {
    return (
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Supplements</Text>
          <TouchableOpacity
            onPress={() => router.push('/supplements')}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
            accessibilityLabel="Manage supplements"
          >
            <Ionicons name="add-circle-outline" size={24} color={colors.primary.main} />
          </TouchableOpacity>
        </View>
        <TouchableOpacity
          style={styles.emptyState}
          onPress={() => router.push('/supplements')}
          activeOpacity={0.7}
        >
          <Ionicons name="medical-outline" size={32} color={colors.text.disabled} />
          <Text style={styles.emptyStateText}>Add your first supplement</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const { sortedGroups, asNeeded } = createDoseSlots(todayStatus.supplements);
  const hasRequired = todayStatus.totalSupplements > 0;
  const allComplete = todayStatus.completionRate === 100;

  return (
    <View style={styles.section}>
      {/* Header */}
      <View style={styles.sectionHeader}>
        <View style={styles.titleRow}>
          <Text style={styles.sectionTitle}>Supplements</Text>
          {hasRequired && (
            <View
              style={[
                styles.progressBadge,
                allComplete && styles.progressBadgeComplete,
              ]}
            >
              {allComplete ? (
                <Ionicons name="checkmark" size={12} color={colors.text.primary} />
              ) : (
                <Text style={styles.progressBadgeText}>
                  {todayStatus.completedSupplements}/{todayStatus.totalSupplements}
                </Text>
              )}
            </View>
          )}
        </View>
        <TouchableOpacity
          onPress={() => router.push('/supplements')}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          accessibilityLabel="Manage supplements"
        >
          <Ionicons name="settings-outline" size={20} color={colors.text.tertiary} />
        </TouchableOpacity>
      </View>

      {/* Progress bar */}
      {hasRequired && (
        <View style={styles.progressBar}>
          <LinearGradient
            colors={allComplete ? gradients.success : gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[
              styles.progressFill,
              { width: `${Math.min(todayStatus.completionRate, 100)}%` }
            ]}
          />
        </View>
      )}

      {/* Grouped dose slots by time */}
      {sortedGroups.map(([time, doseSlots]) => (
        <View key={time} style={styles.timeGroup}>
          <View style={styles.timeHeader}>
            <Ionicons
              name={TIME_ICONS[time as SupplementTimeOfDay] as keyof typeof Ionicons.glyphMap}
              size={14}
              color={colors.text.tertiary}
            />
            <Text style={styles.timeLabel}>
              {TIME_LABELS[time as SupplementTimeOfDay]}
            </Text>
          </View>

          {doseSlots.map(slot => (
            <DoseSlotItem
              key={`${slot.supplementId}-${slot.slotIndex}`}
              slot={slot}
              onTake={() => handleLogIntake(slot.supplementId, false)}
              onSkip={() => handleLogIntake(slot.supplementId, true)}
            />
          ))}
        </View>
      ))}

      {/* As needed supplements */}
      {asNeeded.length > 0 && (
        <View style={styles.timeGroup}>
          <View style={styles.timeHeader}>
            <Ionicons name="ellipsis-horizontal" size={14} color={colors.text.tertiary} />
            <Text style={styles.timeLabel}>As Needed</Text>
          </View>

          {asNeeded.map(status => (
            <AsNeededItem
              key={status.supplement.id}
              status={status}
              onTake={() => handleLogIntake(status.supplement.id, false)}
            />
          ))}
        </View>
      )}
    </View>
  );
}

// Component for a single dose slot (used for scheduled supplements)
interface DoseSlotItemProps {
  slot: DoseSlot;
  onTake: () => void;
  onSkip: () => void;
}

function DoseSlotItem({ slot, onTake, onSkip }: DoseSlotItemProps) {
  const { supplement, isSlotComplete, isNextToTake, totalSlots, takenCount } = slot;

  // For single-dose supplements, don't show slot indicator
  const showSlotIndicator = totalSlots > 1;

  return (
    <View style={[styles.supplementItem, isSlotComplete && styles.supplementItemComplete]}>
      <View
        style={[
          styles.colorDot,
          { backgroundColor: supplement.color || colors.primary.main },
          isSlotComplete && styles.colorDotComplete,
        ]}
      />

      <View style={styles.supplementInfo}>
        <Text style={[styles.supplementName, isSlotComplete && styles.textComplete]} numberOfLines={1}>
          {supplement.name}
        </Text>
        <Text style={[styles.supplementDosage, isSlotComplete && styles.textComplete]}>
          {supplement.dosageAmount} {supplement.dosageUnit}
          {showSlotIndicator && ` • Dose ${slot.slotIndex + 1}/${totalSlots}`}
        </Text>
      </View>

      <View style={styles.actions}>
        {isSlotComplete ? (
          <View style={styles.completeBadge}>
            <Ionicons name="checkmark-circle" size={24} color={colors.status.success} />
          </View>
        ) : isNextToTake ? (
          <>
            <TouchableOpacity
              style={styles.skipButton}
              onPress={onSkip}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              accessibilityLabel={`Skip ${supplement.name}`}
            >
              <Ionicons name="close" size={16} color={colors.text.disabled} />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.takeButton}
              onPress={onTake}
              activeOpacity={0.8}
              accessibilityLabel={`Take ${supplement.name}`}
            >
              <LinearGradient
                colors={gradients.primary}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.takeButtonGradient}
              >
                <Ionicons name="checkmark" size={16} color={colors.text.primary} />
              </LinearGradient>
            </TouchableOpacity>
          </>
        ) : (
          // Slot is pending but not next (previous slot not taken yet)
          <View style={styles.pendingBadge}>
            <Ionicons name="time-outline" size={18} color={colors.text.disabled} />
          </View>
        )}
      </View>
    </View>
  );
}

// Component for "as needed" supplements
interface AsNeededItemProps {
  status: SupplementStatus;
  onTake: () => void;
}

function AsNeededItem({ status, onTake }: AsNeededItemProps) {
  const { supplement, takenCount } = status;

  return (
    <View style={styles.supplementItem}>
      <View
        style={[
          styles.colorDot,
          { backgroundColor: supplement.color || colors.primary.main },
        ]}
      />

      <View style={styles.supplementInfo}>
        <Text style={styles.supplementName} numberOfLines={1}>
          {supplement.name}
        </Text>
        <Text style={styles.supplementDosage}>
          {supplement.dosageAmount} {supplement.dosageUnit}
          {takenCount > 0 && ` • ${takenCount} taken today`}
        </Text>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.takeButton}
          onPress={onTake}
          activeOpacity={0.8}
          accessibilityLabel={`Take ${supplement.name}`}
        >
          <LinearGradient
            colors={['#666', '#555']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.takeButtonGradient}
          >
            <Ionicons name="add" size={16} color={colors.text.primary} />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    marginBottom: spacing.xl,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  sectionTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.5,
  },
  progressBadge: {
    backgroundColor: colors.background.elevated,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.full,
    minWidth: 32,
    alignItems: 'center',
  },
  progressBadgeComplete: {
    backgroundColor: colors.status.success,
  },
  progressBadgeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  progressBar: {
    height: 4,
    backgroundColor: colors.background.elevated,
    borderRadius: 2,
    marginBottom: spacing.md,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },

  // Loading
  loadingContainer: {
    paddingVertical: spacing.xl,
    alignItems: 'center',
  },

  // Empty state
  emptyState: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    alignItems: 'center',
    gap: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    borderStyle: 'dashed',
  },
  emptyStateText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },

  // Time groups
  timeGroup: {
    marginBottom: spacing.md,
  },
  timeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.xs,
  },
  timeLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },

  // Supplement item
  supplementItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    marginBottom: spacing.xs,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  supplementItemComplete: {
    opacity: 0.6,
  },
  colorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.sm,
  },
  colorDotComplete: {
    opacity: 0.5,
  },
  textComplete: {
    textDecorationLine: 'line-through',
    opacity: 0.7,
  },
  supplementInfo: {
    flex: 1,
    marginRight: spacing.sm,
  },
  supplementName: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  supplementDosage: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 1,
  },

  // Actions
  actions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  completeBadge: {
    padding: 2,
  },
  pendingBadge: {
    padding: 2,
    opacity: 0.5,
  },
  skipButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.background.elevated,
    justifyContent: 'center',
    alignItems: 'center',
  },
  takeButton: {
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
  },
  takeButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    gap: 2,
  },
  remainingText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
});
