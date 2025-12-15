import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Modal,
  TextInput,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { useResponsive } from '@/hooks/useResponsive';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { supplementsApi } from '@/lib/api/supplements';
import type {
  Supplement,
  TodaySupplementStatus,
  SupplementStatus,
  CreateSupplementInput,
  SupplementFrequency,
  SupplementTimeOfDay,
} from '@/lib/types';

const FREQUENCY_OPTIONS: { value: SupplementFrequency; label: string }[] = [
  { value: 'DAILY', label: 'Daily' },
  { value: 'TWICE_DAILY', label: 'Twice Daily' },
  { value: 'THREE_TIMES_DAILY', label: '3x Daily' },
  { value: 'WEEKLY', label: 'Weekly' },
  { value: 'EVERY_OTHER_DAY', label: 'Every Other Day' },
  { value: 'AS_NEEDED', label: 'As Needed' },
];

const TIME_OF_DAY_OPTIONS: { value: SupplementTimeOfDay; label: string }[] = [
  { value: 'MORNING', label: 'Morning' },
  { value: 'AFTERNOON', label: 'Afternoon' },
  { value: 'EVENING', label: 'Evening' },
  { value: 'BEFORE_BED', label: 'Before Bed' },
  { value: 'WITH_BREAKFAST', label: 'With Breakfast' },
  { value: 'WITH_LUNCH', label: 'With Lunch' },
  { value: 'WITH_DINNER', label: 'With Dinner' },
  { value: 'EMPTY_STOMACH', label: 'Empty Stomach' },
];

const SUPPLEMENT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
  '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
];

export default function SupplementsScreen() {
  const router = useRouter();
  const { getResponsiveValue } = useResponsive();

  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [todayStatus, setTodayStatus] = useState<TodaySupplementStatus | null>(null);
  const [supplements, setSupplements] = useState<Supplement[]>([]);

  // Add/Edit modal state
  const [showModal, setShowModal] = useState(false);
  const [editingSupplement, setEditingSupplement] = useState<Supplement | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // Form state
  const [formName, setFormName] = useState('');
  const [formBrand, setFormBrand] = useState('');
  const [formDosageAmount, setFormDosageAmount] = useState('');
  const [formDosageUnit, setFormDosageUnit] = useState('mg');
  const [formFrequency, setFormFrequency] = useState<SupplementFrequency>('DAILY');
  const [formTimeOfDay, setFormTimeOfDay] = useState<SupplementTimeOfDay[]>(['MORNING']);
  const [formWithFood, setFormWithFood] = useState(false);
  const [formColor, setFormColor] = useState(SUPPLEMENT_COLORS[0]);
  const [formNotes, setFormNotes] = useState('');

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });
  const maxContentWidth = getResponsiveValue({
    small: undefined,
    medium: undefined,
    large: 600,
    tablet: 500,
    default: undefined,
  });

  const loadData = useCallback(async () => {
    try {
      const [statusData, supplementsData] = await Promise.all([
        supplementsApi.getTodayStatus(),
        supplementsApi.getAll(false),
      ]);
      setTodayStatus(statusData);
      setSupplements(supplementsData);
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to load supplements'));
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, [loadData]);

  const resetForm = () => {
    setFormName('');
    setFormBrand('');
    setFormDosageAmount('');
    setFormDosageUnit('mg');
    setFormFrequency('DAILY');
    setFormTimeOfDay(['MORNING']);
    setFormWithFood(false);
    setFormColor(SUPPLEMENT_COLORS[Math.floor(Math.random() * SUPPLEMENT_COLORS.length)]);
    setFormNotes('');
    setEditingSupplement(null);
  };

  const openAddModal = () => {
    resetForm();
    setShowModal(true);
  };

  const openEditModal = (supplement: Supplement) => {
    setEditingSupplement(supplement);
    setFormName(supplement.name);
    setFormBrand(supplement.brand || '');
    setFormDosageAmount(supplement.dosageAmount.toString());
    setFormDosageUnit(supplement.dosageUnit);
    setFormFrequency(supplement.frequency);
    setFormTimeOfDay(supplement.timeOfDay);
    setFormWithFood(supplement.withFood);
    setFormColor(supplement.color || SUPPLEMENT_COLORS[0]);
    setFormNotes(supplement.notes || '');
    setShowModal(true);
  };

  const handleSave = async () => {
    if (!formName.trim()) {
      showAlert('Error', 'Please enter a supplement name');
      return;
    }
    if (!formDosageAmount || parseFloat(formDosageAmount) <= 0) {
      showAlert('Error', 'Please enter a valid dosage amount');
      return;
    }

    setIsSaving(true);
    try {
      const data: CreateSupplementInput = {
        name: formName.trim(),
        brand: formBrand.trim() || undefined,
        dosageAmount: parseFloat(formDosageAmount),
        dosageUnit: formDosageUnit,
        frequency: formFrequency,
        timeOfDay: formTimeOfDay,
        withFood: formWithFood,
        color: formColor,
        notes: formNotes.trim() || undefined,
      };

      if (editingSupplement) {
        await supplementsApi.update(editingSupplement.id, data);
        showAlert('Success', 'Supplement updated successfully');
      } else {
        await supplementsApi.create(data);
        showAlert('Success', 'Supplement added successfully');
      }

      setShowModal(false);
      resetForm();
      loadData();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to save supplement'));
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = (supplement: Supplement) => {
    showAlert(
      'Delete Supplement',
      `Are you sure you want to delete "${supplement.name}"? This will also delete all intake logs.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await supplementsApi.delete(supplement.id);
              showAlert('Success', 'Supplement deleted');
              loadData();
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to delete supplement'));
            }
          },
        },
      ]
    );
  };

  const handleLogIntake = async (supplementId: string, skipped: boolean = false) => {
    try {
      await supplementsApi.logIntake({
        supplementId,
        skipped,
      });
      loadData();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to log intake'));
    }
  };

  const toggleTimeOfDay = (time: SupplementTimeOfDay) => {
    if (formTimeOfDay.includes(time)) {
      if (formTimeOfDay.length > 1) {
        setFormTimeOfDay(formTimeOfDay.filter(t => t !== time));
      }
    } else {
      setFormTimeOfDay([...formTimeOfDay, time]);
    }
  };

  const getCompletionColor = (rate: number) => {
    if (rate >= 100) return colors.status.success;
    if (rate >= 50) return colors.status.warning;
    return colors.status.error;
  };

  const renderTodayCard = (status: SupplementStatus) => {
    const isComplete = status.isComplete;
    const remaining = status.targetCount - status.takenCount;

    return (
      <View key={status.supplement.id} style={styles.todayCard}>
        <View style={styles.todayCardLeft}>
          <View
            style={[
              styles.supplementColorDot,
              { backgroundColor: status.supplement.color || colors.primary.main },
            ]}
          />
          <View style={styles.todayCardInfo}>
            <Text style={styles.todayCardName}>{status.supplement.name}</Text>
            <Text style={styles.todayCardDosage}>
              {status.supplement.dosageAmount} {status.supplement.dosageUnit}
              {status.supplement.brand ? ` - ${status.supplement.brand}` : ''}
            </Text>
            <Text style={styles.todayCardProgress}>
              {status.takenCount}/{status.targetCount} taken
              {status.skippedCount > 0 && ` (${status.skippedCount} skipped)`}
            </Text>
          </View>
        </View>

        <View style={styles.todayCardActions}>
          {isComplete ? (
            <View style={styles.completeBadge}>
              <Ionicons name="checkmark-circle" size={24} color={colors.status.success} />
            </View>
          ) : (
            <>
              <TouchableOpacity
                style={styles.skipButton}
                onPress={() => handleLogIntake(status.supplement.id, true)}
                accessibilityLabel={`Skip ${status.supplement.name}`}
              >
                <Ionicons name="close" size={18} color={colors.text.tertiary} />
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.takeButton}
                onPress={() => handleLogIntake(status.supplement.id, false)}
                accessibilityLabel={`Take ${status.supplement.name}`}
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.takeButtonGradient}
                >
                  <Ionicons name="checkmark" size={18} color={colors.text.primary} />
                  {remaining > 1 && (
                    <Text style={styles.takeButtonText}>{remaining}</Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            </>
          )}
        </View>
      </View>
    );
  };

  const renderSupplementCard = (supplement: Supplement) => {
    const frequencyLabel = FREQUENCY_OPTIONS.find(f => f.value === supplement.frequency)?.label || supplement.frequency;
    const timeLabels = supplement.timeOfDay
      .map(t => TIME_OF_DAY_OPTIONS.find(opt => opt.value === t)?.label || t)
      .join(', ');

    return (
      <TouchableOpacity
        key={supplement.id}
        style={styles.supplementCard}
        onPress={() => openEditModal(supplement)}
        onLongPress={() => handleDelete(supplement)}
        activeOpacity={0.7}
        accessibilityLabel={`Edit ${supplement.name}`}
        accessibilityHint="Tap to edit, long press to delete"
      >
        <View style={styles.supplementCardLeft}>
          <View
            style={[
              styles.supplementColorBar,
              { backgroundColor: supplement.color || colors.primary.main },
            ]}
          />
          <View style={styles.supplementCardInfo}>
            <Text style={styles.supplementCardName}>{supplement.name}</Text>
            {supplement.brand && (
              <Text style={styles.supplementCardBrand}>{supplement.brand}</Text>
            )}
            <Text style={styles.supplementCardDosage}>
              {supplement.dosageAmount} {supplement.dosageUnit} - {frequencyLabel}
            </Text>
            <Text style={styles.supplementCardTime}>
              {timeLabels}
              {supplement.withFood && ' (with food)'}
            </Text>
          </View>
        </View>

        <View style={styles.supplementCardRight}>
          <TouchableOpacity
            onPress={() => handleDelete(supplement)}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
            accessibilityLabel={`Delete ${supplement.name}`}
          >
            <Ionicons name="trash-outline" size={20} color={colors.text.tertiary} />
          </TouchableOpacity>
          <Ionicons name="chevron-forward" size={20} color={colors.text.disabled} />
        </View>
      </TouchableOpacity>
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading supplements...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="supplements-screen">
      {/* Header */}
      <View style={[styles.header, { paddingHorizontal: contentPadding }]}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
          testID="supplements-back-button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Supplements</Text>
        <TouchableOpacity
          onPress={openAddModal}
          style={styles.addButton}
          accessibilityLabel="Add supplement"
          testID="supplements-add-button"
        >
          <Ionicons name="add" size={24} color={colors.primary.main} />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
          />
        }
      >
        <View
          style={[
            styles.content,
            { padding: contentPadding },
            maxContentWidth ? { maxWidth: maxContentWidth, alignSelf: 'center', width: '100%' } : null,
          ]}
        >
          {/* Today's Progress */}
          {todayStatus && todayStatus.supplements.length > 0 && (
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Today's Progress</Text>
                <View
                  style={[
                    styles.progressBadge,
                    { backgroundColor: `${getCompletionColor(todayStatus.completionRate)}20` },
                  ]}
                >
                  <Text
                    style={[
                      styles.progressBadgeText,
                      { color: getCompletionColor(todayStatus.completionRate) },
                    ]}
                  >
                    {Math.round(todayStatus.completionRate)}%
                  </Text>
                </View>
              </View>

              <View style={styles.progressBar}>
                <View
                  style={[
                    styles.progressBarFill,
                    {
                      width: `${todayStatus.completionRate}%`,
                      backgroundColor: getCompletionColor(todayStatus.completionRate),
                    },
                  ]}
                />
              </View>

              <Text style={styles.progressSummary}>
                {todayStatus.completedSupplements} of {todayStatus.totalSupplements} supplements completed
              </Text>

              {todayStatus.supplements.map(renderTodayCard)}
            </View>
          )}

          {/* All Supplements */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>All Supplements</Text>
              <Text style={styles.sectionCount}>{supplements.length}</Text>
            </View>

            {supplements.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="medical-outline" size={48} color={colors.text.disabled} />
                <Text style={styles.emptyStateTitle}>No supplements yet</Text>
                <Text style={styles.emptyStateText}>
                  Tap the + button to add your first supplement
                </Text>
              </View>
            ) : (
              supplements.map(renderSupplementCard)
            )}
          </View>
        </View>
      </ScrollView>

      {/* Add/Edit Modal */}
      <Modal
        visible={showModal}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => {
          if (!isSaving) {
            setShowModal(false);
            resetForm();
          }
        }}
      >
        <SafeAreaView style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            {/* Modal Header - Outside ScrollView for proper touch handling */}
            <View style={styles.modalHeader}>
              <TouchableOpacity
                onPress={() => {
                  setShowModal(false);
                  resetForm();
                }}
                disabled={isSaving}
                hitSlop={{ top: 15, bottom: 15, left: 15, right: 15 }}
                accessibilityLabel="Cancel"
                accessibilityRole="button"
              >
                <Text style={styles.modalCancelText}>Cancel</Text>
              </TouchableOpacity>
              <Text style={styles.modalTitle}>
                {editingSupplement ? 'Edit Supplement' : 'Add Supplement'}
              </Text>
              <TouchableOpacity
                onPress={handleSave}
                disabled={isSaving}
                hitSlop={{ top: 15, bottom: 15, left: 15, right: 15 }}
                accessibilityLabel="Save supplement"
                accessibilityRole="button"
                testID="supplements-save-button"
              >
                {isSaving ? (
                  <ActivityIndicator size="small" color={colors.primary.main} />
                ) : (
                  <Text style={styles.modalSaveText}>Save</Text>
                )}
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">

              {/* Form Fields */}
              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Name *</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={styles.input}
                    value={formName}
                    onChangeText={setFormName}
                    placeholder="e.g., Vitamin D3"
                    placeholderTextColor={colors.text.disabled}
                  />
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Brand (optional)</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={styles.input}
                    value={formBrand}
                    onChangeText={setFormBrand}
                    placeholder="e.g., Nature Made"
                    placeholderTextColor={colors.text.disabled}
                  />
                </View>
              </View>

              <View style={styles.formRow}>
                <View style={[styles.formSection, { flex: 1 }]}>
                  <Text style={styles.formLabel}>Dosage *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={formDosageAmount}
                      onChangeText={setFormDosageAmount}
                      placeholder="1000"
                      placeholderTextColor={colors.text.disabled}
                      keyboardType="numeric"
                    />
                  </View>
                </View>
                <View style={[styles.formSection, { flex: 1, marginLeft: spacing.md }]}>
                  <Text style={styles.formLabel}>Unit</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={formDosageUnit}
                      onChangeText={setFormDosageUnit}
                      placeholder="mg"
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Frequency</Text>
                <View style={styles.chipContainer}>
                  {FREQUENCY_OPTIONS.map(option => (
                    <TouchableOpacity
                      key={option.value}
                      style={[
                        styles.chip,
                        formFrequency === option.value && styles.chipSelected,
                      ]}
                      onPress={() => setFormFrequency(option.value)}
                    >
                      <Text
                        style={[
                          styles.chipText,
                          formFrequency === option.value && styles.chipTextSelected,
                        ]}
                      >
                        {option.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Time of Day</Text>
                <View style={styles.chipContainer}>
                  {TIME_OF_DAY_OPTIONS.map(option => (
                    <TouchableOpacity
                      key={option.value}
                      style={[
                        styles.chip,
                        formTimeOfDay.includes(option.value) && styles.chipSelected,
                      ]}
                      onPress={() => toggleTimeOfDay(option.value)}
                    >
                      <Text
                        style={[
                          styles.chipText,
                          formTimeOfDay.includes(option.value) && styles.chipTextSelected,
                        ]}
                      >
                        {option.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              <TouchableOpacity
                style={styles.toggleRow}
                onPress={() => setFormWithFood(!formWithFood)}
                activeOpacity={0.7}
              >
                <View style={styles.toggleInfo}>
                  <Text style={styles.toggleLabel}>Take with food</Text>
                  <Text style={styles.toggleDescription}>
                    Enable if this supplement should be taken with meals
                  </Text>
                </View>
                <View
                  style={[
                    styles.toggle,
                    formWithFood && styles.toggleActive,
                  ]}
                >
                  <View
                    style={[
                      styles.toggleKnob,
                      formWithFood && styles.toggleKnobActive,
                    ]}
                  />
                </View>
              </TouchableOpacity>

              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Color</Text>
                <View style={styles.colorContainer}>
                  {SUPPLEMENT_COLORS.map(color => (
                    <TouchableOpacity
                      key={color}
                      style={[
                        styles.colorOption,
                        { backgroundColor: color },
                        formColor === color && styles.colorOptionSelected,
                      ]}
                      onPress={() => setFormColor(color)}
                    >
                      {formColor === color && (
                        <Ionicons name="checkmark" size={16} color="#FFF" />
                      )}
                    </TouchableOpacity>
                  ))}
                </View>
              </View>

              <View style={styles.formSection}>
                <Text style={styles.formLabel}>Notes (optional)</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={[styles.input, styles.textArea]}
                    value={formNotes}
                    onChangeText={setFormNotes}
                    placeholder="Any additional notes..."
                    placeholderTextColor={colors.text.disabled}
                    multiline
                    numberOfLines={3}
                    textAlignVertical="top"
                  />
                </View>
              </View>

              {/* Extra padding at bottom for keyboard */}
              <View style={{ height: Platform.OS === 'ios' ? 40 : 20 }} />
            </ScrollView>
          </View>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
  },
  loadingText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  addButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Content
  scrollView: {
    flex: 1,
  },
  content: {},

  // Section
  section: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  sectionCount: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    backgroundColor: colors.background.elevated,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },

  // Progress
  progressBadge: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  progressBadgeText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
  },
  progressBar: {
    height: 6,
    backgroundColor: colors.background.elevated,
    borderRadius: 3,
    marginBottom: spacing.sm,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  progressSummary: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.lg,
  },

  // Today Card
  todayCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  todayCardLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  supplementColorDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: spacing.md,
  },
  todayCardInfo: {
    flex: 1,
  },
  todayCardName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  todayCardDosage: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  todayCardProgress: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
    marginTop: 4,
  },
  todayCardActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  completeBadge: {
    padding: spacing.xs,
  },
  skipButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  takeButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  takeButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.xs,
  },
  takeButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Supplement Card
  supplementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    marginBottom: spacing.sm,
    overflow: 'hidden',
  },
  supplementCardLeft: {
    flexDirection: 'row',
    flex: 1,
  },
  supplementColorBar: {
    width: 4,
    alignSelf: 'stretch',
  },
  supplementCardInfo: {
    flex: 1,
    padding: spacing.md,
  },
  supplementCardName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  supplementCardBrand: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  supplementCardDosage: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: 4,
  },
  supplementCardTime: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  supplementCardRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingRight: spacing.md,
  },

  // Empty State
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing['2xl'],
  },
  emptyStateTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginTop: spacing.md,
  },
  emptyStateText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
    textAlign: 'center',
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  modalContent: {
    flex: 1,
    padding: spacing.lg,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: spacing.md,
    marginBottom: spacing.xl,
    minHeight: 44,
  },
  modalTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  modalCancelText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  modalSaveText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Form
  formSection: {
    marginBottom: spacing.lg,
  },
  formRow: {
    flexDirection: 'row',
  },
  formLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  input: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 48,
  },
  textArea: {
    height: 80,
    paddingTop: spacing.md,
  },

  // Chips
  chipContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.elevated,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  chipSelected: {
    backgroundColor: colors.primary.main,
    borderColor: colors.primary.main,
  },
  chipText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  chipTextSelected: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
  },

  // Toggle
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    marginBottom: spacing.lg,
  },
  toggleInfo: {
    flex: 1,
    marginRight: spacing.md,
  },
  toggleLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  toggleDescription: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  toggle: {
    width: 48,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.background.elevated,
    padding: 2,
    justifyContent: 'center',
  },
  toggleActive: {
    backgroundColor: colors.primary.main,
  },
  toggleKnob: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: colors.text.primary,
    ...shadows.sm,
  },
  toggleKnobActive: {
    alignSelf: 'flex-end',
  },

  // Colors
  colorContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  colorOption: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  colorOptionSelected: {
    borderWidth: 3,
    borderColor: colors.text.primary,
  },
});
