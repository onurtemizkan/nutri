import { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { weightApi } from '@/lib/api/weight';
import { WeightUnit, kgToLb, lbToKg, formatWeight, getBmiColor } from '@/lib/types/weight';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function WeightGoalScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const [goalWeightInput, setGoalWeightInput] = useState('');
  const [unit, setUnit] = useState<WeightUnit>('kg');
  const [currentWeight, setCurrentWeight] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const progress = await weightApi.getProgress();
      if (progress.currentWeight !== null) {
        setCurrentWeight(progress.currentWeight);
      }
      if (progress.goalWeight !== null) {
        setGoalWeightInput(progress.goalWeight.toFixed(1));
      }
    } catch (error) {
      console.error('Failed to load weight data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    const weightValue = parseFloat(goalWeightInput);

    if (isNaN(weightValue) || weightValue <= 0) {
      showAlert('Invalid Weight', 'Please enter a valid goal weight');
      return;
    }

    // Convert to kg if needed
    const weightInKg = unit === 'lb' ? lbToKg(weightValue) : weightValue;

    // Validate weight range
    if (weightInKg < 20 || weightInKg > 500) {
      showAlert('Invalid Weight', 'Goal weight must be between 20 kg and 500 kg');
      return;
    }

    setIsSubmitting(true);

    try {
      await weightApi.updateGoal(weightInKg);
      router.back();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to update goal weight'));
    } finally {
      setIsSubmitting(false);
    }
  };

  // Calculate predicted BMI for goal weight
  const getGoalBmi = () => {
    const goalWeight = parseFloat(goalWeightInput);
    if (isNaN(goalWeight) || goalWeight <= 0 || !user?.height) return null;

    const weightInKg = unit === 'lb' ? lbToKg(goalWeight) : goalWeight;
    const heightInMeters = user.height / 100;
    const bmi = weightInKg / (heightInMeters * heightInMeters);

    let category = '';
    if (bmi < 18.5) category = 'Underweight';
    else if (bmi < 25) category = 'Normal';
    else if (bmi < 30) category = 'Overweight';
    else if (bmi < 35) category = 'Obese Class I';
    else if (bmi < 40) category = 'Obese Class II';
    else category = 'Obese Class III';

    return { value: Math.round(bmi * 10) / 10, category };
  };

  const goalBmi = getGoalBmi();

  const getWeightDifference = () => {
    const goalWeight = parseFloat(goalWeightInput);
    if (isNaN(goalWeight) || goalWeight <= 0 || currentWeight === null) return null;

    const weightInKg = unit === 'lb' ? lbToKg(goalWeight) : goalWeight;
    const diff = weightInKg - currentWeight;

    return {
      value: Math.abs(diff),
      direction: diff > 0 ? 'gain' : diff < 0 ? 'lose' : 'maintain',
    };
  };

  const weightDiff = getWeightDifference();

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <View style={styles.header}>
            <TouchableOpacity
              onPress={() => router.back()}
              style={styles.closeButton}
              accessibilityLabel="Close"
            >
              <Ionicons name="close" size={24} color={colors.text.primary} />
            </TouchableOpacity>
            <Text style={styles.title}>Weight Goal</Text>
            <View style={styles.placeholder} />
          </View>

          {/* Current Weight Display */}
          {currentWeight !== null && (
            <View style={styles.currentWeightCard}>
              <View style={styles.currentWeightInfo}>
                <Text style={styles.currentWeightLabel}>Current Weight</Text>
                <Text style={styles.currentWeightValue}>
                  {formatWeight(currentWeight, unit, 1)}
                </Text>
              </View>
              <View style={styles.currentWeightIcon}>
                <Ionicons name="scale-outline" size={24} color={colors.primary.main} />
              </View>
            </View>
          )}

          {/* Goal Weight Input */}
          <View style={styles.inputSection}>
            <Text style={styles.sectionLabel}>Goal Weight</Text>
            <View style={styles.weightInputContainer}>
              <TextInput
                style={styles.weightInput}
                value={goalWeightInput}
                onChangeText={setGoalWeightInput}
                placeholder="0.0"
                placeholderTextColor={colors.text.disabled}
                keyboardType="decimal-pad"
                autoFocus
                maxLength={6}
              />
              <View style={styles.unitToggle}>
                <TouchableOpacity
                  style={[styles.unitButton, unit === 'kg' && styles.unitButtonActive]}
                  onPress={() => setUnit('kg')}
                >
                  <Text style={[styles.unitText, unit === 'kg' && styles.unitTextActive]}>kg</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.unitButton, unit === 'lb' && styles.unitButtonActive]}
                  onPress={() => setUnit('lb')}
                >
                  <Text style={[styles.unitText, unit === 'lb' && styles.unitTextActive]}>lb</Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Quick Adjust Buttons */}
            <View style={styles.quickAdjustContainer}>
              {[-5, -1, 1, 5].map((amount) => (
                <TouchableOpacity
                  key={amount}
                  style={styles.quickAdjustButton}
                  onPress={() => {
                    const current = parseFloat(goalWeightInput) || 0;
                    const adjusted = Math.max(0, current + amount);
                    setGoalWeightInput(adjusted.toFixed(1));
                  }}
                >
                  <Text style={styles.quickAdjustText}>
                    {amount > 0 ? '+' : ''}
                    {amount}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Goal Summary */}
          {(weightDiff || goalBmi) && (
            <View style={styles.summarySection}>
              <Text style={styles.sectionLabel}>Goal Summary</Text>

              {/* Weight Difference */}
              {weightDiff && (
                <View style={styles.summaryCard}>
                  <View
                    style={[
                      styles.summaryIconContainer,
                      {
                        backgroundColor:
                          weightDiff.direction === 'lose'
                            ? colors.status.success + '20'
                            : weightDiff.direction === 'gain'
                              ? colors.status.info + '20'
                              : colors.primary.main + '20',
                      },
                    ]}
                  >
                    <Ionicons
                      name={
                        weightDiff.direction === 'lose'
                          ? 'trending-down'
                          : weightDiff.direction === 'gain'
                            ? 'trending-up'
                            : 'remove'
                      }
                      size={24}
                      color={
                        weightDiff.direction === 'lose'
                          ? colors.status.success
                          : weightDiff.direction === 'gain'
                            ? colors.status.info
                            : colors.primary.main
                      }
                    />
                  </View>
                  <View style={styles.summaryInfo}>
                    <Text style={styles.summaryLabel}>
                      {weightDiff.direction === 'maintain'
                        ? 'Maintain current weight'
                        : `To ${weightDiff.direction}`}
                    </Text>
                    {weightDiff.direction !== 'maintain' && (
                      <Text style={styles.summaryValue}>
                        {formatWeight(weightDiff.value, unit, 1)}
                      </Text>
                    )}
                  </View>
                </View>
              )}

              {/* Goal BMI */}
              {goalBmi && (
                <View style={styles.summaryCard}>
                  <View
                    style={[
                      styles.summaryIconContainer,
                      { backgroundColor: getBmiColor(goalBmi.category) + '20' },
                    ]}
                  >
                    <Ionicons name="body-outline" size={24} color={getBmiColor(goalBmi.category)} />
                  </View>
                  <View style={styles.summaryInfo}>
                    <Text style={styles.summaryLabel}>Goal BMI</Text>
                    <View style={styles.bmiValueRow}>
                      <Text style={styles.summaryValue}>{goalBmi.value}</Text>
                      <View
                        style={[
                          styles.bmiCategoryBadge,
                          { backgroundColor: getBmiColor(goalBmi.category) + '20' },
                        ]}
                      >
                        <Text
                          style={[styles.bmiCategoryText, { color: getBmiColor(goalBmi.category) }]}
                        >
                          {goalBmi.category}
                        </Text>
                      </View>
                    </View>
                  </View>
                </View>
              )}
            </View>
          )}

          {/* Tips */}
          <View style={styles.tipsSection}>
            <View style={styles.tipCard}>
              <Ionicons name="information-circle-outline" size={20} color={colors.status.info} />
              <Text style={styles.tipText}>
                A healthy weight loss rate is 0.5-1 kg (1-2 lbs) per week. Set realistic goals for
                sustainable results.
              </Text>
            </View>
          </View>
        </ScrollView>

        {/* Save Button */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleSave}
            disabled={isSubmitting || !goalWeightInput}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={
                isSubmitting || !goalWeightInput
                  ? [colors.border.secondary, colors.border.secondary]
                  : gradients.primary
              }
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.saveButtonGradient}
            >
              {isSubmitting ? (
                <ActivityIndicator color={colors.text.primary} />
              ) : (
                <Text style={styles.saveButtonText}>Set Goal</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
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
  },
  keyboardView: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.lg,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
  },
  closeButton: {
    padding: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  placeholder: {
    width: 32,
  },

  // Current Weight Card
  currentWeightCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  currentWeightInfo: {
    flex: 1,
  },
  currentWeightLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  currentWeightValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  currentWeightIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Input Section
  inputSection: {
    marginBottom: spacing.xl,
  },
  sectionLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.md,
  },
  weightInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  weightInput: {
    fontSize: 56,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    minWidth: 140,
  },
  unitToggle: {
    flexDirection: 'row',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.xs,
    marginLeft: spacing.md,
  },
  unitButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.sm,
  },
  unitButtonActive: {
    backgroundColor: colors.primary.main,
  },
  unitText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  unitTextActive: {
    color: colors.text.primary,
  },

  // Quick Adjust
  quickAdjustContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.md,
  },
  quickAdjustButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  quickAdjustText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },

  // Summary Section
  summarySection: {
    marginBottom: spacing.xl,
  },
  summaryCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  summaryIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  summaryInfo: {
    flex: 1,
  },
  summaryLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  summaryValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  bmiValueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  bmiCategoryBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  bmiCategoryText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },

  // Tips Section
  tipsSection: {
    marginBottom: spacing.xl,
  },
  tipCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: colors.status.info + '10',
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.info + '30',
  },
  tipText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginLeft: spacing.sm,
    lineHeight: 20,
  },

  // Footer
  footer: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  saveButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  saveButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 50,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
});
