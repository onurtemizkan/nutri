import { useState, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import Slider from '@react-native-community/slider';
import { waterApi } from '@/lib/api/water';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

// Common goal presets in ml
const GOAL_PRESETS = [
  { label: '1.5L', value: 1500 },
  { label: '2L', value: 2000 },
  { label: '2.5L', value: 2500 },
  { label: '3L', value: 3000 },
  { label: '3.5L', value: 3500 },
  { label: '4L', value: 4000 },
];

export default function WaterGoalScreen() {
  const [goalWater, setGoalWater] = useState(2000);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const router = useRouter();

  useEffect(() => {
    loadGoal();
  }, []);

  const loadGoal = async () => {
    try {
      const data = await waterApi.getWaterGoal();
      setGoalWater(data.goalWater);
    } catch (error) {
      console.error('Failed to load water goal:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoalChange = useCallback((value: number) => {
    // Round to nearest 100ml
    const roundedValue = Math.round(value / 100) * 100;
    setGoalWater(roundedValue);
    setHasChanges(true);
  }, []);

  const handlePresetSelect = useCallback((value: number) => {
    setGoalWater(value);
    setHasChanges(true);
  }, []);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      await waterApi.updateWaterGoal(goalWater);
      showAlert('Success', 'Water goal updated successfully', [
        {
          text: 'OK',
          onPress: () => router.back(),
        },
      ]);
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to update water goal'));
    } finally {
      setIsSaving(false);
    }
  }, [goalWater, router]);

  const formatGoal = (ml: number): string => {
    if (ml >= 1000) {
      return `${(ml / 1000).toFixed(1)}L`;
    }
    return `${ml}ml`;
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.secondary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Water Goal</Text>
        <View style={styles.headerSpacer} />
      </View>

      <View style={styles.content}>
        {/* Current Goal Display */}
        <View style={styles.goalDisplay}>
          <Ionicons name="water" size={48} color={colors.secondary.main} />
          <Text style={styles.goalValue}>{formatGoal(goalWater)}</Text>
          <Text style={styles.goalLabel}>Daily Goal</Text>
        </View>

        {/* Slider */}
        <View style={styles.sliderSection}>
          <View style={styles.sliderLabels}>
            <Text style={styles.sliderMinLabel}>500ml</Text>
            <Text style={styles.sliderMaxLabel}>5L</Text>
          </View>
          <Slider
            style={styles.slider}
            minimumValue={500}
            maximumValue={5000}
            step={100}
            value={goalWater}
            onValueChange={handleGoalChange}
            minimumTrackTintColor={colors.secondary.main}
            maximumTrackTintColor={colors.background.elevated}
            thumbTintColor={colors.secondary.main}
          />
        </View>

        {/* Preset Buttons */}
        <View style={styles.presetsSection}>
          <Text style={styles.sectionTitle}>Quick Select</Text>
          <View style={styles.presetsGrid}>
            {GOAL_PRESETS.map((preset) => (
              <TouchableOpacity
                key={preset.value}
                style={[
                  styles.presetButton,
                  goalWater === preset.value && styles.presetButtonActive,
                ]}
                onPress={() => handlePresetSelect(preset.value)}
                activeOpacity={0.7}
              >
                <Text
                  style={[
                    styles.presetLabel,
                    goalWater === preset.value && styles.presetLabelActive,
                  ]}
                >
                  {preset.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Recommendation Info */}
        <View style={styles.infoSection}>
          <Ionicons name="information-circle" size={20} color={colors.text.tertiary} />
          <Text style={styles.infoText}>
            The recommended daily water intake is 2-3 liters for most adults. Factors like activity
            level, climate, and body weight can affect your needs.
          </Text>
        </View>

        {/* Save Button */}
        <TouchableOpacity
          style={[styles.saveButton, !hasChanges && styles.saveButtonDisabled]}
          onPress={handleSave}
          disabled={!hasChanges || isSaving}
          activeOpacity={0.8}
        >
          {isSaving ? (
            <ActivityIndicator size="small" color={colors.text.primary} />
          ) : (
            <Text style={styles.saveButtonText}>Save Goal</Text>
          )}
        </TouchableOpacity>
      </View>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    padding: spacing.xs,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  headerSpacer: {
    width: 40,
  },
  content: {
    flex: 1,
    padding: spacing.lg,
  },

  // Goal Display
  goalDisplay: {
    alignItems: 'center',
    paddingVertical: spacing['3xl'],
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  goalValue: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.md,
  },
  goalLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },

  // Slider Section
  sliderSection: {
    marginBottom: spacing.xl,
  },
  sliderLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
  },
  sliderMinLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  sliderMaxLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  slider: {
    width: '100%',
    height: 40,
  },

  // Presets Section
  presetsSection: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.md,
  },
  presetsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  presetButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  presetButtonActive: {
    backgroundColor: colors.secondary.main,
    borderColor: colors.secondary.main,
  },
  presetLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  presetLabelActive: {
    color: colors.text.primary,
  },

  // Info Section
  infoSection: {
    flexDirection: 'row',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    gap: spacing.sm,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  infoText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: 20,
  },

  // Save Button
  saveButton: {
    backgroundColor: colors.secondary.main,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
    alignItems: 'center',
    ...shadows.sm,
  },
  saveButtonDisabled: {
    backgroundColor: colors.background.elevated,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
});
