import { useState, useEffect } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity, Alert, Keyboard } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { OnboardingStepLayout } from '@/lib/components/onboarding';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import {
  PRIMARY_GOAL_OPTIONS,
  DIETARY_PREFERENCE_OPTIONS,
  TOTAL_ONBOARDING_STEPS,
} from '@/lib/onboarding/config';
import { OnboardingStep2Data, PrimaryGoal, DietaryPreference } from '@/lib/onboarding/types';
import { colors, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingGoals() {
  const router = useRouter();
  const { saveStep, isLoading, getDraftForStep, updateDraft } = useOnboarding();

  // Form state
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal>('GENERAL_HEALTH');
  const [goalWeight, setGoalWeight] = useState('');
  const [dietaryPreferences, setDietaryPreferences] = useState<DietaryPreference[]>([]);
  const [showCustomMacros, setShowCustomMacros] = useState(false);
  const [customCalories, setCustomCalories] = useState('');
  const [customProtein, setCustomProtein] = useState('');
  const [customCarbs, setCustomCarbs] = useState('');
  const [customFat, setCustomFat] = useState('');

  // Load draft data on mount only
  useEffect(() => {
    const draft = getDraftForStep<OnboardingStep2Data>(2);
    if (draft) {
      setPrimaryGoal(draft.primaryGoal || 'GENERAL_HEALTH');
      setGoalWeight(draft.goalWeight?.toString() || '');
      setDietaryPreferences(draft.dietaryPreferences || []);
      if (draft.customMacros) {
        setShowCustomMacros(true);
        setCustomCalories(draft.customMacros.goalCalories?.toString() || '');
        setCustomProtein(draft.customMacros.goalProtein?.toString() || '');
        setCustomCarbs(draft.customMacros.goalCarbs?.toString() || '');
        setCustomFat(draft.customMacros.goalFat?.toString() || '');
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save draft when form changes
  useEffect(() => {
    const data: Partial<OnboardingStep2Data> = {
      primaryGoal,
      dietaryPreferences,
    };
    if (goalWeight) {
      data.goalWeight = parseFloat(goalWeight);
    }
    if (showCustomMacros) {
      data.customMacros = {};
      if (customCalories) data.customMacros.goalCalories = parseInt(customCalories, 10);
      if (customProtein) data.customMacros.goalProtein = parseFloat(customProtein);
      if (customCarbs) data.customMacros.goalCarbs = parseFloat(customCarbs);
      if (customFat) data.customMacros.goalFat = parseFloat(customFat);
    }
    updateDraft(2, data);
  }, [primaryGoal, goalWeight, dietaryPreferences, showCustomMacros, customCalories, customProtein, customCarbs, customFat, updateDraft]);

  const toggleDietaryPreference = (pref: DietaryPreference) => {
    if (pref === 'none') {
      setDietaryPreferences(['none']);
    } else {
      setDietaryPreferences((prev) => {
        const filtered = prev.filter((p) => p !== 'none');
        if (filtered.includes(pref)) {
          return filtered.filter((p) => p !== pref);
        }
        return [...filtered, pref];
      });
    }
  };

  const handleNext = async () => {
    const data: OnboardingStep2Data = {
      primaryGoal,
      dietaryPreferences,
    };

    if (goalWeight) {
      data.goalWeight = parseFloat(goalWeight);
    }

    if (showCustomMacros) {
      data.customMacros = {};
      if (customCalories) data.customMacros.goalCalories = parseInt(customCalories, 10);
      if (customProtein) data.customMacros.goalProtein = parseFloat(customProtein);
      if (customCarbs) data.customMacros.goalCarbs = parseFloat(customCarbs);
      if (customFat) data.customMacros.goalFat = parseFloat(customFat);
    }

    try {
      await saveStep(2, data);
      router.push('/onboarding/permissions');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to save goals'));
    }
  };

  const handleBack = () => {
    router.back();
  };

  const needsGoalWeight = ['WEIGHT_LOSS', 'MUSCLE_GAIN'].includes(primaryGoal);

  return (
    <OnboardingStepLayout
      title="Your Goals"
      subtitle="What do you want to achieve with Nutri?"
      currentStep={2}
      totalSteps={TOTAL_ONBOARDING_STEPS}
      onBack={handleBack}
      onNext={handleNext}
      isLoading={isLoading}
      showBack={true}
    >
      {/* Primary Goal */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Primary Goal</Text>
        <View style={styles.goalGrid}>
          {PRIMARY_GOAL_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.goalCard,
                primaryGoal === option.value && styles.goalCardSelected,
              ]}
              onPress={() => setPrimaryGoal(option.value)}
            >
              <Ionicons
                name={option.icon as keyof typeof Ionicons.glyphMap}
                size={28}
                color={primaryGoal === option.value ? colors.text.primary : colors.text.secondary}
              />
              <Text
                style={[
                  styles.goalCardText,
                  primaryGoal === option.value && styles.goalCardTextSelected,
                ]}
                numberOfLines={2}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Goal Weight (conditional) */}
      {needsGoalWeight && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Goal Weight (kg)</Text>
          <TextInput
            style={styles.textInput}
            value={goalWeight}
            onChangeText={setGoalWeight}
            placeholder="Enter your goal weight"
            placeholderTextColor={colors.text.tertiary}
            keyboardType="numeric"
          />
        </View>
      )}

      {/* Dietary Preferences */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Dietary Preferences</Text>
        <Text style={styles.helperText}>Select all that apply</Text>
        <View style={styles.preferencesGrid}>
          {DIETARY_PREFERENCE_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.preferenceChip,
                dietaryPreferences.includes(option.value) && styles.preferenceChipSelected,
              ]}
              onPress={() => toggleDietaryPreference(option.value)}
            >
              <Text
                style={[
                  styles.preferenceChipText,
                  dietaryPreferences.includes(option.value) && styles.preferenceChipTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Custom Macros Toggle */}
      <TouchableOpacity
        style={styles.customMacrosToggle}
        onPress={() => {
          Keyboard.dismiss();
          setShowCustomMacros(!showCustomMacros);
        }}
      >
        <Text style={styles.customMacrosToggleText}>
          {showCustomMacros ? 'Use auto-calculated macros' : 'Set custom macro targets'}
        </Text>
        <Ionicons
          name={showCustomMacros ? 'chevron-up' : 'chevron-down'}
          size={20}
          color={colors.primary.main}
        />
      </TouchableOpacity>

      {/* Custom Macros (conditional) */}
      {showCustomMacros && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Custom Macro Targets</Text>
          <View style={styles.macrosGrid}>
            <View style={styles.macroInput}>
              <Text style={styles.macroLabel}>Calories</Text>
              <TextInput
                style={styles.macroTextInput}
                value={customCalories}
                onChangeText={setCustomCalories}
                placeholder="2000"
                placeholderTextColor={colors.text.tertiary}
                keyboardType="numeric"
              />
            </View>
            <View style={styles.macroInput}>
              <Text style={styles.macroLabel}>Protein (g)</Text>
              <TextInput
                style={styles.macroTextInput}
                value={customProtein}
                onChangeText={setCustomProtein}
                placeholder="150"
                placeholderTextColor={colors.text.tertiary}
                keyboardType="numeric"
              />
            </View>
            <View style={styles.macroInput}>
              <Text style={styles.macroLabel}>Carbs (g)</Text>
              <TextInput
                style={styles.macroTextInput}
                value={customCarbs}
                onChangeText={setCustomCarbs}
                placeholder="200"
                placeholderTextColor={colors.text.tertiary}
                keyboardType="numeric"
              />
            </View>
            <View style={styles.macroInput}>
              <Text style={styles.macroLabel}>Fat (g)</Text>
              <TextInput
                style={styles.macroTextInput}
                value={customFat}
                onChangeText={setCustomFat}
                placeholder="65"
                placeholderTextColor={colors.text.tertiary}
                keyboardType="numeric"
              />
            </View>
          </View>
        </View>
      )}
    </OnboardingStepLayout>
  );
}

const styles = StyleSheet.create({
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.h3,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  helperText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  textInput: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    ...typography.body,
    color: colors.text.primary,
    textAlignVertical: 'center',
  },
  goalGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  goalCard: {
    width: '31%',
    backgroundColor: colors.surface.card,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  goalCardSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.dark,
  },
  goalCardText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginTop: spacing.xs,
    textAlign: 'center',
  },
  goalCardTextSelected: {
    color: colors.text.primary,
  },
  preferencesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  preferenceChip: {
    backgroundColor: colors.surface.card,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  preferenceChipSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.dark,
  },
  preferenceChipText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  preferenceChipTextSelected: {
    color: colors.text.primary,
  },
  customMacrosToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
    marginBottom: spacing.md,
  },
  customMacrosToggleText: {
    ...typography.bodySmall,
    color: colors.primary.main,
    marginRight: spacing.xs,
  },
  macrosGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  macroInput: {
    width: '48%',
  },
  macroLabel: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginBottom: spacing.xs,
  },
  macroTextInput: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    ...typography.body,
    color: colors.text.primary,
  },
});
