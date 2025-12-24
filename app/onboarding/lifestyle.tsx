import { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, TextInput } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Slider from '@react-native-community/slider';
import { OnboardingStepLayout } from '@/lib/components/onboarding';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import {
  NICOTINE_USE_OPTIONS,
  ALCOHOL_USE_OPTIONS,
  WORK_SCHEDULE_OPTIONS,
  TOTAL_ONBOARDING_STEPS,
} from '@/lib/onboarding/config';
import {
  OnboardingStep5Data,
  NicotineUseLevel,
  AlcoholUseLevel,
  WorkSchedule,
} from '@/lib/onboarding/types';
import { colors, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingLifestyle() {
  const router = useRouter();
  const { saveStep, skipStep, isLoading, getDraftForStep, updateDraft } = useOnboarding();

  // Form state
  const [nicotineUse, setNicotineUse] = useState<NicotineUseLevel>('NONE');
  const [alcoholUse, setAlcoholUse] = useState<AlcoholUseLevel>('NONE');
  const [caffeineDaily, setCaffeineDaily] = useState('');
  const [typicalBedtime, setTypicalBedtime] = useState('22:00');
  const [typicalWakeTime, setTypicalWakeTime] = useState('07:00');
  const [sleepQuality, setSleepQuality] = useState(5);
  const [stressLevel, setStressLevel] = useState(5);
  const [workSchedule, setWorkSchedule] = useState<WorkSchedule>('regular');

  // Load draft data on mount
  useEffect(() => {
    const draft = getDraftForStep<OnboardingStep5Data>(5);
    if (draft) {
      setNicotineUse(draft.nicotineUse || 'NONE');
      setAlcoholUse(draft.alcoholUse || 'NONE');
      setCaffeineDaily(draft.caffeineDaily?.toString() || '');
      setTypicalBedtime(draft.typicalBedtime || '22:00');
      setTypicalWakeTime(draft.typicalWakeTime || '07:00');
      setSleepQuality(draft.sleepQuality || 5);
      setStressLevel(draft.stressLevel || 5);
      setWorkSchedule(draft.workSchedule || 'regular');
    }
  }, [getDraftForStep]);

  // Save draft when form changes
  useEffect(() => {
    updateDraft(5, {
      nicotineUse,
      alcoholUse,
      caffeineDaily: caffeineDaily ? parseInt(caffeineDaily, 10) : undefined,
      typicalBedtime,
      typicalWakeTime,
      sleepQuality,
      stressLevel,
      workSchedule,
    });
  }, [nicotineUse, alcoholUse, caffeineDaily, typicalBedtime, typicalWakeTime, sleepQuality, stressLevel, workSchedule, updateDraft]);

  const handleNext = async () => {
    const data: OnboardingStep5Data = {
      nicotineUse,
      alcoholUse,
      caffeineDaily: caffeineDaily ? parseInt(caffeineDaily, 10) : undefined,
      typicalBedtime,
      typicalWakeTime,
      sleepQuality,
      stressLevel,
      workSchedule,
    };

    try {
      await saveStep(5, data);
      router.push('/onboarding/complete');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to save lifestyle data'));
    }
  };

  const handleSkip = async () => {
    try {
      await skipStep(5);
      router.push('/onboarding/complete');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to skip step'));
    }
  };

  const handleBack = () => {
    router.back();
  };

  const getSleepQualityLabel = (value: number): string => {
    if (value <= 2) return 'Poor';
    if (value <= 4) return 'Fair';
    if (value <= 6) return 'Good';
    if (value <= 8) return 'Very Good';
    return 'Excellent';
  };

  const getStressLabel = (value: number): string => {
    if (value <= 2) return 'Very Low';
    if (value <= 4) return 'Low';
    if (value <= 6) return 'Moderate';
    if (value <= 8) return 'High';
    return 'Very High';
  };

  return (
    <OnboardingStepLayout
      title="Lifestyle"
      subtitle="Help us understand your daily habits (optional)"
      currentStep={5}
      totalSteps={TOTAL_ONBOARDING_STEPS}
      onBack={handleBack}
      onNext={handleNext}
      onSkip={handleSkip}
      isLoading={isLoading}
      showBack={true}
      showSkip={true}
    >
      {/* Caffeine Intake */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Daily caffeine intake</Text>
        <View style={styles.caffeineRow}>
          <TextInput
            style={styles.caffeineInput}
            value={caffeineDaily}
            onChangeText={setCaffeineDaily}
            placeholder="0"
            placeholderTextColor={colors.text.tertiary}
            keyboardType="numeric"
          />
          <Text style={styles.caffeineUnit}>mg/day</Text>
        </View>
        <Text style={styles.helperText}>
          A cup of coffee has ~95mg, tea ~50mg, energy drink ~80mg
        </Text>
      </View>

      {/* Alcohol Use */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Alcohol consumption</Text>
        <View style={styles.optionsContainer}>
          {ALCOHOL_USE_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.optionButton, alcoholUse === option.value && styles.optionButtonSelected]}
              onPress={() => setAlcoholUse(option.value)}
            >
              <Text
                style={[
                  styles.optionText,
                  alcoholUse === option.value && styles.optionTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Nicotine Use */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Nicotine use</Text>
        <View style={styles.optionsContainer}>
          {NICOTINE_USE_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.optionButton, nicotineUse === option.value && styles.optionButtonSelected]}
              onPress={() => setNicotineUse(option.value)}
            >
              <Text
                style={[
                  styles.optionText,
                  nicotineUse === option.value && styles.optionTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Sleep Schedule */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Sleep schedule</Text>
        <View style={styles.sleepRow}>
          <View style={styles.sleepInput}>
            <Ionicons name="moon-outline" size={20} color={colors.text.secondary} />
            <TextInput
              style={styles.timeInput}
              value={typicalBedtime}
              onChangeText={setTypicalBedtime}
              placeholder="22:00"
              placeholderTextColor={colors.text.tertiary}
            />
          </View>
          <Ionicons name="arrow-forward" size={20} color={colors.text.tertiary} />
          <View style={styles.sleepInput}>
            <Ionicons name="sunny-outline" size={20} color={colors.text.secondary} />
            <TextInput
              style={styles.timeInput}
              value={typicalWakeTime}
              onChangeText={setTypicalWakeTime}
              placeholder="07:00"
              placeholderTextColor={colors.text.tertiary}
            />
          </View>
        </View>
      </View>

      {/* Sleep Quality */}
      <View style={styles.section}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sectionTitle}>Sleep quality</Text>
          <Text style={styles.sliderValue}>{getSleepQualityLabel(sleepQuality)}</Text>
        </View>
        <Slider
          style={styles.slider}
          minimumValue={1}
          maximumValue={10}
          step={1}
          value={sleepQuality}
          onValueChange={setSleepQuality}
          minimumTrackTintColor={colors.primary.main}
          maximumTrackTintColor={colors.surface.card}
          thumbTintColor={colors.primary.main}
        />
        <View style={styles.sliderLabels}>
          <Text style={styles.sliderLabel}>Poor</Text>
          <Text style={styles.sliderLabel}>Excellent</Text>
        </View>
      </View>

      {/* Stress Level */}
      <View style={styles.section}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sectionTitle}>Typical stress level</Text>
          <Text style={styles.sliderValue}>{getStressLabel(stressLevel)}</Text>
        </View>
        <Slider
          style={styles.slider}
          minimumValue={1}
          maximumValue={10}
          step={1}
          value={stressLevel}
          onValueChange={setStressLevel}
          minimumTrackTintColor={colors.semantic.warning}
          maximumTrackTintColor={colors.surface.card}
          thumbTintColor={colors.semantic.warning}
        />
        <View style={styles.sliderLabels}>
          <Text style={styles.sliderLabel}>Very Low</Text>
          <Text style={styles.sliderLabel}>Very High</Text>
        </View>
      </View>

      {/* Work Schedule */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Work schedule</Text>
        <View style={styles.optionsContainer}>
          {WORK_SCHEDULE_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.optionButton, workSchedule === option.value && styles.optionButtonSelected]}
              onPress={() => setWorkSchedule(option.value)}
            >
              <Text
                style={[
                  styles.optionText,
                  workSchedule === option.value && styles.optionTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </OnboardingStepLayout>
  );
}

const styles = StyleSheet.create({
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  helperText: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  caffeineRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  caffeineInput: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    ...typography.h2,
    color: colors.text.primary,
    width: 100,
    textAlign: 'center',
  },
  caffeineUnit: {
    ...typography.body,
    color: colors.text.secondary,
    marginLeft: spacing.sm,
  },
  optionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  optionButton: {
    backgroundColor: colors.surface.card,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  optionButtonSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.dark,
  },
  optionText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  optionTextSelected: {
    color: colors.text.primary,
  },
  sleepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  sleepInput: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    marginHorizontal: spacing.xs,
  },
  timeInput: {
    ...typography.body,
    color: colors.text.primary,
    marginLeft: spacing.sm,
    flex: 1,
  },
  sliderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  sliderValue: {
    ...typography.bodyBold,
    color: colors.primary.main,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  sliderLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  sliderLabel: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
  },
});
