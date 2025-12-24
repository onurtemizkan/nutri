import { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, TextInput } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { OnboardingStepLayout } from '@/lib/components/onboarding';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import {
  CHRONIC_CONDITION_OPTIONS,
  SUPPLEMENT_OPTIONS,
  ALLERGY_OPTIONS,
  TOTAL_ONBOARDING_STEPS,
} from '@/lib/onboarding/config';
import {
  OnboardingStep4Data,
  ChronicConditionType,
  SupplementName,
  Allergy,
} from '@/lib/onboarding/types';
import { colors, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingHealthBackground() {
  const router = useRouter();
  const { saveStep, skipStep, isLoading, getDraftForStep, updateDraft } = useOnboarding();

  // Form state
  const [conditions, setConditions] = useState<ChronicConditionType[]>([]);
  const [supplements, setSupplements] = useState<SupplementName[]>([]);
  const [allergies, setAllergies] = useState<Allergy[]>([]);
  const [allergyNotes, setAllergyNotes] = useState('');

  // Load draft data on mount
  useEffect(() => {
    const draft = getDraftForStep<OnboardingStep4Data>(4);
    if (draft) {
      setConditions(draft.chronicConditions?.map((c) => c.type) || []);
      setSupplements(draft.supplements?.map((s) => s.name) || []);
      setAllergies(draft.allergies || []);
      setAllergyNotes(draft.allergyNotes || '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save draft when form changes
  useEffect(() => {
    updateDraft(4, {
      chronicConditions: conditions.map((type) => ({ type })),
      medications: [],
      supplements: supplements.map((name) => ({ name })),
      allergies,
      allergyNotes,
    });
  }, [conditions, supplements, allergies, allergyNotes, updateDraft]);

  const toggleCondition = (condition: ChronicConditionType) => {
    setConditions((prev) =>
      prev.includes(condition) ? prev.filter((c) => c !== condition) : [...prev, condition]
    );
  };

  const toggleSupplement = (supplement: SupplementName) => {
    setSupplements((prev) =>
      prev.includes(supplement) ? prev.filter((s) => s !== supplement) : [...prev, supplement]
    );
  };

  const toggleAllergy = (allergy: Allergy) => {
    setAllergies((prev) =>
      prev.includes(allergy) ? prev.filter((a) => a !== allergy) : [...prev, allergy]
    );
  };

  const handleNext = async () => {
    const data: OnboardingStep4Data = {
      chronicConditions: conditions.map((type) => ({ type })),
      medications: [],
      supplements: supplements.map((name) => ({ name })),
      allergies,
      allergyNotes: allergyNotes || undefined,
    };

    try {
      await saveStep(4, data);
      router.push('/onboarding/lifestyle');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to save health background'));
    }
  };

  const handleSkip = async () => {
    try {
      await skipStep(4);
      router.push('/onboarding/lifestyle');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to skip step'));
    }
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <OnboardingStepLayout
      title="Health Background"
      subtitle="This information helps us provide better insights (optional)"
      currentStep={4}
      totalSteps={TOTAL_ONBOARDING_STEPS}
      onBack={handleBack}
      onNext={handleNext}
      onSkip={handleSkip}
      isLoading={isLoading}
      showBack={true}
      showSkip={true}
    >
      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Ionicons name="information-circle-outline" size={20} color={colors.text.tertiary} />
        <Text style={styles.disclaimerText}>
          This helps us provide personalized insights, not medical advice. Always consult your
          healthcare provider for medical guidance.
        </Text>
      </View>

      {/* Chronic Conditions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Any health conditions?</Text>
        <View style={styles.chipsContainer}>
          {CHRONIC_CONDITION_OPTIONS.slice(0, 10).map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.chip, conditions.includes(option.value) && styles.chipSelected]}
              onPress={() => toggleCondition(option.value)}
            >
              <Text
                style={[
                  styles.chipText,
                  conditions.includes(option.value) && styles.chipTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Supplements */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Taking any supplements?</Text>
        <View style={styles.chipsContainer}>
          {SUPPLEMENT_OPTIONS.slice(0, 12).map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.chip, supplements.includes(option.value) && styles.chipSelected]}
              onPress={() => toggleSupplement(option.value)}
            >
              <Text
                style={[
                  styles.chipText,
                  supplements.includes(option.value) && styles.chipTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Allergies */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Any food allergies?</Text>
        <View style={styles.chipsContainer}>
          {ALLERGY_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[styles.chip, allergies.includes(option.value) && styles.chipSelected]}
              onPress={() => toggleAllergy(option.value)}
            >
              <Text
                style={[
                  styles.chipText,
                  allergies.includes(option.value) && styles.chipTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        {allergies.includes('other') && (
          <TextInput
            style={styles.textInput}
            value={allergyNotes}
            onChangeText={setAllergyNotes}
            placeholder="Please specify other allergies..."
            placeholderTextColor={colors.text.tertiary}
            multiline
          />
        )}
      </View>
    </OnboardingStepLayout>
  );
}

const styles = StyleSheet.create({
  disclaimer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: spacing.md,
    backgroundColor: colors.surface.elevated,
    borderRadius: borderRadius.md,
    marginBottom: spacing.xl,
  },
  disclaimerText: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    flex: 1,
    marginLeft: spacing.sm,
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  chipsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  chip: {
    backgroundColor: colors.surface.card,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  chipSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.dark,
  },
  chipText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  chipTextSelected: {
    color: colors.text.primary,
  },
  textInput: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    ...typography.body,
    color: colors.text.primary,
    marginTop: spacing.sm,
    minHeight: 80,
    textAlignVertical: 'top',
  },
});
