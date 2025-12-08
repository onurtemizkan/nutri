import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import DateTimePicker from '@react-native-community/datetimepicker';
import { OnboardingStepLayout } from '@/lib/components/onboarding';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import { useAuth } from '@/lib/context/AuthContext';
import {
  BIOLOGICAL_SEX_OPTIONS,
  ACTIVITY_LEVEL_OPTIONS,
  TOTAL_ONBOARDING_STEPS,
  DEFAULT_STEP1_DATA,
} from '@/lib/onboarding/config';
import { OnboardingStep1Data, BiologicalSex, ActivityLevel } from '@/lib/onboarding/types';
import { colors, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingProfile() {
  const router = useRouter();
  const { user } = useAuth();
  const { saveStep, isLoading, getDraftForStep, updateDraft } = useOnboarding();

  // Form state
  const [name, setName] = useState(user?.name || '');
  const [dateOfBirth, setDateOfBirth] = useState<Date>(new Date(2000, 0, 1));
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [biologicalSex, setBiologicalSex] = useState<BiologicalSex>('PREFER_NOT_TO_SAY');
  const [height, setHeight] = useState('170');
  const [weight, setWeight] = useState('70');
  const [activityLevel, setActivityLevel] = useState<ActivityLevel>('moderate');

  // Load draft data on mount
  useEffect(() => {
    const draft = getDraftForStep<OnboardingStep1Data>(1);
    if (draft) {
      setName(draft.name || user?.name || '');
      if (draft.dateOfBirth) {
        setDateOfBirth(new Date(draft.dateOfBirth));
      }
      setBiologicalSex(draft.biologicalSex || 'PREFER_NOT_TO_SAY');
      setHeight(String(draft.height || 170));
      setWeight(String(draft.currentWeight || 70));
      setActivityLevel(draft.activityLevel || 'moderate');
    }
  }, [getDraftForStep, user?.name]);

  // Save draft when form changes
  useEffect(() => {
    updateDraft(1, {
      name,
      dateOfBirth: dateOfBirth.toISOString().split('T')[0],
      biologicalSex,
      height: parseFloat(height) || 170,
      currentWeight: parseFloat(weight) || 70,
      activityLevel,
    });
  }, [name, dateOfBirth, biologicalSex, height, weight, activityLevel, updateDraft]);

  const isFormValid = () => {
    return (
      name.trim().length > 0 &&
      parseFloat(height) >= 50 &&
      parseFloat(height) <= 300 &&
      parseFloat(weight) >= 20 &&
      parseFloat(weight) <= 500
    );
  };

  const handleNext = async () => {
    if (!isFormValid()) {
      Alert.alert('Invalid Input', 'Please fill in all required fields with valid values.');
      return;
    }

    const data: OnboardingStep1Data = {
      name: name.trim(),
      dateOfBirth: dateOfBirth.toISOString().split('T')[0],
      biologicalSex,
      height: parseFloat(height),
      currentWeight: parseFloat(weight),
      activityLevel,
    };

    try {
      await saveStep(1, data);
      router.push('/onboarding/goals');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to save profile data'));
    }
  };

  const handleBack = () => {
    router.back();
  };

  const handleDateChange = (event: unknown, selectedDate?: Date) => {
    setShowDatePicker(Platform.OS === 'ios');
    if (selectedDate) {
      setDateOfBirth(selectedDate);
    }
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  return (
    <OnboardingStepLayout
      title="Your Profile"
      subtitle="Tell us about yourself so we can personalize your experience"
      currentStep={1}
      totalSteps={TOTAL_ONBOARDING_STEPS}
      onBack={handleBack}
      onNext={handleNext}
      isNextDisabled={!isFormValid()}
      isLoading={isLoading}
      showBack={true}
    >
      {/* Name */}
      <View style={styles.inputGroup}>
        <Text style={styles.label}>Name</Text>
        <TextInput
          style={styles.textInput}
          value={name}
          onChangeText={setName}
          placeholder="Your name"
          placeholderTextColor={colors.text.tertiary}
          autoCapitalize="words"
        />
      </View>

      {/* Date of Birth */}
      <View style={styles.inputGroup}>
        <Text style={styles.label}>Date of Birth</Text>
        <TouchableOpacity
          style={styles.textInput}
          onPress={() => setShowDatePicker(true)}
        >
          <Text style={styles.dateText}>{formatDate(dateOfBirth)}</Text>
        </TouchableOpacity>
        {showDatePicker && (
          <DateTimePicker
            value={dateOfBirth}
            mode="date"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={handleDateChange}
            maximumDate={new Date()}
            minimumDate={new Date(1920, 0, 1)}
          />
        )}
      </View>

      {/* Biological Sex */}
      <View style={styles.inputGroup}>
        <Text style={styles.label}>Biological Sex</Text>
        <Text style={styles.helperText}>Used for health calculations (BMR, etc.)</Text>
        <View style={styles.optionsRow}>
          {BIOLOGICAL_SEX_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.optionButton,
                biologicalSex === option.value && styles.optionButtonSelected,
              ]}
              onPress={() => setBiologicalSex(option.value)}
            >
              <Text
                style={[
                  styles.optionButtonText,
                  biologicalSex === option.value && styles.optionButtonTextSelected,
                ]}
              >
                {option.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Height & Weight */}
      <View style={styles.rowInputs}>
        <View style={[styles.inputGroup, { flex: 1, marginRight: spacing.sm }]}>
          <Text style={styles.label}>Height (cm)</Text>
          <TextInput
            style={styles.textInput}
            value={height}
            onChangeText={setHeight}
            placeholder="170"
            placeholderTextColor={colors.text.tertiary}
            keyboardType="numeric"
          />
        </View>
        <View style={[styles.inputGroup, { flex: 1, marginLeft: spacing.sm }]}>
          <Text style={styles.label}>Weight (kg)</Text>
          <TextInput
            style={styles.textInput}
            value={weight}
            onChangeText={setWeight}
            placeholder="70"
            placeholderTextColor={colors.text.tertiary}
            keyboardType="numeric"
          />
        </View>
      </View>

      {/* Activity Level */}
      <View style={styles.inputGroup}>
        <Text style={styles.label}>Activity Level</Text>
        <View style={styles.activityOptions}>
          {ACTIVITY_LEVEL_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option.value}
              style={[
                styles.activityOption,
                activityLevel === option.value && styles.activityOptionSelected,
              ]}
              onPress={() => setActivityLevel(option.value)}
            >
              <Text
                style={[
                  styles.activityOptionTitle,
                  activityLevel === option.value && styles.activityOptionTitleSelected,
                ]}
              >
                {option.label}
              </Text>
              <Text style={styles.activityOptionDescription}>{option.description}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </OnboardingStepLayout>
  );
}

const styles = StyleSheet.create({
  inputGroup: {
    marginBottom: spacing.lg,
  },
  label: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
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
  },
  dateText: {
    ...typography.body,
    color: colors.text.primary,
  },
  optionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
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
  optionButtonText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  optionButtonTextSelected: {
    color: colors.primary.main,
  },
  rowInputs: {
    flexDirection: 'row',
  },
  activityOptions: {
    gap: spacing.sm,
  },
  activityOption: {
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  activityOptionSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.dark,
  },
  activityOptionTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  activityOptionTitleSelected: {
    color: colors.primary.main,
  },
  activityOptionDescription: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
});
