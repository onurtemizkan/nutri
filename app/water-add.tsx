import { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { waterApi } from '@/lib/api/water';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

// Quick amount presets in ml
const AMOUNT_PRESETS = [100, 150, 200, 250, 300, 350, 400, 500, 750, 1000];

export default function WaterAddScreen() {
  const [amount, setAmount] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const router = useRouter();

  const handleAmountChange = useCallback((text: string) => {
    // Only allow numbers
    const numericText = text.replace(/[^0-9]/g, '');
    setAmount(numericText);
  }, []);

  const handlePresetSelect = useCallback((value: number) => {
    setAmount(value.toString());
  }, []);

  const handleSave = useCallback(async () => {
    const amountNum = parseInt(amount, 10);

    if (!amount || amountNum <= 0) {
      showAlert('Invalid Amount', 'Please enter a valid amount greater than 0');
      return;
    }

    if (amountNum > 5000) {
      showAlert('Invalid Amount', 'Amount cannot exceed 5000ml');
      return;
    }

    setIsSaving(true);
    try {
      await waterApi.createWaterIntake({ amount: amountNum });
      router.back();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to add water intake'));
    } finally {
      setIsSaving(false);
    }
  }, [amount, router]);

  const formatDisplay = (ml: number): string => {
    if (ml >= 1000) {
      return `${(ml / 1000).toFixed(1)}L`;
    }
    return `${ml}ml`;
  };

  const amountNum = parseInt(amount, 10) || 0;

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="close" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Add Water</Text>
          <TouchableOpacity
            style={[styles.saveButton, !amount && styles.saveButtonDisabled]}
            onPress={handleSave}
            disabled={!amount || isSaving}
          >
            {isSaving ? (
              <ActivityIndicator size="small" color={colors.secondary.main} />
            ) : (
              <Text style={[styles.saveButtonText, !amount && styles.saveButtonTextDisabled]}>
                Add
              </Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} keyboardShouldPersistTaps="handled">
          <View style={styles.content}>
            {/* Amount Display */}
            <View style={styles.amountDisplay}>
              <Ionicons name="water" size={40} color={colors.secondary.main} />
              <View style={styles.amountInputContainer}>
                <TextInput
                  style={styles.amountInput}
                  value={amount}
                  onChangeText={handleAmountChange}
                  placeholder="0"
                  placeholderTextColor={colors.text.disabled}
                  keyboardType="number-pad"
                  maxLength={4}
                  autoFocus
                />
                <Text style={styles.amountUnit}>ml</Text>
              </View>
              {amountNum > 0 && (
                <Text style={styles.amountFormatted}>{formatDisplay(amountNum)}</Text>
              )}
            </View>

            {/* Quick Presets */}
            <View style={styles.presetsSection}>
              <Text style={styles.sectionTitle}>Quick Amounts</Text>
              <View style={styles.presetsGrid}>
                {AMOUNT_PRESETS.map((preset) => (
                  <TouchableOpacity
                    key={preset}
                    style={[styles.presetButton, amountNum === preset && styles.presetButtonActive]}
                    onPress={() => handlePresetSelect(preset)}
                    activeOpacity={0.7}
                  >
                    <Text
                      style={[styles.presetLabel, amountNum === preset && styles.presetLabelActive]}
                    >
                      {formatDisplay(preset)}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  keyboardView: {
    flex: 1,
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
  saveButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  saveButtonDisabled: {
    opacity: 0.5,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.secondary.main,
  },
  saveButtonTextDisabled: {
    color: colors.text.disabled,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
  },

  // Amount Display
  amountDisplay: {
    alignItems: 'center',
    paddingVertical: spacing['3xl'],
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  amountInputContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginTop: spacing.md,
  },
  amountInput: {
    fontSize: typography.fontSize['5xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    minWidth: 100,
    paddingHorizontal: spacing.sm,
  },
  amountUnit: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  amountFormatted: {
    fontSize: typography.fontSize.md,
    color: colors.secondary.main,
    marginTop: spacing.xs,
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
    paddingHorizontal: spacing.md,
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
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  presetLabelActive: {
    color: colors.text.primary,
  },
});
