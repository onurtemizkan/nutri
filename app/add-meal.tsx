import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { mealsApi } from '@/lib/api/meals';
import { CreateMealInput } from '@/lib/types';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { TimePicker } from '@/lib/components/TimePicker';

/**
 * Determines the appropriate meal type based on the current local time
 * - Breakfast: 5:00 AM - 10:59 AM
 * - Lunch: 11:00 AM - 2:59 PM
 * - Dinner: 3:00 PM - 8:59 PM
 * - Snack: 9:00 PM - 4:59 AM
 */
function getMealTypeByTime(): 'breakfast' | 'lunch' | 'dinner' | 'snack' {
  const hour = new Date().getHours();

  if (hour >= 5 && hour < 11) {
    return 'breakfast';
  } else if (hour >= 11 && hour < 15) {
    return 'lunch';
  } else if (hour >= 15 && hour < 21) {
    return 'dinner';
  } else {
    return 'snack';
  }
}

// Micronutrient params type for barcode/AI scan
interface MicronutrientParams {
  sugar?: string;
  saturatedFat?: string;
  transFat?: string;
  cholesterol?: string;
  sodium?: string;
  potassium?: string;
  calcium?: string;
  iron?: string;
  magnesium?: string;
  zinc?: string;
  phosphorus?: string;
  vitaminA?: string;
  vitaminC?: string;
  vitaminD?: string;
  vitaminE?: string;
  vitaminK?: string;
  vitaminB6?: string;
  vitaminB12?: string;
  folate?: string;
  thiamin?: string;
  riboflavin?: string;
  niacin?: string;
}

export default function AddMealScreen() {
  const params = useLocalSearchParams<{
    name?: string;
    calories?: string;
    protein?: string;
    carbs?: string;
    fat?: string;
    fiber?: string;
    servingSize?: string;
    fromScan?: string;
    barcode?: string;
  } & MicronutrientParams>();

  const [mealType, setMealType] = useState<'breakfast' | 'lunch' | 'dinner' | 'snack'>(getMealTypeByTime);
  const [name, setName] = useState('');
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [carbs, setCarbs] = useState('');
  const [fat, setFat] = useState('');
  const [fiber, setFiber] = useState('');
  const [servingSize, setServingSize] = useState('');
  const [notes, setNotes] = useState('');
  const [consumedAt, setConsumedAt] = useState(new Date());
  const [isLoading, setIsLoading] = useState(false);

  // Store micronutrients from scan (hidden state, passed through to API)
  const [micronutrients, setMicronutrients] = useState<Partial<MicronutrientParams>>({});

  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Pre-fill form if coming from food scanner
  useEffect(() => {
    if (params.fromScan === 'true') {
      if (params.name) setName(params.name);
      if (params.calories) setCalories(params.calories);
      if (params.protein) setProtein(params.protein);
      if (params.carbs) setCarbs(params.carbs);
      if (params.fat) setFat(params.fat);
      if (params.fiber) setFiber(params.fiber);
      if (params.servingSize) setServingSize(params.servingSize);

      // Store micronutrients from scan params
      const scannedMicronutrients: Partial<MicronutrientParams> = {};
      if (params.sugar) scannedMicronutrients.sugar = params.sugar;
      if (params.saturatedFat) scannedMicronutrients.saturatedFat = params.saturatedFat;
      if (params.transFat) scannedMicronutrients.transFat = params.transFat;
      if (params.cholesterol) scannedMicronutrients.cholesterol = params.cholesterol;
      if (params.sodium) scannedMicronutrients.sodium = params.sodium;
      if (params.potassium) scannedMicronutrients.potassium = params.potassium;
      if (params.calcium) scannedMicronutrients.calcium = params.calcium;
      if (params.iron) scannedMicronutrients.iron = params.iron;
      if (params.magnesium) scannedMicronutrients.magnesium = params.magnesium;
      if (params.zinc) scannedMicronutrients.zinc = params.zinc;
      if (params.phosphorus) scannedMicronutrients.phosphorus = params.phosphorus;
      if (params.vitaminA) scannedMicronutrients.vitaminA = params.vitaminA;
      if (params.vitaminC) scannedMicronutrients.vitaminC = params.vitaminC;
      if (params.vitaminD) scannedMicronutrients.vitaminD = params.vitaminD;
      if (params.vitaminE) scannedMicronutrients.vitaminE = params.vitaminE;
      if (params.vitaminK) scannedMicronutrients.vitaminK = params.vitaminK;
      if (params.vitaminB6) scannedMicronutrients.vitaminB6 = params.vitaminB6;
      if (params.vitaminB12) scannedMicronutrients.vitaminB12 = params.vitaminB12;
      if (params.folate) scannedMicronutrients.folate = params.folate;
      if (params.thiamin) scannedMicronutrients.thiamin = params.thiamin;
      if (params.riboflavin) scannedMicronutrients.riboflavin = params.riboflavin;
      if (params.niacin) scannedMicronutrients.niacin = params.niacin;

      setMicronutrients(scannedMicronutrients);
    }
  }, [params]);

  const mealTypes: ('breakfast' | 'lunch' | 'dinner' | 'snack')[] = [
    'breakfast',
    'lunch',
    'dinner',
    'snack',
  ];

  const handleSaveMeal = async () => {
    if (!name || !calories || !protein || !carbs || !fat) {
      showAlert('Error', 'Please fill in all required fields');
      return;
    }

    setIsLoading(true);
    try {
      // Parse a micronutrient value to number, returning undefined if invalid
      const parseNutrient = (val: string | undefined): number | undefined => {
        if (!val) return undefined;
        const num = parseFloat(val);
        return isNaN(num) ? undefined : num;
      };

      const mealData: CreateMealInput = {
        name,
        mealType,
        calories: parseFloat(calories),
        protein: parseFloat(protein),
        carbs: parseFloat(carbs),
        fat: parseFloat(fat),
        fiber: fiber ? parseFloat(fiber) : undefined,
        servingSize: servingSize || undefined,
        notes: notes || undefined,
        consumedAt: consumedAt.toISOString(),

        // Micronutrients from scan (if available)
        sugar: parseNutrient(micronutrients.sugar),
        saturatedFat: parseNutrient(micronutrients.saturatedFat),
        transFat: parseNutrient(micronutrients.transFat),
        cholesterol: parseNutrient(micronutrients.cholesterol),
        sodium: parseNutrient(micronutrients.sodium),
        potassium: parseNutrient(micronutrients.potassium),
        calcium: parseNutrient(micronutrients.calcium),
        iron: parseNutrient(micronutrients.iron),
        magnesium: parseNutrient(micronutrients.magnesium),
        zinc: parseNutrient(micronutrients.zinc),
        phosphorus: parseNutrient(micronutrients.phosphorus),
        vitaminA: parseNutrient(micronutrients.vitaminA),
        vitaminC: parseNutrient(micronutrients.vitaminC),
        vitaminD: parseNutrient(micronutrients.vitaminD),
        vitaminE: parseNutrient(micronutrients.vitaminE),
        vitaminK: parseNutrient(micronutrients.vitaminK),
        vitaminB6: parseNutrient(micronutrients.vitaminB6),
        vitaminB12: parseNutrient(micronutrients.vitaminB12),
        folate: parseNutrient(micronutrients.folate),
        thiamin: parseNutrient(micronutrients.thiamin),
        riboflavin: parseNutrient(micronutrients.riboflavin),
        niacin: parseNutrient(micronutrients.niacin),
      };

      await mealsApi.createMeal(mealData);
      // Navigate back to home immediately after success
      router.replace('/');
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to add meal'));
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container} testID="add-meal-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            testID="add-meal-cancel-button"
            accessibilityLabel="Go back"
            accessibilityRole="button"
            style={styles.backButton}
          >
            <Ionicons name="chevron-back" size={24} color={colors.primary.main} />
            <Text style={styles.cancelButton}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Add Meal</Text>
          <TouchableOpacity
            onPress={handleSaveMeal}
            disabled={isLoading}
            testID="add-meal-save-button"
          >
            {isLoading ? (
              <ActivityIndicator color={colors.primary.main} />
            ) : (
              <Text style={styles.saveButton}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={[
            styles.scrollContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.tabletContent
          ]}
        >
          <View style={styles.content}>
            {/* Meal Type Selector */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Meal Type</Text>
              <View style={styles.mealTypeContainer}>
                {mealTypes.map((type) => (
                  <TouchableOpacity
                    key={type}
                    style={styles.mealTypeButton}
                    onPress={() => setMealType(type)}
                    activeOpacity={0.8}
                  >
                    {mealType === type ? (
                      <LinearGradient
                        colors={gradients.primary}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.mealTypeButtonGradient}
                      >
                        <Text
                          style={styles.mealTypeTextActive}
                          numberOfLines={1}
                          adjustsFontSizeToFit
                          minimumFontScale={0.8}
                        >
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </Text>
                      </LinearGradient>
                    ) : (
                      <View style={styles.mealTypeButtonInactive}>
                        <Text
                          style={styles.mealTypeText}
                          numberOfLines={1}
                          adjustsFontSizeToFit
                          minimumFontScale={0.8}
                        >
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                        </Text>
                      </View>
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Time Picker */}
            <TimePicker
              value={consumedAt}
              onChange={setConsumedAt}
              label="Time"
              disabled={isLoading}
              testID="add-meal-time-picker"
            />

            {/* Scan Food Button */}
            <TouchableOpacity
              style={styles.scanButton}
              onPress={() => router.push('/scan-food')}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="add-meal-scan-food-button"
            >
              <Ionicons name="camera" size={24} color={colors.primary.main} />
              <View style={styles.scanButtonText}>
                <Text style={styles.scanButtonTitle}>Scan Food with Camera</Text>
                <Text style={styles.scanButtonSubtitle}>
                  Automatically estimate nutrition using AI
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>

            {/* Barcode Scanner Button */}
            <TouchableOpacity
              style={styles.scanButton}
              onPress={() => router.push('/scan-barcode')}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="add-meal-scan-barcode-button"
            >
              <Ionicons name="barcode-outline" size={24} color={colors.primary.main} />
              <View style={styles.scanButtonText}>
                <Text style={styles.scanButtonTitle}>Scan Product Barcode</Text>
                <Text style={styles.scanButtonSubtitle}>
                  Look up nutrition from 2M+ products
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>

            {params.fromScan === 'true' && (
              <View style={styles.scanBadge}>
                <Ionicons name="checkmark-circle" size={16} color={colors.status.success} />
                <Text style={styles.scanBadgeText}>
                  Values from food scan - you can edit them
                </Text>
              </View>
            )}

            {/* Meal Name */}
            <View style={styles.section}>
              <Text style={styles.label}>Meal Name *</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., Grilled Chicken Salad"
                  placeholderTextColor={colors.text.disabled}
                  value={name}
                  onChangeText={setName}
                  editable={!isLoading}
                  testID="add-meal-name-input"
                />
              </View>
            </View>

            {/* Nutrition Info */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Nutrition Information</Text>

              <View style={styles.row}>
                <View style={styles.halfInput}>
                  <Text style={styles.label}>Calories *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      placeholder="0"
                      placeholderTextColor={colors.text.disabled}
                      value={calories}
                      onChangeText={setCalories}
                      keyboardType="numeric"
                      editable={!isLoading}
                      testID="add-meal-calories-input"
                    />
                  </View>
                </View>

                <View style={styles.halfInput}>
                  <Text style={styles.label}>Protein (g) *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      placeholder="0"
                      placeholderTextColor={colors.text.disabled}
                      value={protein}
                      onChangeText={setProtein}
                      keyboardType="numeric"
                      editable={!isLoading}
                      testID="add-meal-protein-input"
                    />
                  </View>
                </View>
              </View>

              <View style={styles.row}>
                <View style={styles.halfInput}>
                  <Text style={styles.label}>Carbs (g) *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      placeholder="0"
                      placeholderTextColor={colors.text.disabled}
                      value={carbs}
                      onChangeText={setCarbs}
                      keyboardType="numeric"
                      editable={!isLoading}
                      testID="add-meal-carbs-input"
                    />
                  </View>
                </View>

                <View style={styles.halfInput}>
                  <Text style={styles.label}>Fat (g) *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      placeholder="0"
                      placeholderTextColor={colors.text.disabled}
                      value={fat}
                      onChangeText={setFat}
                      keyboardType="numeric"
                      editable={!isLoading}
                      testID="add-meal-fat-input"
                    />
                  </View>
                </View>
              </View>

              <View style={styles.inputContainer}>
                <Text style={styles.label}>Fiber (g)</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={styles.input}
                    placeholder="0"
                    placeholderTextColor={colors.text.disabled}
                    value={fiber}
                    onChangeText={setFiber}
                    keyboardType="numeric"
                    editable={!isLoading}
                  />
                </View>
              </View>

              <View style={styles.inputContainer}>
                <Text style={styles.label}>Serving Size</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={styles.input}
                    placeholder="e.g., 1 plate, 200g"
                    placeholderTextColor={colors.text.disabled}
                    value={servingSize}
                    onChangeText={setServingSize}
                    editable={!isLoading}
                  />
                </View>
              </View>
            </View>

            {/* Notes */}
            <View style={styles.section}>
              <Text style={styles.label}>Notes</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={[styles.input, styles.notesInput]}
                  placeholder="Add any additional notes..."
                  placeholderTextColor={colors.text.disabled}
                  value={notes}
                  onChangeText={setNotes}
                  multiline
                  numberOfLines={4}
                  textAlignVertical="top"
                  editable={!isLoading}
                />
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

  // Header
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
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: -spacing.xs,
  },
  cancelButton: {
    fontSize: typography.fontSize.md,
    color: colors.primary.main,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  saveButton: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Content
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  tabletContent: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  content: {
    paddingTop: spacing.md,
    paddingBottom: spacing['3xl'],
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.3,
  },

  // Meal Type Selector
  mealTypeContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  mealTypeButton: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  mealTypeButtonGradient: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    alignItems: 'center',
    borderRadius: borderRadius.md,
  },
  mealTypeButtonInactive: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  mealTypeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  mealTypeTextActive: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Input
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.3,
  },
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  input: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 48,
  },
  notesInput: {
    height: 100,
    textAlignVertical: 'top',
    paddingTop: spacing.md,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.md,
  },
  halfInput: {
    flex: 1,
  },
  inputContainer: {
    marginBottom: spacing.md,
  },

  // Scan Button
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.highlight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.focus,
  },
  scanButtonText: {
    flex: 1,
    marginLeft: spacing.md,
  },
  scanButtonTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
    marginBottom: spacing.xs,
  },
  scanButtonSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },

  // Scan Badge
  scanBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.highlight,
    padding: spacing.md,
    borderRadius: borderRadius.sm,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.success,
  },
  scanBadgeText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.success,
    marginLeft: spacing.sm,
    fontWeight: typography.fontWeight.medium,
  },
});
