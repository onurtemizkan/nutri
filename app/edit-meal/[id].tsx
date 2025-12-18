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
import { MealType, UpdateMealInput } from '@/lib/types';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { TimePicker } from '@/lib/components/TimePicker';
import { MicronutrientDisplay } from '@/lib/components/MicronutrientDisplay';
import { Meal } from '@/lib/types';

export default function EditMealScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Loading states
  const [isLoadingMeal, setIsLoadingMeal] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Form state
  const [mealType, setMealType] = useState<MealType>('lunch');
  const [name, setName] = useState('');
  const [consumedAt, setConsumedAt] = useState(new Date());
  const [calories, setCalories] = useState('');
  const [protein, setProtein] = useState('');
  const [carbs, setCarbs] = useState('');
  const [fat, setFat] = useState('');
  const [fiber, setFiber] = useState('');
  const [servingSize, setServingSize] = useState('');
  const [notes, setNotes] = useState('');

  // Store the full meal object for micronutrient display
  const [currentMeal, setCurrentMeal] = useState<Meal | null>(null);

  const mealTypes: MealType[] = ['breakfast', 'lunch', 'dinner', 'snack'];

  // Load meal data
  useEffect(() => {
    const loadMeal = async () => {
      if (!id) {
        showAlert('Error', 'No meal ID provided');
        router.back();
        return;
      }

      try {
        const meal = await mealsApi.getMealById(id);
        setCurrentMeal(meal);
        setMealType(meal.mealType);
        setName(meal.name);
        setConsumedAt(new Date(meal.consumedAt));
        setCalories(meal.calories.toString());
        setProtein(meal.protein.toString());
        setCarbs(meal.carbs.toString());
        setFat(meal.fat.toString());
        setFiber(meal.fiber?.toString() || '');
        setServingSize(meal.servingSize || '');
        setNotes(meal.notes || '');
      } catch (error) {
        showAlert('Error', getErrorMessage(error, 'Failed to load meal'));
        router.back();
      } finally {
        setIsLoadingMeal(false);
      }
    };

    loadMeal();
  }, [id, router]);

  const handleSaveMeal = async () => {
    if (!name || !calories || !protein || !carbs || !fat) {
      showAlert('Error', 'Please fill in all required fields');
      return;
    }

    if (!id) return;

    setIsSaving(true);
    try {
      const mealData: UpdateMealInput = {
        name,
        mealType,
        consumedAt: consumedAt.toISOString(),
        calories: parseFloat(calories),
        protein: parseFloat(protein),
        carbs: parseFloat(carbs),
        fat: parseFloat(fat),
        fiber: fiber ? parseFloat(fiber) : undefined,
        servingSize: servingSize || undefined,
        notes: notes || undefined,
      };

      await mealsApi.updateMeal(id, mealData);
      router.replace('/');
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to update meal'));
      setIsSaving(false);
    }
  };

  const handleDeleteMeal = () => {
    showAlert(
      'Delete Meal',
      `Are you sure you want to delete "${name}"? This action cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            if (!id) return;
            setIsDeleting(true);
            try {
              await mealsApi.deleteMeal(id);
              router.replace('/');
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to delete meal'));
              setIsDeleting(false);
            }
          },
        },
      ]
    );
  };

  const isLoading = isSaving || isDeleting;

  if (isLoadingMeal) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading meal...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="edit-meal-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            disabled={isLoading}
            testID="edit-meal-cancel-button"
            accessibilityLabel="Go back"
            accessibilityRole="button"
            style={styles.backButton}
          >
            <Ionicons name="chevron-back" size={24} color={colors.primary.main} />
            <Text style={styles.cancelButton}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Edit Meal</Text>
          <TouchableOpacity
            onPress={handleSaveMeal}
            disabled={isLoading}
            testID="edit-meal-save-button"
          >
            {isSaving ? (
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
                    disabled={isLoading}
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

            {/* Meal Time */}
            <View style={styles.section}>
              <TimePicker
                value={consumedAt}
                onChange={setConsumedAt}
                label="Meal Time"
                disabled={isLoading}
                testID="edit-meal-time-picker"
              />
            </View>

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
                  testID="edit-meal-name-input"
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
                      testID="edit-meal-calories-input"
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
                      testID="edit-meal-protein-input"
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
                      testID="edit-meal-carbs-input"
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
                      testID="edit-meal-fat-input"
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

            {/* Micronutrients Display */}
            {currentMeal && (
              <View style={styles.section}>
                <MicronutrientDisplay meal={currentMeal} />
              </View>
            )}

            {/* Delete Button */}
            <View style={styles.deleteSection}>
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={handleDeleteMeal}
                disabled={isLoading}
                activeOpacity={0.8}
                testID="edit-meal-delete-button"
              >
                {isDeleting ? (
                  <ActivityIndicator color={colors.status.error} />
                ) : (
                  <>
                    <Ionicons name="trash-outline" size={20} color={colors.status.error} />
                    <Text style={styles.deleteButtonText}>Delete Meal</Text>
                  </>
                )}
              </TouchableOpacity>
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

  // Delete Section
  deleteSection: {
    marginTop: spacing.xl,
    paddingTop: spacing.xl,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.status.error,
    backgroundColor: 'transparent',
  },
  deleteButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.error,
  },
});
