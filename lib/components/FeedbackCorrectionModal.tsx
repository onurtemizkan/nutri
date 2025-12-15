/**
 * FeedbackCorrectionModal
 *
 * A polished modal for users to correct misclassified food items.
 * Features:
 * - Quick select from alternatives
 * - Search through common foods
 * - Add custom description for better learning
 * - Animated and accessible design
 */
import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  TextInput,
  ScrollView,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Animated,
} from 'react-native';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import {
  colors,
  gradients,
  shadows,
  spacing,
  borderRadius,
  typography,
} from '@/lib/theme/colors';
import { foodFeedbackApi, COMMON_FOODS } from '@/lib/api/food-feedback';
import { showAlert } from '@/lib/utils/alert';

interface Alternative {
  name: string;
  confidence: number;
}

interface FeedbackCorrectionModalProps {
  visible: boolean;
  onClose: () => void;
  originalPrediction: string;
  originalConfidence: number;
  alternatives?: Alternative[];
  imageHash: string;
  onFeedbackSubmitted?: () => void;
}

export default function FeedbackCorrectionModal({
  visible,
  onClose,
  originalPrediction,
  originalConfidence,
  alternatives = [],
  imageHash,
  onFeedbackSubmitted,
}: FeedbackCorrectionModalProps) {
  const [selectedFood, setSelectedFood] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  // Animation for success checkmark
  const successScale = useRef(new Animated.Value(0)).current;
  const successOpacity = useRef(new Animated.Value(0)).current;

  // Search results
  const searchResults = useMemo(() => {
    if (!searchQuery || searchQuery.length < 2) {
      return [];
    }
    return foodFeedbackApi.searchFoods(searchQuery);
  }, [searchQuery]);

  // Suggested foods based on alternatives
  const suggestedFoods = useMemo(() => {
    const suggestions = alternatives
      .filter(alt => alt.name.toLowerCase() !== originalPrediction.toLowerCase())
      .slice(0, 4);

    // Add some common foods if we don't have enough alternatives
    if (suggestions.length < 4) {
      const commonToAdd = COMMON_FOODS
        .filter(f =>
          !suggestions.some(s => s.name.toLowerCase() === f.toLowerCase()) &&
          f.toLowerCase() !== originalPrediction.toLowerCase()
        )
        .slice(0, 4 - suggestions.length)
        .map(f => ({ name: f, confidence: 0 }));

      return [...suggestions, ...commonToAdd];
    }

    return suggestions;
  }, [alternatives, originalPrediction]);

  const handleSelectFood = useCallback((food: string) => {
    setSelectedFood(food);
    setShowSearch(false);
    setSearchQuery('');
  }, []);

  // Animate success state and auto-close
  useEffect(() => {
    if (isSuccess) {
      // Trigger success haptic
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

      // Animate checkmark
      Animated.parallel([
        Animated.spring(successScale, {
          toValue: 1,
          friction: 4,
          tension: 100,
          useNativeDriver: true,
        }),
        Animated.timing(successOpacity, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();

      // Auto-close after delay
      const timer = setTimeout(() => {
        handleClose();
      }, 1800);

      return () => clearTimeout(timer);
    }
  }, [isSuccess, successScale, successOpacity]);

  const handleSubmit = useCallback(async () => {
    if (!selectedFood) {
      showAlert('Please Select', 'Please select the correct food item');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await foodFeedbackApi.submitFeedback({
        originalPrediction,
        originalConfidence,
        correctedLabel: selectedFood,
        imageHash,
        alternatives,
        userDescription: description || undefined,
      });

      if (response.success) {
        setIsSubmitting(false);
        setIsSuccess(true);
        onFeedbackSubmitted?.();
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
      showAlert('Error', 'Failed to submit feedback. Please try again.');
      setIsSubmitting(false);
    }
  }, [
    selectedFood,
    originalPrediction,
    originalConfidence,
    imageHash,
    alternatives,
    description,
    onFeedbackSubmitted,
  ]);

  const handleClose = useCallback(() => {
    setSelectedFood(null);
    setSearchQuery('');
    setDescription('');
    setShowSearch(false);
    setIsSuccess(false);
    successScale.setValue(0);
    successOpacity.setValue(0);
    onClose();
  }, [onClose, successScale, successOpacity]);

  const formatFoodName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={handleClose}
    >
      <BlurView intensity={20} tint="dark" style={styles.overlay}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.keyboardAvoid}
        >
          <View style={styles.modalContainer}>
            {/* Success State */}
            {isSuccess ? (
              <View style={styles.successContainer}>
                <Animated.View
                  style={[
                    styles.successIconContainer,
                    {
                      opacity: successOpacity,
                      transform: [{ scale: successScale }],
                    },
                  ]}
                >
                  <LinearGradient
                    colors={[colors.status.success, '#2ecc71']}
                    style={styles.successIconGradient}
                  >
                    <Ionicons name="checkmark" size={48} color="#fff" />
                  </LinearGradient>
                </Animated.View>
                <Animated.Text
                  style={[styles.successTitle, { opacity: successOpacity }]}
                >
                  Thank You!
                </Animated.Text>
                <Animated.Text
                  style={[styles.successSubtitle, { opacity: successOpacity }]}
                >
                  Your feedback helps improve{'\n'}food recognition
                </Animated.Text>
              </View>
            ) : (
              <>
                {/* Header */}
                <View style={styles.header}>
                  <View style={styles.headerHandle} />
                  <TouchableOpacity
                    style={styles.closeButton}
                    onPress={handleClose}
                    hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                  >
                    <Ionicons name="close" size={24} color={colors.text.tertiary} />
                  </TouchableOpacity>
                </View>

                <ScrollView
                  style={styles.content}
                  showsVerticalScrollIndicator={false}
                  keyboardShouldPersistTaps="handled"
                >
                  {/* Title */}
                  <View style={styles.titleSection}>
                    <View style={styles.iconContainer}>
                      <Ionicons
                        name="help-circle"
                        size={32}
                        color={colors.primary.main}
                      />
                    </View>
                    <Text style={styles.title}>Not the right food?</Text>
                    <Text style={styles.subtitle}>
                      Help us improve by selecting the correct item
                    </Text>
                  </View>

              {/* Original Prediction */}
              <View style={styles.originalSection}>
                <Text style={styles.sectionLabel}>We detected:</Text>
                <View style={styles.originalPrediction}>
                  <Text style={styles.originalFood}>
                    {formatFoodName(originalPrediction)}
                  </Text>
                  <Text style={styles.originalConfidence}>
                    {Math.round(originalConfidence * 100)}% confident
                  </Text>
                </View>
              </View>

              {/* Quick Select from Alternatives */}
              {suggestedFoods.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionLabel}>Did you mean:</Text>
                  <View style={styles.alternativesGrid}>
                    {suggestedFoods.map((alt, index) => (
                      <TouchableOpacity
                        key={index}
                        style={[
                          styles.alternativeChip,
                          selectedFood === alt.name && styles.alternativeChipSelected,
                        ]}
                        onPress={() => handleSelectFood(alt.name)}
                        activeOpacity={0.7}
                      >
                        <Text
                          style={[
                            styles.alternativeText,
                            selectedFood === alt.name && styles.alternativeTextSelected,
                          ]}
                        >
                          {formatFoodName(alt.name)}
                        </Text>
                        {selectedFood === alt.name && (
                          <Ionicons
                            name="checkmark-circle"
                            size={16}
                            color={colors.primary.main}
                            style={styles.checkIcon}
                          />
                        )}
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
              )}

              {/* Search Section */}
              <View style={styles.section}>
                <Text style={styles.sectionLabel}>Or search for food:</Text>
                <View style={styles.searchContainer}>
                  <Ionicons
                    name="search"
                    size={20}
                    color={colors.text.tertiary}
                    style={styles.searchIcon}
                  />
                  <TextInput
                    style={styles.searchInput}
                    placeholder="Type to search..."
                    placeholderTextColor={colors.text.disabled}
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                    onFocus={() => setShowSearch(true)}
                    autoCapitalize="none"
                    autoCorrect={false}
                  />
                  {searchQuery.length > 0 && (
                    <TouchableOpacity
                      onPress={() => setSearchQuery('')}
                      style={styles.clearButton}
                    >
                      <Ionicons
                        name="close-circle"
                        size={20}
                        color={colors.text.tertiary}
                      />
                    </TouchableOpacity>
                  )}
                </View>

                {/* Search Results */}
                {showSearch && searchResults.length > 0 && (
                  <View style={styles.searchResults}>
                    {searchResults.map((food, index) => (
                      <TouchableOpacity
                        key={index}
                        style={[
                          styles.searchResultItem,
                          selectedFood === food && styles.searchResultItemSelected,
                        ]}
                        onPress={() => handleSelectFood(food)}
                      >
                        <Text
                          style={[
                            styles.searchResultText,
                            selectedFood === food && styles.searchResultTextSelected,
                          ]}
                        >
                          {formatFoodName(food)}
                        </Text>
                        {selectedFood === food && (
                          <Ionicons
                            name="checkmark"
                            size={18}
                            color={colors.primary.main}
                          />
                        )}
                      </TouchableOpacity>
                    ))}
                  </View>
                )}
              </View>

              {/* Selected Food Display */}
              {selectedFood && (
                <View style={styles.selectedSection}>
                  <View style={styles.selectedBadge}>
                    <Ionicons
                      name="checkmark-circle"
                      size={20}
                      color={colors.status.success}
                    />
                    <Text style={styles.selectedText}>
                      Selected: <Text style={styles.selectedFoodName}>
                        {formatFoodName(selectedFood)}
                      </Text>
                    </Text>
                  </View>
                </View>
              )}

              {/* Optional Description */}
              <View style={styles.section}>
                <Text style={styles.sectionLabel}>
                  Add a description (optional):
                </Text>
                <Text style={styles.descriptionHint}>
                  Describe how the food looks to help improve recognition
                </Text>
                <TextInput
                  style={styles.descriptionInput}
                  placeholder='e.g., "a red ripe tomato on a white plate"'
                  placeholderTextColor={colors.text.disabled}
                  value={description}
                  onChangeText={setDescription}
                  multiline
                  numberOfLines={3}
                  maxLength={200}
                  textAlignVertical="top"
                />
                <Text style={styles.charCount}>
                  {description.length}/200
                </Text>
              </View>
            </ScrollView>

                {/* Submit Button */}
                <View style={styles.footer}>
                  <TouchableOpacity
                    style={[
                      styles.submitButton,
                      !selectedFood && styles.submitButtonDisabled,
                    ]}
                    onPress={handleSubmit}
                    disabled={!selectedFood || isSubmitting}
                    activeOpacity={0.8}
                  >
                    {isSubmitting ? (
                      <ActivityIndicator size="small" color="#fff" />
                    ) : (
                      <LinearGradient
                        colors={selectedFood ? gradients.primary : ['#555', '#444']}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.submitButtonGradient}
                      >
                        <Ionicons
                          name="send"
                          size={18}
                          color="#fff"
                          style={styles.submitIcon}
                        />
                        <Text style={styles.submitButtonText}>Submit Feedback</Text>
                      </LinearGradient>
                    )}
                  </TouchableOpacity>
                </View>
              </>
            )}
          </View>
        </KeyboardAvoidingView>
      </BlurView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  keyboardAvoid: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalContainer: {
    backgroundColor: colors.background.secondary,
    borderTopLeftRadius: borderRadius['2xl'],
    borderTopRightRadius: borderRadius['2xl'],
    maxHeight: '90%',
    ...shadows.lg,
  },
  header: {
    alignItems: 'center',
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  headerHandle: {
    width: 36,
    height: 4,
    backgroundColor: colors.text.disabled,
    borderRadius: 2,
    marginBottom: spacing.sm,
  },
  closeButton: {
    position: 'absolute',
    right: spacing.md,
    top: spacing.md,
    padding: spacing.xs,
  },
  content: {
    paddingHorizontal: spacing.lg,
  },
  titleSection: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
  },
  iconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  title: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  subtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
  originalSection: {
    marginBottom: spacing.lg,
  },
  sectionLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  originalPrediction: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  originalFood: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  originalConfidence: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  section: {
    marginBottom: spacing.lg,
  },
  alternativesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  alternativeChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  alternativeChipSelected: {
    backgroundColor: colors.special.highlight,
    borderColor: colors.primary.main,
  },
  alternativeText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  alternativeTextSelected: {
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  checkIcon: {
    marginLeft: spacing.xs,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    paddingHorizontal: spacing.md,
  },
  searchIcon: {
    marginRight: spacing.sm,
  },
  searchInput: {
    flex: 1,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  clearButton: {
    padding: spacing.xs,
  },
  searchResults: {
    marginTop: spacing.sm,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    maxHeight: 200,
  },
  searchResultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  searchResultItemSelected: {
    backgroundColor: colors.special.highlight,
  },
  searchResultText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  searchResultTextSelected: {
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  selectedSection: {
    marginBottom: spacing.lg,
  },
  selectedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${colors.status.success}20`,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.status.success,
  },
  selectedText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginLeft: spacing.sm,
  },
  selectedFoodName: {
    color: colors.status.success,
    fontWeight: typography.fontWeight.semibold,
  },
  descriptionHint: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginBottom: spacing.sm,
  },
  descriptionInput: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    minHeight: 80,
  },
  charCount: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    textAlign: 'right',
    marginTop: spacing.xs,
  },
  footer: {
    padding: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  submitButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    ...shadows.md,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonGradient: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  submitIcon: {
    marginRight: spacing.sm,
  },
  submitButtonText: {
    color: '#fff',
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  // Success state styles
  successContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing['3xl'],
    paddingHorizontal: spacing.lg,
    minHeight: 280,
  },
  successIconContainer: {
    marginBottom: spacing.lg,
  },
  successIconGradient: {
    width: 88,
    height: 88,
    borderRadius: 44,
    justifyContent: 'center',
    alignItems: 'center',
    ...shadows.lg,
  },
  successTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  successSubtitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: 22,
  },
});
