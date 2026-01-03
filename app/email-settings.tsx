/**
 * Email Settings Screen
 * Manage email notification preferences and marketing consent
 */

import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Switch,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { showAlert } from '@/lib/utils/alert';
import { emailApi, EmailPreferences, EmailCategories, EmailFrequency } from '@/lib/api/email';
import { getErrorMessage } from '@/lib/utils/errorHandling';

// Email category config
const CATEGORIES: {
  id: keyof EmailCategories;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  {
    id: 'weekly_reports',
    title: 'Weekly Reports',
    description: 'Your nutrition summary every week',
    icon: 'bar-chart-outline',
  },
  {
    id: 'health_insights',
    title: 'Health Insights',
    description: 'ML-powered nutrition-health correlations',
    icon: 'analytics-outline',
  },
  {
    id: 'tips',
    title: 'Tips & Advice',
    description: 'Personalized nutrition tips',
    icon: 'bulb-outline',
  },
  {
    id: 'features',
    title: 'Feature Updates',
    description: 'New features and improvements',
    icon: 'rocket-outline',
  },
  {
    id: 'promotions',
    title: 'Promotions',
    description: 'Special offers and discounts',
    icon: 'pricetag-outline',
  },
  {
    id: 'newsletter',
    title: 'Newsletter',
    description: 'Monthly nutrition newsletter',
    icon: 'newspaper-outline',
  },
];

// Frequency options
const FREQUENCY_OPTIONS: { value: EmailFrequency; label: string; description: string }[] = [
  { value: 'REALTIME', label: 'Real-time', description: 'Get emails as they happen' },
  { value: 'DAILY_DIGEST', label: 'Daily Digest', description: 'One summary email per day' },
  { value: 'WEEKLY_DIGEST', label: 'Weekly Digest', description: 'One summary email per week' },
];

// Category toggle component
function CategoryToggle({
  category,
  enabled,
  onToggle,
  disabled,
}: {
  category: (typeof CATEGORIES)[number];
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <View style={[styles.categoryItem, disabled && styles.categoryItemDisabled]}>
      <View style={styles.categoryLeft}>
        <View style={[styles.categoryIcon, disabled && styles.categoryIconDisabled]}>
          <Ionicons
            name={category.icon}
            size={20}
            color={disabled ? colors.text.disabled : colors.primary.main}
          />
        </View>
        <View style={styles.categoryInfo}>
          <Text style={[styles.categoryTitle, disabled && styles.textDisabled]}>
            {category.title}
          </Text>
          <Text style={[styles.categoryDescription, disabled && styles.textDisabled]}>
            {category.description}
          </Text>
        </View>
      </View>
      <Switch
        value={enabled}
        onValueChange={onToggle}
        disabled={disabled}
        trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
        thumbColor={colors.text.primary}
        accessibilityLabel={`${enabled ? 'Disable' : 'Enable'} ${category.title}`}
      />
    </View>
  );
}

// Frequency selector component
function FrequencySelector({
  value,
  onChange,
}: {
  value: EmailFrequency;
  onChange: (freq: EmailFrequency) => void;
}) {
  return (
    <View style={styles.frequencyContainer}>
      {FREQUENCY_OPTIONS.map((option) => (
        <TouchableOpacity
          key={option.value}
          style={[styles.frequencyOption, value === option.value && styles.frequencyOptionActive]}
          onPress={() => onChange(option.value)}
          accessibilityLabel={option.label}
          accessibilityRole="radio"
          accessibilityState={{ checked: value === option.value }}
        >
          <View style={styles.frequencyRadio}>
            {value === option.value && <View style={styles.frequencyRadioInner} />}
          </View>
          <View style={styles.frequencyText}>
            <Text
              style={[styles.frequencyLabel, value === option.value && styles.frequencyLabelActive]}
            >
              {option.label}
            </Text>
            <Text style={styles.frequencyDescription}>{option.description}</Text>
          </View>
        </TouchableOpacity>
      ))}
    </View>
  );
}

export default function EmailSettingsScreen() {
  const router = useRouter();
  const { isTablet, isLandscape } = useResponsive();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [preferences, setPreferences] = useState<EmailPreferences | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load preferences
  const loadPreferences = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await emailApi.getPreferences();
      setPreferences(data);
    } catch (err) {
      const message = getErrorMessage(err, 'Failed to load email preferences');
      setError(message);
      showAlert('Error', message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPreferences();
  }, [loadPreferences]);

  // Update a single category
  const updateCategory = async (categoryId: keyof EmailCategories, enabled: boolean) => {
    if (!preferences) return;

    const newCategories = { ...preferences.categories, [categoryId]: enabled };

    try {
      setSaving(true);
      const updated = await emailApi.updatePreferences({ categories: { [categoryId]: enabled } });
      setPreferences({ ...preferences, categories: updated.categories });
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to update preference'));
      // Revert UI
      setPreferences(preferences);
    } finally {
      setSaving(false);
    }
  };

  // Update frequency
  const updateFrequency = async (frequency: EmailFrequency) => {
    if (!preferences) return;

    try {
      setSaving(true);
      await emailApi.updatePreferences({ frequency });
      setPreferences({ ...preferences, frequency });
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to update frequency'));
    } finally {
      setSaving(false);
    }
  };

  // Toggle marketing opt-in
  const toggleMarketingOptIn = async () => {
    if (!preferences) return;

    const newValue = !preferences.marketingOptIn;

    try {
      setSaving(true);
      await emailApi.updatePreferences({ marketingOptIn: newValue });

      // If opting in and not confirmed, request double opt-in
      if (newValue && !preferences.doubleOptInConfirmed) {
        await emailApi.requestDoubleOptIn();
        showAlert(
          'Confirmation Email Sent',
          'Please check your email to confirm your marketing preferences.'
        );
      }

      setPreferences({ ...preferences, marketingOptIn: newValue });
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to update marketing preference'));
    } finally {
      setSaving(false);
    }
  };

  // Resubscribe (for globally unsubscribed users)
  const handleResubscribe = async () => {
    try {
      setSaving(true);
      await emailApi.resubscribe();
      await loadPreferences();
      showAlert('Success', 'You have been resubscribed to emails.');
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to resubscribe'));
    } finally {
      setSaving(false);
    }
  };

  // Responsive container width
  const containerStyle = [
    styles.scrollContent,
    (isTablet || isLandscape) && {
      maxWidth: FORM_MAX_WIDTH,
      alignSelf: 'center' as const,
      width: '100%' as const,
    },
  ];

  if (loading) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            style={styles.backButton}
            accessibilityLabel="Go back"
          >
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Email Settings</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading preferences...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || !preferences) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            style={styles.backButton}
            accessibilityLabel="Go back"
          >
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Email Settings</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={48} color={colors.status.error} />
          <Text style={styles.errorText}>{error || 'Failed to load preferences'}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={loadPreferences}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // If globally unsubscribed, show resubscribe option
  if (preferences.globalUnsubscribed) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            style={styles.backButton}
            accessibilityLabel="Go back"
          >
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Email Settings</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.unsubscribedContainer}>
          <Ionicons name="mail-unread-outline" size={64} color={colors.text.tertiary} />
          <Text style={styles.unsubscribedTitle}>You&apos;re Unsubscribed</Text>
          <Text style={styles.unsubscribedText}>
            You have unsubscribed from all email communications.
          </Text>
          <TouchableOpacity
            style={styles.resubscribeButton}
            onPress={handleResubscribe}
            disabled={saving}
          >
            {saving ? (
              <ActivityIndicator size="small" color={colors.text.inverse} />
            ) : (
              <Text style={styles.resubscribeButtonText}>Resubscribe to Emails</Text>
            )}
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
        >
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Email Settings</Text>
        <View style={styles.headerRight}>
          {saving && <ActivityIndicator size="small" color={colors.primary.main} />}
        </View>
      </View>

      <ScrollView style={styles.scrollView} contentContainerStyle={containerStyle}>
        {/* Marketing Opt-In Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Marketing Emails</Text>
          <View style={styles.marketingCard}>
            <View style={styles.marketingHeader}>
              <View style={styles.marketingIcon}>
                <Ionicons name="mail-outline" size={24} color={colors.primary.main} />
              </View>
              <View style={styles.marketingInfo}>
                <Text style={styles.marketingTitle}>Receive Marketing Emails</Text>
                <Text style={styles.marketingDescription}>
                  Get updates about new features, tips, and promotions
                </Text>
              </View>
              <Switch
                value={preferences.marketingOptIn}
                onValueChange={toggleMarketingOptIn}
                trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
                thumbColor={colors.text.primary}
                accessibilityLabel="Toggle marketing emails"
              />
            </View>
            {preferences.marketingOptIn && !preferences.doubleOptInConfirmed && (
              <View style={styles.confirmationBanner}>
                <Ionicons name="information-circle" size={20} color={colors.status.warning} />
                <Text style={styles.confirmationText}>
                  Check your email to confirm your subscription
                </Text>
              </View>
            )}
            {preferences.marketingOptIn && preferences.doubleOptInConfirmed && (
              <View style={styles.confirmedBanner}>
                <Ionicons name="checkmark-circle" size={20} color={colors.status.success} />
                <Text style={styles.confirmedText}>Email subscription confirmed</Text>
              </View>
            )}
          </View>
        </View>

        {/* Email Frequency Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Email Frequency</Text>
          <Text style={styles.sectionDescription}>
            Choose how often you&apos;d like to receive email updates
          </Text>
          <FrequencySelector value={preferences.frequency} onChange={updateFrequency} />
        </View>

        {/* Categories Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Email Categories</Text>
          <Text style={styles.sectionDescription}>
            Choose which types of emails you want to receive
          </Text>
          <View style={styles.categoriesCard}>
            {CATEGORIES.map((category, index) => (
              <View key={category.id}>
                <CategoryToggle
                  category={category}
                  enabled={preferences.categories[category.id]}
                  onToggle={(enabled) => updateCategory(category.id, enabled)}
                  disabled={saving}
                />
                {index < CATEGORIES.length - 1 && <View style={styles.categoryDivider} />}
              </View>
            ))}
          </View>
        </View>

        {/* Info Section */}
        <View style={styles.infoSection}>
          <Ionicons name="shield-checkmark-outline" size={24} color={colors.text.tertiary} />
          <Text style={styles.infoText}>
            We respect your privacy. You can unsubscribe at any time by clicking the unsubscribe
            link in any email we send.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
    backgroundColor: colors.background.elevated,
  },
  backButton: {
    padding: spacing.xs,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  headerRight: {
    width: 40,
    alignItems: 'flex-end',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
  },
  loadingText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.lg,
  },
  errorText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  retryButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  sectionDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginBottom: spacing.md,
  },
  marketingCard: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    ...shadows.sm,
  },
  marketingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  marketingIcon: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: `${colors.primary.main}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  marketingInfo: {
    flex: 1,
  },
  marketingTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  marketingDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: 2,
  },
  confirmationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.md,
    padding: spacing.sm,
    backgroundColor: `${colors.status.warning}15`,
    borderRadius: borderRadius.sm,
  },
  confirmationText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.warning,
    flex: 1,
  },
  confirmedBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.md,
    padding: spacing.sm,
    backgroundColor: `${colors.status.success}15`,
    borderRadius: borderRadius.sm,
  },
  confirmedText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.success,
    flex: 1,
  },
  frequencyContainer: {
    gap: spacing.sm,
  },
  frequencyOption: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  frequencyOptionActive: {
    borderColor: colors.primary.main,
    backgroundColor: `${colors.primary.main}08`,
  },
  frequencyRadio: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: colors.border.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  frequencyRadioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary.main,
  },
  frequencyText: {
    flex: 1,
  },
  frequencyLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  frequencyLabelActive: {
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  frequencyDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: 2,
  },
  categoriesCard: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.lg,
    padding: spacing.sm,
    ...shadows.sm,
  },
  categoryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.sm,
  },
  categoryItemDisabled: {
    opacity: 0.5,
  },
  categoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    flex: 1,
  },
  categoryIcon: {
    width: 40,
    height: 40,
    borderRadius: borderRadius.sm,
    backgroundColor: `${colors.primary.main}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  categoryIconDisabled: {
    backgroundColor: colors.background.primary,
  },
  categoryInfo: {
    flex: 1,
  },
  categoryTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  categoryDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: 2,
  },
  textDisabled: {
    color: colors.text.disabled,
  },
  categoryDivider: {
    height: 1,
    backgroundColor: colors.border.primary,
    marginHorizontal: spacing.sm,
  },
  infoSection: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    marginTop: spacing.md,
  },
  infoText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    flex: 1,
    lineHeight: 20,
  },
  unsubscribedContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.xl,
  },
  unsubscribedTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  unsubscribedText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  resubscribeButton: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
    marginTop: spacing.md,
    minWidth: 200,
    alignItems: 'center',
  },
  resubscribeButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
});
