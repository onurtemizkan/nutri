/**
 * UpgradePrompt Component
 *
 * Contextual upgrade prompt shown when free users try to access
 * premium features. Can be displayed as inline, modal, or banner.
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';

interface UpgradePromptProps {
  /** The feature that requires Pro */
  featureName: string;
  /** Description of what the feature does */
  featureDescription?: string;
  /** Display variant */
  variant?: 'inline' | 'banner' | 'modal';
  /** Whether the modal is visible (only for modal variant) */
  visible?: boolean;
  /** Callback when the prompt is dismissed */
  onDismiss?: () => void;
  /** Custom CTA text */
  ctaText?: string;
}

export function UpgradePrompt({
  featureName,
  featureDescription,
  variant = 'inline',
  visible = true,
  onDismiss,
  ctaText = 'Upgrade to Pro',
}: UpgradePromptProps) {
  const router = useRouter();

  const handleUpgrade = () => {
    onDismiss?.();
    router.push('/paywall');
  };

  if (variant === 'modal') {
    return (
      <Modal visible={visible} transparent animationType="fade" onRequestClose={onDismiss}>
        <Pressable style={styles.modalOverlay} onPress={onDismiss}>
          <Pressable style={styles.modalContent} onPress={(e) => e.stopPropagation()}>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={onDismiss}
              accessibilityRole="button"
              accessibilityLabel="Close"
            >
              <Ionicons name="close" size={24} color={colors.text.tertiary} />
            </TouchableOpacity>

            <View style={styles.modalIconContainer}>
              <LinearGradient
                colors={[colors.primary.main, colors.primary.gradient.end]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.modalIcon}
              >
                <Ionicons name="lock-closed" size={28} color={colors.text.primary} />
              </LinearGradient>
            </View>

            <Text style={styles.modalTitle}>Pro Feature</Text>
            <Text style={styles.modalFeatureName}>{featureName}</Text>
            {featureDescription && (
              <Text style={styles.modalDescription}>{featureDescription}</Text>
            )}

            <TouchableOpacity
              style={styles.modalButton}
              onPress={handleUpgrade}
              accessibilityRole="button"
              accessibilityLabel={ctaText}
            >
              <LinearGradient
                colors={[colors.primary.main, colors.primary.gradient.end]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.modalButtonGradient}
              >
                <Text style={styles.modalButtonText}>{ctaText}</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.modalDismissButton}
              onPress={onDismiss}
              accessibilityRole="button"
              accessibilityLabel="Maybe later"
            >
              <Text style={styles.modalDismissText}>Maybe later</Text>
            </TouchableOpacity>
          </Pressable>
        </Pressable>
      </Modal>
    );
  }

  if (variant === 'banner') {
    return (
      <View style={styles.bannerContainer}>
        <View style={styles.bannerContent}>
          <Ionicons name="star" size={20} color={colors.primary.main} />
          <View style={styles.bannerTextContainer}>
            <Text style={styles.bannerTitle}>{featureName}</Text>
            <Text style={styles.bannerSubtitle}>Requires Pro subscription</Text>
          </View>
        </View>
        <TouchableOpacity
          style={styles.bannerButton}
          onPress={handleUpgrade}
          accessibilityRole="button"
          accessibilityLabel="Upgrade"
        >
          <Text style={styles.bannerButtonText}>Upgrade</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Inline variant (default)
  return (
    <View style={styles.inlineContainer}>
      <View style={styles.inlineHeader}>
        <View style={styles.inlineLockIcon}>
          <Ionicons name="lock-closed" size={16} color={colors.primary.main} />
        </View>
        <Text style={styles.inlineTitle}>{featureName}</Text>
      </View>
      {featureDescription && <Text style={styles.inlineDescription}>{featureDescription}</Text>}
      <TouchableOpacity
        style={styles.inlineButton}
        onPress={handleUpgrade}
        accessibilityRole="button"
        accessibilityLabel={ctaText}
      >
        <LinearGradient
          colors={[colors.primary.main, colors.primary.gradient.end]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.inlineButtonGradient}
        >
          <Text style={styles.inlineButtonText}>{ctaText}</Text>
        </LinearGradient>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  // Modal variant
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay.medium,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  modalContent: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    width: '100%',
    maxWidth: 340,
    alignItems: 'center',
    ...shadows.xl,
  },
  modalCloseButton: {
    position: 'absolute',
    top: spacing.md,
    right: spacing.md,
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalIconContainer: {
    marginBottom: spacing.lg,
  },
  modalIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.glow,
  },
  modalTitle: {
    ...typography.caption,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  modalFeatureName: {
    ...typography.h2,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  modalDescription: {
    ...typography.body,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  modalButton: {
    width: '100%',
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  modalButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  modalButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  modalDismissButton: {
    paddingVertical: spacing.sm,
  },
  modalDismissText: {
    ...typography.body,
    color: colors.text.tertiary,
  },

  // Banner variant
  bannerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderWidth: 1,
    borderColor: colors.primary.main,
  },
  bannerContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  bannerTextContainer: {
    marginLeft: spacing.sm,
  },
  bannerTitle: {
    ...typography.bodySmall,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  bannerSubtitle: {
    ...typography.caption,
    color: colors.text.tertiary,
  },
  bannerButton: {
    backgroundColor: colors.primary.main,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  bannerButtonText: {
    ...typography.bodySmall,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Inline variant
  inlineContainer: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  inlineHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  inlineLockIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.special.highlight,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.sm,
  },
  inlineTitle: {
    ...typography.h3,
    color: colors.text.primary,
  },
  inlineDescription: {
    ...typography.body,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },
  inlineButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  inlineButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  inlineButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
});

export default UpgradePrompt;
