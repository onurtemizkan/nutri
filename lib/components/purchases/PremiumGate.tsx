/**
 * PremiumGate Component
 *
 * Wraps content that requires a Pro subscription.
 * Shows upgrade prompt for free users, renders children for Pro users.
 */

import React, { useState, useCallback, ReactNode } from 'react';
import { View, StyleSheet } from 'react-native';

import { useSubscriptionContext } from '@/lib/context/SubscriptionContext';
import { UpgradePrompt } from './UpgradePrompt';
import type { FeatureFlags } from '@/lib/services/purchases/types';

interface PremiumGateProps {
  /** The feature being gated */
  feature: keyof FeatureFlags;
  /** Human-readable feature name for upgrade prompt */
  featureName: string;
  /** Optional description of the feature */
  featureDescription?: string;
  /** Content to render when user has access */
  children: ReactNode;
  /** Fallback content for free users (optional, defaults to UpgradePrompt) */
  fallback?: ReactNode;
  /** Whether to show upgrade prompt in modal form */
  useModal?: boolean;
  /** Callback when upgrade prompt is dismissed */
  onDismiss?: () => void;
  /** Callback when feature access is denied */
  onAccessDenied?: () => void;
  /** Whether to hide content completely (vs blur/disable) */
  hideContent?: boolean;
}

/**
 * Gate content behind a Pro subscription
 *
 * @example
 * ```tsx
 * <PremiumGate feature="mlInsights" featureName="ML Insights">
 *   <MLInsightsComponent />
 * </PremiumGate>
 * ```
 */
export function PremiumGate({
  feature,
  featureName,
  featureDescription,
  children,
  fallback,
  useModal = false,
  onDismiss,
  onAccessDenied,
  hideContent = true,
}: PremiumGateProps) {
  const { isPro, checkFeature, isLoading } = useSubscriptionContext();
  const [showModal, setShowModal] = useState(false);

  const hasAccess = checkFeature(feature);

  // Handle dismiss for modal mode
  const handleDismiss = useCallback(() => {
    setShowModal(false);
    onDismiss?.();
  }, [onDismiss]);

  // If still loading, show nothing to prevent flash
  if (isLoading) {
    return null;
  }

  // If user has access, render children
  if (hasAccess) {
    return <>{children}</>;
  }

  // User doesn't have access
  onAccessDenied?.();

  // If custom fallback provided, use it
  if (fallback) {
    return <>{fallback}</>;
  }

  // Default to UpgradePrompt
  return (
    <View style={styles.container}>
      {!hideContent && <View style={styles.blockedContent}>{children}</View>}
      <UpgradePrompt
        featureName={featureName}
        featureDescription={featureDescription}
        variant={useModal ? 'modal' : 'inline'}
        visible={useModal ? showModal : true}
        onDismiss={handleDismiss}
      />
    </View>
  );
}

/**
 * Hook for imperative feature gating
 *
 * @example
 * ```tsx
 * const { canAccess, showUpgradePrompt, UpgradeModal } = usePremiumGate('mlInsights');
 *
 * const handlePress = () => {
 *   if (!canAccess) {
 *     showUpgradePrompt();
 *     return;
 *   }
 *   // Do premium action
 * };
 *
 * return (
 *   <>
 *     <Button onPress={handlePress}>Use ML Insights</Button>
 *     <UpgradeModal />
 *   </>
 * );
 * ```
 */
export function usePremiumGate(
  feature: keyof FeatureFlags,
  featureName: string,
  featureDescription?: string
) {
  const { checkFeature, isLoading } = useSubscriptionContext();
  const [showModal, setShowModal] = useState(false);

  const canAccess = checkFeature(feature);

  const showUpgradePrompt = useCallback(() => {
    setShowModal(true);
  }, []);

  const hideUpgradePrompt = useCallback(() => {
    setShowModal(false);
  }, []);

  // Modal component to render
  const UpgradeModal = useCallback(
    () => (
      <UpgradePrompt
        featureName={featureName}
        featureDescription={featureDescription}
        variant="modal"
        visible={showModal}
        onDismiss={hideUpgradePrompt}
      />
    ),
    [featureName, featureDescription, showModal, hideUpgradePrompt]
  );

  return {
    canAccess,
    isLoading,
    showUpgradePrompt,
    hideUpgradePrompt,
    UpgradeModal,
  };
}

/**
 * HOC for wrapping entire screens with premium gating
 *
 * @example
 * ```tsx
 * export default withPremiumGate(
 *   MyPremiumScreen,
 *   'advancedAnalytics',
 *   'Advanced Analytics'
 * );
 * ```
 */
export function withPremiumGate<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  feature: keyof FeatureFlags,
  featureName: string,
  featureDescription?: string
) {
  return function PremiumGatedComponent(props: P) {
    return (
      <PremiumGate
        feature={feature}
        featureName={featureName}
        featureDescription={featureDescription}
        hideContent={false}
      >
        <WrappedComponent {...props} />
      </PremiumGate>
    );
  };
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  blockedContent: {
    opacity: 0.3,
    pointerEvents: 'none',
  },
});

export default PremiumGate;
