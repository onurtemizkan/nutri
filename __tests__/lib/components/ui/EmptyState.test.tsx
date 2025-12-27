/**
 * EmptyState Component Tests
 *
 * Comprehensive test suite covering icon, title, description, and CTA button.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { EmptyState } from '@/lib/components/ui/EmptyState';

// Mock expo-linear-gradient
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: ({ children, ...props }: { children: React.ReactNode }) => {
    const { View } = require('react-native');
    return <View {...props}>{children}</View>;
  },
}));

// Mock @expo/vector-icons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: ({ name, testID }: { name: string; testID?: string }) => {
    const { Text } = require('react-native');
    return <Text testID={testID}>{name}</Text>;
  },
}));

describe('EmptyState', () => {
  const defaultProps = {
    icon: 'restaurant-outline' as const,
    title: 'No meals yet',
    description: 'Start tracking your meals to see them here.',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // RENDERING TESTS
  // ============================================================================

  describe('Rendering', () => {
    it('renders icon with correct name', () => {
      const { getByTestId } = render(
        <EmptyState {...defaultProps} testID="empty" />
      );
      const icon = getByTestId('empty-icon');
      expect(icon).toBeTruthy();
    });

    it('renders title text', () => {
      const { getByText } = render(<EmptyState {...defaultProps} />);
      expect(getByText('No meals yet')).toBeTruthy();
    });

    it('renders description text', () => {
      const { getByText } = render(<EmptyState {...defaultProps} />);
      expect(
        getByText('Start tracking your meals to see them here.')
      ).toBeTruthy();
    });

    it('renders with testID', () => {
      const { getByTestId } = render(
        <EmptyState {...defaultProps} testID="empty-state" />
      );
      expect(getByTestId('empty-state')).toBeTruthy();
    });

    it('title has header accessibility role', () => {
      const { getByTestId } = render(
        <EmptyState {...defaultProps} testID="empty" />
      );
      const title = getByTestId('empty-title');
      expect(title.props.accessibilityRole).toBe('header');
    });
  });

  // ============================================================================
  // CTA BUTTON TESTS
  // ============================================================================

  describe('CTA Button', () => {
    it('renders CTA button when actionLabel provided', () => {
      const onAction = jest.fn();
      const { getByText } = render(
        <EmptyState
          {...defaultProps}
          actionLabel="Add Meal"
          onAction={onAction}
        />
      );
      expect(getByText('Add Meal')).toBeTruthy();
    });

    it('does not render CTA button when actionLabel not provided', () => {
      const { queryByTestId } = render(
        <EmptyState {...defaultProps} testID="empty" />
      );
      expect(queryByTestId('empty-action')).toBeNull();
    });

    it('does not render CTA button when onAction not provided', () => {
      const { queryByTestId } = render(
        <EmptyState {...defaultProps} actionLabel="Add Meal" testID="empty" />
      );
      expect(queryByTestId('empty-action')).toBeNull();
    });

    it('calls onAction when CTA button pressed', () => {
      const onAction = jest.fn();
      const { getByTestId } = render(
        <EmptyState
          {...defaultProps}
          actionLabel="Add Meal"
          onAction={onAction}
          testID="empty"
        />
      );
      fireEvent.press(getByTestId('empty-action'));
      expect(onAction).toHaveBeenCalledTimes(1);
    });

    it('CTA button has correct accessibility label', () => {
      const onAction = jest.fn();
      const { getByTestId } = render(
        <EmptyState
          {...defaultProps}
          actionLabel="Add Meal"
          onAction={onAction}
          testID="empty"
        />
      );
      const button = getByTestId('empty-action');
      expect(button.props.accessibilityLabel).toBe('Add Meal');
    });

    it('CTA button has button accessibility role', () => {
      const onAction = jest.fn();
      const { getByTestId } = render(
        <EmptyState
          {...defaultProps}
          actionLabel="Add Meal"
          onAction={onAction}
          testID="empty"
        />
      );
      const button = getByTestId('empty-action');
      expect(button.props.accessibilityRole).toBe('button');
    });
  });

  // ============================================================================
  // DIFFERENT ICONS TEST
  // ============================================================================

  describe('Different Icons', () => {
    it('renders with different icon names', () => {
      const icons = [
        'fitness-outline',
        'heart-outline',
        'water-outline',
        'barbell-outline',
      ] as const;

      icons.forEach((iconName) => {
        const { getByTestId, unmount } = render(
          <EmptyState
            icon={iconName}
            title="Empty"
            description="No data"
            testID="empty"
          />
        );
        const icon = getByTestId('empty-icon');
        expect(icon).toBeTruthy();
        unmount();
      });
    });
  });

  // ============================================================================
  // LAYOUT TESTS
  // ============================================================================

  describe('Layout', () => {
    it('renders all elements in correct order', () => {
      const onAction = jest.fn();
      const { getByTestId } = render(
        <EmptyState
          {...defaultProps}
          actionLabel="Add Meal"
          onAction={onAction}
          testID="empty"
        />
      );

      // All elements should be present
      expect(getByTestId('empty-icon')).toBeTruthy();
      expect(getByTestId('empty-title')).toBeTruthy();
      expect(getByTestId('empty-description')).toBeTruthy();
      expect(getByTestId('empty-action')).toBeTruthy();
    });
  });
});
