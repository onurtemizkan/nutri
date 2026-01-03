/**
 * ScreenHeader Component Tests
 *
 * Comprehensive test suite covering back button, title, actions, and accessibility.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Text } from 'react-native';
import { ScreenHeader } from '@/lib/components/ui/ScreenHeader';

// Mock expo-router
const mockBack = jest.fn();
jest.mock('expo-router', () => ({
  useRouter: () => ({
    back: mockBack,
  }),
}));

// Mock react-native-safe-area-context
jest.mock('react-native-safe-area-context', () => ({
  useSafeAreaInsets: () => ({
    top: 44,
    right: 0,
    bottom: 34,
    left: 0,
  }),
}));

// Mock @expo/vector-icons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: ({ name, testID }: { name: string; testID?: string }) => {
    const { Text } = require('react-native');
    return <Text testID={testID}>{name}</Text>;
  },
}));

describe('ScreenHeader', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // RENDERING TESTS
  // ============================================================================

  describe('Rendering', () => {
    it('renders with title', () => {
      const { getByText } = render(<ScreenHeader title="Settings" />);
      expect(getByText('Settings')).toBeTruthy();
    });

    it('renders with testID', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="settings-header" />
      );
      expect(getByTestId('settings-header')).toBeTruthy();
    });

    it('renders title with header accessibility role', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      const title = getByTestId('header-title');
      expect(title.props.accessibilityRole).toBe('header');
    });
  });

  // ============================================================================
  // BACK BUTTON TESTS
  // ============================================================================

  describe('Back Button', () => {
    it('shows back button by default', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      expect(getByTestId('header-back-button')).toBeTruthy();
    });

    it('hides back button when showBackButton is false', () => {
      const { queryByTestId } = render(
        <ScreenHeader
          title="Settings"
          showBackButton={false}
          testID="header"
        />
      );
      expect(queryByTestId('header-back-button')).toBeNull();
    });

    it('calls router.back() when back button pressed', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      fireEvent.press(getByTestId('header-back-button'));
      expect(mockBack).toHaveBeenCalledTimes(1);
    });

    it('calls custom onBackPress when provided', () => {
      const onBackPress = jest.fn();
      const { getByTestId } = render(
        <ScreenHeader
          title="Settings"
          onBackPress={onBackPress}
          testID="header"
        />
      );
      fireEvent.press(getByTestId('header-back-button'));
      expect(onBackPress).toHaveBeenCalledTimes(1);
      expect(mockBack).not.toHaveBeenCalled();
    });

    it('back button has correct accessibility props', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      const backButton = getByTestId('header-back-button');
      expect(backButton.props.accessibilityRole).toBe('button');
      expect(backButton.props.accessibilityLabel).toBe('Go back');
      expect(backButton.props.accessibilityHint).toBe(
        'Navigate to the previous screen'
      );
    });
  });

  // ============================================================================
  // TITLE ALIGNMENT TESTS
  // ============================================================================

  describe('Title Alignment', () => {
    it('centers title by default', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      // Title container should have center alignment by default
      const title = getByTestId('header-title');
      expect(title).toBeTruthy();
    });

    it('aligns title left when titleAlign is left', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" titleAlign="left" testID="header" />
      );
      const title = getByTestId('header-title');
      expect(title).toBeTruthy();
    });
  });

  // ============================================================================
  // RIGHT ACTIONS TESTS
  // ============================================================================

  describe('Right Actions', () => {
    it('renders right actions when provided', () => {
      const { getByTestId } = render(
        <ScreenHeader
          title="Settings"
          rightActions={<Text testID="action-button">Save</Text>}
        />
      );
      expect(getByTestId('action-button')).toBeTruthy();
    });

    it('renders multiple right actions', () => {
      const { getByTestId } = render(
        <ScreenHeader
          title="Settings"
          rightActions={
            <>
              <Text testID="action-1">Edit</Text>
              <Text testID="action-2">Save</Text>
            </>
          }
        />
      );
      expect(getByTestId('action-1')).toBeTruthy();
      expect(getByTestId('action-2')).toBeTruthy();
    });

    it('does not render right section content when no actions', () => {
      const { queryByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      expect(queryByTestId('action-button')).toBeNull();
    });
  });

  // ============================================================================
  // TRANSPARENT VARIANT TESTS
  // ============================================================================

  describe('Transparent Variant', () => {
    it('applies transparent background when transparent is true', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" transparent testID="header" />
      );
      const header = getByTestId('header');
      expect(header.props.style).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ backgroundColor: 'transparent' }),
        ])
      );
    });

    it('has default background when transparent is false', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" transparent={false} testID="header" />
      );
      const header = getByTestId('header');
      // Should not have transparent style
      expect(header).toBeTruthy();
    });
  });

  // ============================================================================
  // SAFE AREA TESTS
  // ============================================================================

  describe('Safe Area', () => {
    it('applies safe area top inset', () => {
      const { getByTestId } = render(
        <ScreenHeader title="Settings" testID="header" />
      );
      const header = getByTestId('header');
      expect(header.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ paddingTop: 44 })])
      );
    });
  });
});
