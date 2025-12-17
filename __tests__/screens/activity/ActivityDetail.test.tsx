/**
 * Activity Detail Screen Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { Alert } from 'react-native';
import ActivityDetailScreen from '@/app/activity/[id]';
import { activitiesApi } from '@/lib/api/activities';
import { Activity } from '@/lib/types/activities';

// Mock dependencies
jest.mock('@/lib/api/activities');
jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
  useLocalSearchParams: () => ({ id: 'activity-1' }),
}));
jest.mock('@react-navigation/native', () => ({
  useFocusEffect: (callback: () => void) => {
    // Execute the callback immediately in tests
    const { useEffect } = require('react');
    useEffect(() => {
      callback();
    }, []);
  },
}));
jest.mock('@/hooks/useResponsive', () => ({
  useResponsive: () => ({
    isTablet: false,
    isLandscape: false,
    width: 390,
    getResponsiveValue: (values: Record<string, unknown>) => values.default || values.medium,
  }),
}));
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: 'LinearGradient',
}));

// Mock Alert
jest.spyOn(Alert, 'alert');

const mockedActivitiesApi = activitiesApi as jest.Mocked<typeof activitiesApi>;

describe('ActivityDetailScreen', () => {
  const mockActivity: Activity = {
    id: 'activity-1',
    userId: 'user-1',
    activityType: 'RUNNING',
    intensity: 'HIGH',
    startedAt: '2024-01-15T08:00:00Z',
    endedAt: '2024-01-15T08:45:00Z',
    duration: 45,
    caloriesBurned: 500,
    averageHeartRate: 155,
    maxHeartRate: 180,
    distance: 5500,
    steps: 6500,
    source: 'manual',
    notes: 'Morning tempo run',
    createdAt: '2024-01-15T08:45:00Z',
    updatedAt: '2024-01-15T08:45:00Z',
  };

  const mockRouter = {
    push: jest.fn(),
    back: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue(mockRouter);
    mockedActivitiesApi.getById.mockResolvedValue(mockActivity);
    mockedActivitiesApi.delete.mockResolvedValue(undefined);
  });

  it('renders loading state initially', async () => {
    mockedActivitiesApi.getById.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockActivity), 1000))
    );

    const { getByText } = render(<ActivityDetailScreen />);

    expect(getByText('Loading activity...')).toBeTruthy();
  });

  it('renders activity details when loaded', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    // Check activity type is displayed
    expect(getByText('Running')).toBeTruthy();
    expect(getByText('High')).toBeTruthy();
  });

  it('displays all activity stats', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    // Check duration
    expect(getByText('45 min')).toBeTruthy();

    // Check calories
    expect(getByText('500')).toBeTruthy();

    // Check distance
    expect(getByText('5.50 km')).toBeTruthy();

    // Check steps
    expect(getByText('6,500')).toBeTruthy();
  });

  it('displays heart rate data', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    expect(getByText('Heart Rate')).toBeTruthy();
    expect(getByText('155 bpm')).toBeTruthy(); // Average
    expect(getByText('180 bpm')).toBeTruthy(); // Max
  });

  it('displays notes when available', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    expect(getByText('Notes')).toBeTruthy();
    expect(getByText('Morning tempo run')).toBeTruthy();
  });

  it('displays source information', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    expect(getByText('Source')).toBeTruthy();
    expect(getByText('Manual Entry')).toBeTruthy();
  });

  it('navigates back when back button pressed', async () => {
    const { getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    const backButton = getByTestId('activity-detail-back-button');
    await act(async () => {
      fireEvent.press(backButton);
    });

    expect(mockRouter.back).toHaveBeenCalled();
  });

  it('navigates to edit screen when edit button pressed', async () => {
    const { getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    const editButton = getByTestId('activity-edit-button');
    await act(async () => {
      fireEvent.press(editButton);
    });

    expect(mockRouter.push).toHaveBeenCalledWith({
      pathname: '/activity/add',
      params: { editId: 'activity-1' },
    });
  });

  it('shows confirmation dialog when delete button pressed', async () => {
    const { getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    const deleteButton = getByTestId('activity-delete-button');
    await act(async () => {
      fireEvent.press(deleteButton);
    });

    expect(Alert.alert).toHaveBeenCalledWith(
      'Delete Activity',
      'Are you sure you want to delete this activity? This action cannot be undone.',
      expect.arrayContaining([
        expect.objectContaining({ text: 'Cancel' }),
        expect.objectContaining({ text: 'Delete', style: 'destructive' }),
      ])
    );
  });

  it('deletes activity and navigates back on confirm', async () => {
    // Mock Alert to auto-confirm delete
    (Alert.alert as jest.Mock).mockImplementation((title, message, buttons) => {
      const deleteButton = buttons?.find((b: { text: string }) => b.text === 'Delete');
      if (deleteButton?.onPress) {
        deleteButton.onPress();
      }
    });

    const { getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    const deleteButton = getByTestId('activity-delete-button');
    await act(async () => {
      fireEvent.press(deleteButton);
    });

    await waitFor(() => {
      expect(mockedActivitiesApi.delete).toHaveBeenCalledWith('activity-1');
    });

    expect(mockRouter.back).toHaveBeenCalled();
  });

  it('shows error alert on delete failure', async () => {
    mockedActivitiesApi.delete.mockRejectedValue(new Error('Network error'));

    // Mock Alert to auto-confirm delete
    (Alert.alert as jest.Mock).mockImplementation((title, message, buttons) => {
      const deleteButton = buttons?.find((b: { text: string }) => b.text === 'Delete');
      if (deleteButton?.onPress) {
        deleteButton.onPress();
      }
    });

    const { getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    const deleteButton = getByTestId('activity-delete-button');
    await act(async () => {
      fireEvent.press(deleteButton);
    });

    await waitFor(() => {
      expect(Alert.alert).toHaveBeenCalledWith(
        'Error',
        'Failed to delete activity. Please try again.'
      );
    });
  });

  it('handles activity not found', async () => {
    mockedActivitiesApi.getById.mockRejectedValue({
      response: { status: 404 },
    });

    const { getByText } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByText('Failed to load activity')).toBeTruthy();
    });
  });

  it('displays correct date and time', async () => {
    const { getByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    // Check time section exists
    expect(getByText('Time')).toBeTruthy();
    expect(getByText('Started')).toBeTruthy();
    expect(getByText('Ended')).toBeTruthy();
  });

  it('handles activity without optional fields', async () => {
    const minimalActivity: Activity = {
      id: 'activity-2',
      userId: 'user-1',
      activityType: 'YOGA',
      intensity: 'LOW',
      startedAt: '2024-01-15T18:00:00Z',
      endedAt: '2024-01-15T18:30:00Z',
      duration: 30,
      source: 'manual',
      createdAt: '2024-01-15T18:30:00Z',
      updatedAt: '2024-01-15T18:30:00Z',
    };

    mockedActivitiesApi.getById.mockResolvedValue(minimalActivity);

    const { getByText, queryByText, getByTestId } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-detail-screen')).toBeTruthy();
    });

    // Activity type and basic info should show
    expect(getByText('Yoga')).toBeTruthy();
    expect(getByText('30 min')).toBeTruthy();

    // Optional fields should not be visible
    expect(queryByText('Heart Rate')).toBeNull();
    expect(queryByText('Notes')).toBeNull();
  });

  it('shows retry button on error', async () => {
    mockedActivitiesApi.getById.mockRejectedValue(new Error('Network error'));

    const { getByText } = render(<ActivityDetailScreen />);

    await waitFor(() => {
      expect(getByText('Retry')).toBeTruthy();
    });

    // Clear mock
    mockedActivitiesApi.getById.mockClear();
    mockedActivitiesApi.getById.mockResolvedValue(mockActivity);

    // Tap retry
    const retryButton = getByText('Retry');
    await act(async () => {
      fireEvent.press(retryButton);
    });

    await waitFor(() => {
      expect(mockedActivitiesApi.getById).toHaveBeenCalled();
    });
  });
});
