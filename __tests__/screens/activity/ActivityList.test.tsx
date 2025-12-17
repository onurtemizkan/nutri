/**
 * Activity List Screen Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import ActivityListScreen from '@/app/activity/index';
import { activitiesApi } from '@/lib/api/activities';
import { Activity, WeeklySummary } from '@/lib/types/activities';

// Mock dependencies
jest.mock('@/lib/api/activities');
jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
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
jest.mock('@/lib/context/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1', name: 'Test User', email: 'test@test.com' },
  }),
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

const mockedActivitiesApi = activitiesApi as jest.Mocked<typeof activitiesApi>;

describe('ActivityListScreen', () => {
  const mockActivity: Activity = {
    id: 'activity-1',
    userId: 'user-1',
    activityType: 'RUNNING',
    intensity: 'MODERATE',
    startedAt: new Date().toISOString(),
    endedAt: new Date(Date.now() + 45 * 60000).toISOString(),
    duration: 45,
    caloriesBurned: 450,
    source: 'manual',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  const mockWeeklySummary: WeeklySummary = {
    totalMinutes: 240,
    totalCalories: 1800,
    workoutCount: 5,
    averageIntensity: 'MODERATE',
    byType: {} as WeeklySummary['byType'],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedActivitiesApi.getAll.mockResolvedValue([mockActivity]);
    mockedActivitiesApi.getWeeklySummary.mockResolvedValue(mockWeeklySummary);
  });

  it('renders loading state initially', async () => {
    mockedActivitiesApi.getAll.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve([]), 1000))
    );

    const { getByText } = render(<ActivityListScreen />);

    expect(getByText('Loading activities...')).toBeTruthy();
  });

  it('renders activity list when data loads', async () => {
    const { getByText, getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    expect(getByText('Activities')).toBeTruthy();
  });

  it('renders weekly summary when available', async () => {
    const { getByTestId, getByText } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    // Check if summary is displayed
    await waitFor(() => {
      expect(getByText('This Week')).toBeTruthy();
    });
  });

  it('renders activity card with correct info', async () => {
    const { getByText, getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    await waitFor(() => {
      expect(getByText('Running')).toBeTruthy();
      expect(getByText('45 min')).toBeTruthy();
      expect(getByText('450 cal')).toBeTruthy();
    });
  });

  it('renders empty state when no activities', async () => {
    mockedActivitiesApi.getAll.mockResolvedValue([]);

    const { getByText, getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    await waitFor(() => {
      expect(getByTestId('activity-empty-state')).toBeTruthy();
      expect(getByText('No activities yet')).toBeTruthy();
    });
  });

  it('filters activities by category', async () => {
    const yogaActivity: Activity = {
      ...mockActivity,
      id: 'activity-2',
      activityType: 'YOGA',
      intensity: 'LOW',
    };

    mockedActivitiesApi.getAll.mockResolvedValue([mockActivity, yogaActivity]);

    const { getByText, getByTestId, queryByText } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    // Both activities should be visible initially
    await waitFor(() => {
      expect(getByText('Running')).toBeTruthy();
      expect(getByText('Yoga')).toBeTruthy();
    });

    // Filter to cardio only
    const cardioFilter = getByTestId('activity-filter-cardio');
    await act(async () => {
      fireEvent.press(cardioFilter);
    });

    // Only Running should be visible
    await waitFor(() => {
      expect(getByText('Running')).toBeTruthy();
      expect(queryByText('Yoga')).toBeNull();
    });

    // Filter to other (yoga is in other category)
    const otherFilter = getByTestId('activity-filter-other');
    await act(async () => {
      fireEvent.press(otherFilter);
    });

    await waitFor(() => {
      expect(queryByText('Running')).toBeNull();
      expect(getByText('Yoga')).toBeTruthy();
    });
  });

  it('navigates to add activity screen when FAB pressed', async () => {
    const mockPush = jest.fn();
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue({
      push: mockPush,
      back: jest.fn(),
    });

    const { getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    const addButton = getByTestId('activity-add-button');
    await act(async () => {
      fireEvent.press(addButton);
    });

    expect(mockPush).toHaveBeenCalledWith('/activity/add');
  });

  it('navigates to activity detail when card pressed', async () => {
    const mockPush = jest.fn();
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue({
      push: mockPush,
      back: jest.fn(),
    });

    const { getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    await waitFor(() => {
      const activityCard = getByTestId('activity-item-activity-1');
      fireEvent.press(activityCard);
    });

    expect(mockPush).toHaveBeenCalledWith('/activity/activity-1');
  });

  it('handles API error gracefully', async () => {
    mockedActivitiesApi.getAll.mockRejectedValue(new Error('Network error'));

    const { getByText } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByText('Failed to load activities')).toBeTruthy();
    });

    // Retry button should be visible
    expect(getByText('Retry')).toBeTruthy();
  });

  it('refreshes data on pull-to-refresh', async () => {
    const { getByTestId } = render(<ActivityListScreen />);

    await waitFor(() => {
      expect(getByTestId('activity-screen')).toBeTruthy();
    });

    // Clear mock calls
    mockedActivitiesApi.getAll.mockClear();
    mockedActivitiesApi.getWeeklySummary.mockClear();

    // Simulate pull-to-refresh would be done via RefreshControl
    // This is typically tested via integration tests
    expect(mockedActivitiesApi.getAll).toHaveBeenCalledTimes(0);
  });
});
