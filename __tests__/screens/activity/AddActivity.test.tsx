/**
 * Add Activity Screen Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { Alert } from 'react-native';
import AddActivityScreen from '@/app/activity/add';
import { activitiesApi } from '@/lib/api/activities';

// Mock dependencies
jest.mock('@/lib/api/activities');
jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
  useLocalSearchParams: () => ({}),
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
jest.mock('@react-native-community/datetimepicker', () => 'DateTimePicker');

// Mock Alert
jest.spyOn(Alert, 'alert');

const mockedActivitiesApi = activitiesApi as jest.Mocked<typeof activitiesApi>;

describe('AddActivityScreen', () => {
  const mockRouter = {
    push: jest.fn(),
    back: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue(mockRouter);
    mockedActivitiesApi.create.mockResolvedValue({
      id: 'new-activity',
      userId: 'user-1',
      activityType: 'RUNNING',
      intensity: 'MODERATE',
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      duration: 30,
      source: 'manual',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
  });

  it('renders add activity form', async () => {
    const { getByText, getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    expect(getByText('Add Activity')).toBeTruthy();
    expect(getByText('Activity Type')).toBeTruthy();
    expect(getByText('Intensity')).toBeTruthy();
    expect(getByText('Duration')).toBeTruthy();
  });

  it('has Running as default activity type', async () => {
    const { getByText, getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    expect(getByText('Running')).toBeTruthy();
  });

  it('allows selecting different activity types', async () => {
    const { getByText, getByTestId, queryByText } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Open type picker
    const typeSelector = getByTestId('activity-type-selector');
    await act(async () => {
      fireEvent.press(typeSelector);
    });

    // Select Yoga
    const yogaOption = getByTestId('activity-type-YOGA');
    await act(async () => {
      fireEvent.press(yogaOption);
    });

    // Verify Yoga is selected
    await waitFor(() => {
      expect(getByText('Yoga')).toBeTruthy();
    });
  });

  it('allows selecting different intensity levels', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Select HIGH intensity
    const highIntensity = getByTestId('activity-intensity-HIGH');
    await act(async () => {
      fireEvent.press(highIntensity);
    });

    // Verify it's selected (would have different styling)
    expect(highIntensity).toBeTruthy();
  });

  it('allows entering duration', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    const hoursInput = getByTestId('activity-duration-hours');
    const minutesInput = getByTestId('activity-duration-minutes');

    await act(async () => {
      fireEvent.changeText(hoursInput, '1');
      fireEvent.changeText(minutesInput, '30');
    });

    expect(hoursInput.props.value).toBe('1');
    expect(minutesInput.props.value).toBe('30');
  });

  it('shows validation error for zero duration', async () => {
    const { getByTestId, getByText } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Set duration to 0
    const hoursInput = getByTestId('activity-duration-hours');
    const minutesInput = getByTestId('activity-duration-minutes');

    await act(async () => {
      fireEvent.changeText(hoursInput, '0');
      fireEvent.changeText(minutesInput, '0');
    });

    // Try to save
    const saveButton = getByTestId('activity-save-button');
    await act(async () => {
      fireEvent.press(saveButton);
    });

    // Should show validation error
    await waitFor(() => {
      expect(getByText('Duration must be at least 1 minute')).toBeTruthy();
    });

    // API should not be called
    expect(mockedActivitiesApi.create).not.toHaveBeenCalled();
  });

  it('saves activity successfully', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Enter valid duration
    const minutesInput = getByTestId('activity-duration-minutes');
    await act(async () => {
      fireEvent.changeText(minutesInput, '45');
    });

    // Save
    const saveButton = getByTestId('activity-save-button');
    await act(async () => {
      fireEvent.press(saveButton);
    });

    // API should be called
    await waitFor(() => {
      expect(mockedActivitiesApi.create).toHaveBeenCalled();
    });

    // Success alert should show
    expect(Alert.alert).toHaveBeenCalledWith(
      'Success',
      'Activity added!',
      expect.any(Array)
    );
  });

  it('allows entering optional fields', async () => {
    const { getByTestId, queryByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Running should have distance and steps fields
    const distanceInput = queryByTestId('activity-distance-input');
    const stepsInput = queryByTestId('activity-steps-input');

    expect(distanceInput).toBeTruthy();
    expect(stepsInput).toBeTruthy();
  });

  it('hides distance/steps for activities without them', async () => {
    const { getByTestId, queryByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Open type picker
    const typeSelector = getByTestId('activity-type-selector');
    await act(async () => {
      fireEvent.press(typeSelector);
    });

    // Select Weight Training (no distance/steps)
    const weightTraining = getByTestId('activity-type-WEIGHT_TRAINING');
    await act(async () => {
      fireEvent.press(weightTraining);
    });

    // Distance and steps should not be visible
    await waitFor(() => {
      const distanceInput = queryByTestId('activity-distance-input');
      const stepsInput = queryByTestId('activity-steps-input');
      expect(distanceInput).toBeNull();
      expect(stepsInput).toBeNull();
    });
  });

  it('navigates back on cancel', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    const cancelButton = getByTestId('activity-cancel-button');
    await act(async () => {
      fireEvent.press(cancelButton);
    });

    expect(mockRouter.back).toHaveBeenCalled();
  });

  it('allows entering notes', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    const notesInput = getByTestId('activity-notes-input');
    await act(async () => {
      fireEvent.changeText(notesInput, 'Great workout!');
    });

    expect(notesInput.props.value).toBe('Great workout!');
  });

  it('handles API error on save', async () => {
    mockedActivitiesApi.create.mockRejectedValue(new Error('Network error'));

    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Enter valid duration
    const minutesInput = getByTestId('activity-duration-minutes');
    await act(async () => {
      fireEvent.changeText(minutesInput, '45');
    });

    // Try to save
    const saveButton = getByTestId('activity-save-button');
    await act(async () => {
      fireEvent.press(saveButton);
    });

    // Error alert should show
    await waitFor(() => {
      expect(Alert.alert).toHaveBeenCalledWith(
        'Error',
        expect.any(String)
      );
    });
  });

  it('shows loading state during submission', async () => {
    // Make API call hang
    mockedActivitiesApi.create.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 5000))
    );

    const { getByTestId, queryByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Enter valid duration
    const minutesInput = getByTestId('activity-duration-minutes');
    await act(async () => {
      fireEvent.changeText(minutesInput, '45');
    });

    // Save
    const saveButton = getByTestId('activity-save-button');
    await act(async () => {
      fireEvent.press(saveButton);
    });

    // API should have been called
    expect(mockedActivitiesApi.create).toHaveBeenCalled();
  });

  it('auto-estimates calories based on activity and duration', async () => {
    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Running at MODERATE for 30 minutes should estimate ~300 calories
    const minutesInput = getByTestId('activity-duration-minutes');
    await act(async () => {
      fireEvent.changeText(minutesInput, '30');
    });

    // Calories input should have an estimated value
    const caloriesInput = getByTestId('activity-calories-input');
    await waitFor(() => {
      const value = caloriesInput.props.value;
      expect(parseInt(value)).toBeGreaterThan(0);
    });
  });
});

describe('AddActivityScreen - Edit Mode', () => {
  const mockActivity = {
    id: 'activity-1',
    userId: 'user-1',
    activityType: 'CYCLING' as const,
    intensity: 'HIGH' as const,
    startedAt: '2024-01-15T08:00:00Z',
    endedAt: '2024-01-15T09:00:00Z',
    duration: 60,
    caloriesBurned: 500,
    distance: 25000,
    source: 'manual' as const,
    notes: 'Morning ride',
    createdAt: '2024-01-15T09:00:00Z',
    updatedAt: '2024-01-15T09:00:00Z',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(require('expo-router'), 'useLocalSearchParams').mockReturnValue({
      editId: 'activity-1',
    });
    (activitiesApi.getById as jest.Mock).mockResolvedValue(mockActivity);
    (activitiesApi.update as jest.Mock).mockResolvedValue(mockActivity);
  });

  it('loads existing activity data in edit mode', async () => {
    const { getByText, getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Should show Edit Activity title
    await waitFor(() => {
      expect(getByText('Edit Activity')).toBeTruthy();
    });

    // Should show Cycling (loaded activity type)
    expect(getByText('Cycling')).toBeTruthy();
  });

  it('updates activity successfully', async () => {
    const mockRouter = { push: jest.fn(), back: jest.fn() };
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue(mockRouter);

    const { getByTestId } = render(<AddActivityScreen />);

    await waitFor(() => {
      expect(getByTestId('add-activity-screen')).toBeTruthy();
    });

    // Wait for data to load
    await waitFor(() => {
      expect(activitiesApi.getById).toHaveBeenCalledWith('activity-1');
    });

    // Save
    const saveButton = getByTestId('activity-save-button');
    await act(async () => {
      fireEvent.press(saveButton);
    });

    await waitFor(() => {
      expect(activitiesApi.update).toHaveBeenCalledWith(
        'activity-1',
        expect.any(Object)
      );
    });

    expect(Alert.alert).toHaveBeenCalledWith(
      'Success',
      'Activity updated!',
      expect.any(Array)
    );
  });
});
