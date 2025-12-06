/**
 * Component tests for Add Health Metric screen
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import AddHealthMetricScreen from '@/app/health/add';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import { showAlert } from '@/lib/utils/alert';

// Mock dependencies
jest.mock('@/lib/api/health-metrics');
jest.mock('@/lib/utils/alert');
jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
}));
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: ({ children }: { children: React.ReactNode }) => children,
}));
jest.mock('@react-native-community/datetimepicker', () => {
  const React = require('react');
  return ({ value, onChange }: { value: Date; onChange: (event: unknown, date?: Date) => void }) => {
    return React.createElement('DateTimePicker', {
      testID: 'datetime-picker',
      value,
      onChange: () => onChange({}, new Date()),
    });
  };
});
jest.mock('@expo/vector-icons', () => ({
  Ionicons: ({ name, ...props }: { name: string; [key: string]: unknown }) => {
    const { Text } = require('react-native');
    return <Text {...props}>{name}</Text>;
  },
}));

const mockedHealthMetricsApi = healthMetricsApi as jest.Mocked<typeof healthMetricsApi>;
const mockedShowAlert = showAlert as jest.Mock;

describe('AddHealthMetricScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders header with title and buttons', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByText('Add Health Metric');
    await findByText('Cancel');
    await findByText('Save');
  });

  it('renders default metric type (Resting Heart Rate)', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByText('Resting Heart Rate');
    await findByText('Unit: bpm');
  });

  it('renders value input with placeholder', async () => {
    // Act
    const { findByPlaceholderText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByPlaceholderText('Enter rhr');
  });

  it('renders date and time pickers', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByText('Date');
    await findByText('Time');
  });

  it('shows Manual Entry as data source', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByText('Data Source');
    await findByText('Manual Entry');
  });

  it('shows metric description', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert
    await findByText('Your heart rate while at rest');
  });

  it('shows valid range hint', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Assert - check that valid range text exists (rendered across multiple Text nodes)
    await findByText(/Valid range:/);
  });

  it('opens metric picker modal when tapped', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);
    await findByText('Resting Heart Rate');

    // Tap on the metric type selector
    const metricSelector = getByText('Resting Heart Rate');
    fireEvent.press(metricSelector);

    // Assert - modal should open
    await findByText('Select Metric Type');
  });

  it('shows all metric categories in picker modal', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);
    await findByText('Resting Heart Rate');

    // Open modal
    fireEvent.press(getByText('Resting Heart Rate'));

    // Assert - categories should be visible
    await findByText('Cardiovascular');
    await findByText('Sleep');
    await findByText('Recovery');
    await findByText('Body Composition');
  });

  it('changes metric type when selected from picker', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);
    await findByText('Resting Heart Rate');

    // Open modal and select Sleep Duration
    fireEvent.press(getByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    const sleepOption = await findByText('Sleep Duration');
    fireEvent.press(sleepOption);

    // Assert - metric type should change (displayName changes to Sleep Duration)
    await findByText('Sleep Duration');
    await findByText('Unit: hours');
  });

  it('shows validation error when value is empty', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);
    await findByText('Add Health Metric');

    // Try to save without entering value
    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert
    await findByText('Value is required');
  });

  it('shows validation error for invalid number', async () => {
    // Act
    const { findByText, getByText, getByPlaceholderText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter invalid value (non-numeric characters are filtered, but empty becomes invalid)
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '...');

    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert
    await findByText('Please enter a valid number');
  });

  it('shows validation error for out-of-range value', async () => {
    // Act
    const { findByText, getByText, getByPlaceholderText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter value outside valid range (RHR: 20-200)
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '10');

    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert - should show validation error (check for any error state)
    await findByText(/must be|invalid|out of range|between/i);
  });

  it('clears error when value is changed', async () => {
    // Act
    const { findByText, getByText, getByPlaceholderText, queryByText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Trigger error
    const saveButton = getByText('Save');
    fireEvent.press(saveButton);
    await findByText('Value is required');

    // Enter valid value
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62');

    // Assert - error should be cleared
    await waitFor(() => {
      expect(queryByText('Value is required')).toBeNull();
    });
  });

  it('filters non-numeric characters from input', async () => {
    // Act
    const { findByText, getByPlaceholderText } = render(<AddHealthMetricScreen />);
    await findByText('Add Health Metric');

    // Enter value with letters
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62abc');

    // Assert - only numbers should remain
    expect(input.props.value).toBe('62');
  });

  it('allows decimal values', async () => {
    // Act
    const { findByText, getByPlaceholderText } = render(<AddHealthMetricScreen />);
    await findByText('Add Health Metric');

    // Enter decimal value
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62.5');

    // Assert
    expect(input.props.value).toBe('62.5');
  });

  it('calls API and shows success alert on valid submission', async () => {
    // Arrange
    mockedHealthMetricsApi.create.mockResolvedValueOnce({
      id: 'metric-1',
      userId: 'user-1',
      metricType: 'RESTING_HEART_RATE',
      value: 62,
      unit: 'bpm',
      recordedAt: new Date().toISOString(),
      source: 'manual',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });

    // Act
    const { findByText, getByText, getByPlaceholderText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter valid value
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62');

    // Save
    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert
    await waitFor(() => {
      expect(mockedHealthMetricsApi.create).toHaveBeenCalledWith(
        expect.objectContaining({
          metricType: 'RESTING_HEART_RATE',
          value: 62,
          unit: 'bpm',
          source: 'manual',
        })
      );
    });

    await waitFor(() => {
      expect(mockedShowAlert).toHaveBeenCalledWith(
        'Success',
        'Health metric added successfully!',
        expect.any(Array)
      );
    });
  });

  it('shows error alert on API failure', async () => {
    // Arrange
    mockedHealthMetricsApi.create.mockRejectedValueOnce(new Error('Network error'));

    // Act
    const { findByText, getByText, getByPlaceholderText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter valid value
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62');

    // Save
    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert - showAlert should be called with error
    await waitFor(() => {
      expect(mockedShowAlert).toHaveBeenCalledWith(
        'Error',
        expect.any(String)
      );
    }, { timeout: 3000 });
  });

  it('disables form during submission', async () => {
    // Arrange
    mockedHealthMetricsApi.create.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    // Act
    const { findByText, getByText, getByPlaceholderText, findByTestId } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter valid value
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62');

    // Save
    const saveButton = getByText('Save');
    fireEvent.press(saveButton);

    // Assert - should show loading state (ActivityIndicator replaces Save text)
    await waitFor(() => {
      expect(input.props.editable).toBe(false);
    });
  });

  it('cancels without saving when cancel button pressed', async () => {
    // Arrange
    const mockBack = jest.fn();
    jest.spyOn(require('expo-router'), 'useRouter').mockReturnValue({
      push: jest.fn(),
      back: mockBack,
    });

    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);
    await findByText('Add Health Metric');

    // Press cancel
    const cancelButton = getByText('Cancel');
    fireEvent.press(cancelButton);

    // Assert
    expect(mockBack).toHaveBeenCalled();
    expect(mockedHealthMetricsApi.create).not.toHaveBeenCalled();
  });

  it('closes metric picker modal when X button pressed', async () => {
    // Act
    const { findByText, getByText, queryByText } = render(<AddHealthMetricScreen />);
    await findByText('Resting Heart Rate');

    // Open modal
    fireEvent.press(getByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    // Close modal (find close button by test behavior)
    // The X button uses Ionicons with name="close"
    const closeButton = getByText('Select Metric Type').parent?.parent?.children[1];
    if (closeButton) {
      fireEvent.press(closeButton);
    }

    // Assert - modal should be closed
    await waitFor(() => {
      // Select Metric Type should no longer be visible as modal title
      // But the component might still render it - check if modal content area is gone
      expect(queryByText('Cardiovascular')).toBeNull();
    });
  });

  it('resets value when metric type changes', async () => {
    // Act
    const { findByText, getByText, getByPlaceholderText } = render(
      <AddHealthMetricScreen />
    );
    await findByText('Add Health Metric');

    // Enter value for RHR
    const input = getByPlaceholderText('Enter rhr');
    fireEvent.changeText(input, '62');
    expect(input.props.value).toBe('62');

    // Change metric type
    fireEvent.press(getByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    const hrvOption = await findByText('Heart Rate Variability (SDNN)');
    fireEvent.press(hrvOption);

    // Assert - value should be reset (shortName is 'HRV' -> lowercase 'hrv')
    const newInput = getByPlaceholderText('Enter hrv');
    expect(newInput.props.value).toBe('');
  });
});

describe('AddHealthMetricScreen - Different Metric Types', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows correct validation range for Sleep Duration', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);

    // Open modal and select Sleep Duration
    fireEvent.press(await findByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    const sleepOption = await findByText('Sleep Duration');
    fireEvent.press(sleepOption);

    // Assert
    await findByText(/Valid range: 0 - 24 hours/);
  });

  it('shows correct validation range for Body Fat Percentage', async () => {
    // Act
    const { findByText } = render(<AddHealthMetricScreen />);

    // Open modal and select Body Fat Percentage
    fireEvent.press(await findByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    const bodyFatOption = await findByText('Body Fat Percentage');
    fireEvent.press(bodyFatOption);

    // Assert
    await findByText(/Valid range:/);
  });

  it('shows correct validation range for Recovery Score', async () => {
    // Act
    const { findByText, getByText } = render(<AddHealthMetricScreen />);

    // Open modal and select Recovery Score
    fireEvent.press(await findByText('Resting Heart Rate'));
    await findByText('Select Metric Type');

    const recoveryOption = await findByText('Recovery Score');
    fireEvent.press(recoveryOption);

    // Assert
    await findByText(/Valid range: 0 - 100 %/);
  });
});
