/**
 * Activities API Client Unit Tests
 */

import api from '@/lib/api/client';
import { activitiesApi } from '@/lib/api/activities';
import {
  Activity,
  CreateActivityInput,
  UpdateActivityInput,
  WeeklySummary,
  DailySummary,
  ActivityStats,
} from '@/lib/types/activities';

// Mock the API client
jest.mock('@/lib/api/client');
const mockedApi = api as jest.Mocked<typeof api>;

describe('activitiesApi', () => {
  // Test fixtures
  const mockActivity: Activity = {
    id: 'activity-1',
    userId: 'user-1',
    activityType: 'RUNNING',
    intensity: 'MODERATE',
    startedAt: '2024-01-15T08:00:00Z',
    endedAt: '2024-01-15T08:45:00Z',
    duration: 45,
    caloriesBurned: 450,
    averageHeartRate: 145,
    maxHeartRate: 175,
    distance: 5500,
    steps: 6500,
    source: 'manual',
    notes: 'Morning run',
    createdAt: '2024-01-15T08:45:00Z',
    updatedAt: '2024-01-15T08:45:00Z',
  };

  const mockWeeklySummary: WeeklySummary = {
    totalMinutes: 240,
    totalCalories: 1800,
    workoutCount: 5,
    averageIntensity: 'MODERATE',
    byType: {
      RUNNING: { count: 2, minutes: 90, calories: 900 },
      CYCLING: { count: 1, minutes: 60, calories: 400 },
      YOGA: { count: 1, minutes: 45, calories: 150 },
      WEIGHT_TRAINING: { count: 1, minutes: 45, calories: 350 },
    } as WeeklySummary['byType'],
  };

  const mockDailySummary: DailySummary = {
    totalMinutes: 45,
    totalCalories: 450,
    workoutCount: 1,
    activities: [mockActivity],
  };

  const mockStats: ActivityStats = {
    count: 10,
    totalMinutes: 450,
    totalCalories: 4500,
    averageDuration: 45,
    averageCalories: 450,
    longestDuration: 90,
    mostRecentDate: '2024-01-15T08:00:00Z',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('create', () => {
    it('should create a new activity', async () => {
      const createData: CreateActivityInput = {
        activityType: 'RUNNING',
        intensity: 'MODERATE',
        startedAt: '2024-01-15T08:00:00Z',
        endedAt: '2024-01-15T08:45:00Z',
        duration: 45,
        caloriesBurned: 450,
        source: 'manual',
      };

      mockedApi.post.mockResolvedValueOnce({ data: mockActivity });

      const result = await activitiesApi.create(createData);

      expect(mockedApi.post).toHaveBeenCalledWith('/activities', createData);
      expect(result).toEqual(mockActivity);
    });

    it('should handle creation error', async () => {
      const createData: CreateActivityInput = {
        activityType: 'RUNNING',
        intensity: 'MODERATE',
        startedAt: '2024-01-15T08:00:00Z',
        endedAt: '2024-01-15T08:45:00Z',
        duration: 45,
        source: 'manual',
      };

      mockedApi.post.mockRejectedValueOnce(new Error('Network error'));

      await expect(activitiesApi.create(createData)).rejects.toThrow('Network error');
    });
  });

  describe('createBulk', () => {
    it('should create multiple activities in bulk', async () => {
      const activities: CreateActivityInput[] = [
        {
          activityType: 'RUNNING',
          intensity: 'MODERATE',
          startedAt: '2024-01-15T08:00:00Z',
          endedAt: '2024-01-15T08:45:00Z',
          duration: 45,
          source: 'manual',
        },
        {
          activityType: 'YOGA',
          intensity: 'LOW',
          startedAt: '2024-01-15T18:00:00Z',
          endedAt: '2024-01-15T18:30:00Z',
          duration: 30,
          source: 'manual',
        },
      ];

      const response = { created: 2, errors: [] };
      mockedApi.post.mockResolvedValueOnce({ data: response });

      const result = await activitiesApi.createBulk(activities);

      expect(mockedApi.post).toHaveBeenCalledWith('/activities/bulk', { activities });
      expect(result).toEqual(response);
    });
  });

  describe('getAll', () => {
    it('should return array of activities', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [mockActivity] } });

      const result = await activitiesApi.getAll();

      expect(mockedApi.get).toHaveBeenCalledWith('/activities', { params: undefined });
      expect(result).toEqual([mockActivity]);
    });

    it('should pass query params', async () => {
      const params = {
        activityType: 'RUNNING' as const,
        startDate: '2024-01-01',
        endDate: '2024-01-31',
        limit: 10,
      };

      mockedApi.get.mockResolvedValueOnce({ data: { activities: [mockActivity] } });

      await activitiesApi.getAll(params);

      expect(mockedApi.get).toHaveBeenCalledWith('/activities', { params });
    });

    it('should handle empty results', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [] } });

      const result = await activitiesApi.getAll();

      expect(result).toEqual([]);
    });

    it('should handle legacy response format (array directly)', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: [mockActivity] });

      const result = await activitiesApi.getAll();

      expect(result).toEqual([mockActivity]);
    });
  });

  describe('getById', () => {
    it('should return single activity by ID', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: mockActivity });

      const result = await activitiesApi.getById('activity-1');

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/activity-1');
      expect(result).toEqual(mockActivity);
    });

    it('should handle 404 not found', async () => {
      mockedApi.get.mockRejectedValueOnce({
        response: { status: 404, data: { error: 'Activity not found' } },
      });

      await expect(activitiesApi.getById('nonexistent')).rejects.toMatchObject({
        response: { status: 404 },
      });
    });
  });

  describe('update', () => {
    it('should update an activity', async () => {
      const updateData: UpdateActivityInput = {
        intensity: 'HIGH',
        notes: 'Updated notes',
      };

      const updatedActivity = { ...mockActivity, ...updateData };
      mockedApi.put.mockResolvedValueOnce({ data: updatedActivity });

      const result = await activitiesApi.update('activity-1', updateData);

      expect(mockedApi.put).toHaveBeenCalledWith('/activities/activity-1', updateData);
      expect(result).toEqual(updatedActivity);
    });
  });

  describe('delete', () => {
    it('should delete an activity', async () => {
      mockedApi.delete.mockResolvedValueOnce({ data: {} });

      await activitiesApi.delete('activity-1');

      expect(mockedApi.delete).toHaveBeenCalledWith('/activities/activity-1');
    });

    it('should handle delete error', async () => {
      mockedApi.delete.mockRejectedValueOnce({
        response: { status: 404, data: { error: 'Activity not found' } },
      });

      await expect(activitiesApi.delete('nonexistent')).rejects.toMatchObject({
        response: { status: 404 },
      });
    });
  });

  describe('getDailySummary', () => {
    it('should return daily summary', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: mockDailySummary });

      const result = await activitiesApi.getDailySummary();

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/summary/daily', { params: {} });
      expect(result).toEqual(mockDailySummary);
    });

    it('should pass date param', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: mockDailySummary });

      await activitiesApi.getDailySummary('2024-01-15');

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/summary/daily', {
        params: { date: '2024-01-15' },
      });
    });
  });

  describe('getWeeklySummary', () => {
    it('should return weekly summary', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: mockWeeklySummary });

      const result = await activitiesApi.getWeeklySummary();

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/summary/weekly');
      expect(result).toEqual(mockWeeklySummary);
    });
  });

  describe('getRecoveryTime', () => {
    it('should return recovery recommendation', async () => {
      const recoveryData = {
        recommendedHours: 24,
        lastActivityAt: '2024-01-15T08:45:00Z',
        intensity: 'HIGH',
        message: 'Consider light activity or rest',
      };

      mockedApi.get.mockResolvedValueOnce({ data: recoveryData });

      const result = await activitiesApi.getRecoveryTime();

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/recovery');
      expect(result).toEqual(recoveryData);
    });
  });

  describe('getStatsByType', () => {
    it('should return stats for activity type', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: mockStats });

      const result = await activitiesApi.getStatsByType('RUNNING');

      expect(mockedApi.get).toHaveBeenCalledWith('/activities/stats/RUNNING');
      expect(result).toEqual(mockStats);
    });

    it('should return null for 404', async () => {
      mockedApi.get.mockRejectedValueOnce({
        response: { status: 404 },
      });

      const result = await activitiesApi.getStatsByType('GOLF');

      expect(result).toBeNull();
    });

    it('should throw for non-404 errors', async () => {
      mockedApi.get.mockRejectedValueOnce({
        response: { status: 500 },
      });

      await expect(activitiesApi.getStatsByType('RUNNING')).rejects.toMatchObject({
        response: { status: 500 },
      });
    });
  });

  describe('getRecentByType', () => {
    it('should return recent activities for type', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [mockActivity] } });

      const result = await activitiesApi.getRecentByType('RUNNING', 5);

      expect(mockedApi.get).toHaveBeenCalledWith('/activities', {
        params: { activityType: 'RUNNING', limit: 5 },
      });
      expect(result).toEqual([mockActivity]);
    });

    it('should use default limit of 10', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [] } });

      await activitiesApi.getRecentByType('YOGA');

      expect(mockedApi.get).toHaveBeenCalledWith('/activities', {
        params: { activityType: 'YOGA', limit: 10 },
      });
    });
  });

  describe('getForDateRange', () => {
    it('should return activities for date range', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [mockActivity] } });

      const result = await activitiesApi.getForDateRange('2024-01-01', '2024-01-31');

      expect(mockedApi.get).toHaveBeenCalledWith('/activities', {
        params: { startDate: '2024-01-01', endDate: '2024-01-31' },
      });
      expect(result).toEqual([mockActivity]);
    });
  });

  describe('getTodayActivities', () => {
    it('should return today\'s activities', async () => {
      mockedApi.get.mockResolvedValueOnce({ data: { activities: [mockActivity] } });

      const result = await activitiesApi.getTodayActivities();

      expect(mockedApi.get).toHaveBeenCalled();
      expect(result).toEqual([mockActivity]);
    });
  });
});
