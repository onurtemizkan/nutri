# Task ID: 4

**Title:** Build Activity Tracking Mobile UI Screens

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Create mobile screens for viewing and manually logging activities. Backend API is complete at /api/activities.

**Details:**

1. Create new screens in `app/` directory:
   - `app/activity/index.tsx` - Activity list/history
   - `app/activity/[id].tsx` - Activity detail view
   - `app/activity/add.tsx` - Manual activity entry form

2. Activity List (`app/activity/index.tsx`):
   - Weekly summary card: total minutes, calories, workout count
   - Filter by activity type (All, Cardio, Strength, Flexibility)
   - List of recent activities with icon, duration, calories
   - Floating action button to add new activity
   - Pull-to-refresh

3. Activity Detail View (`app/activity/[id].tsx`):
   - Display all activity fields: type, duration, intensity, calories
   - Heart rate data if available (avg, max)
   - Distance and steps for applicable activities
   - Notes field
   - Edit/Delete buttons

4. Manual Entry Form (`app/activity/add.tsx`):
   - Activity type picker (21 types from ActivityType enum)
   - Intensity picker (Low, Moderate, High, Maximum)
   - Duration input (hours:minutes picker)
   - Date/time pickers for start time
   - Optional fields: calories, heart rate, distance, notes
   - Validation: duration > 0, end time > start time

5. Create API client in `lib/api/activities.ts`:
```typescript
export const activitiesApi = {
  getAll: (params?: { activityType?: string; startDate?: string }) =>
    apiClient.get('/activities', { params }),
  getById: (id: string) => apiClient.get(`/activities/${id}`),
  create: (data: CreateActivityInput) => apiClient.post('/activities', data),
  update: (id: string, data: Partial<CreateActivityInput>) =>
    apiClient.put(`/activities/${id}`, data),
  delete: (id: string) => apiClient.delete(`/activities/${id}`),
  getWeeklySummary: () => apiClient.get('/activities/weekly-summary'),
}
```

6. Add activity icons mapping for different activity types

**Test Strategy:**

1. Component tests for each screen
2. Test form validation (duration, time constraints)
3. Test activity type filtering
4. Test CRUD operations with mock API
5. Test weekly summary calculation display
6. Test edit/delete flows with confirmation dialogs
