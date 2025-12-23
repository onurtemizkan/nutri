# Task ID: 40

**Title:** Push Notifications System with Admin Management and User Preferences

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Implement a comprehensive push notification system supporting iOS (APNs) and Android (FCM) with granular user preference controls, admin panel notification management, and contextual permission requests during onboarding. Follow 2025 iOS development best practices including provisional authorization, Focus Mode respect, Time Sensitive notifications, and Live Activities support.

**Details:**

## Technical Architecture

### Mobile App (expo-notifications)
1. **Permission Flow**
   - Pre-permission priming screen with value proposition visuals
   - Provisional authorization for iOS (silent notifications first)
   - Just-in-time permission request during onboarding (after first meal logged)
   - Graceful decline handling with in-app follow-up prompts
   - Check and request notification permissions via `expo-notifications`

2. **Notification Categories & Actions**
   - MEAL_REMINDER: Quick-log actions, snooze, dismiss
   - GOAL_PROGRESS: View details, log meal
   - HEALTH_INSIGHT: View correlation, dismiss
   - SUPPLEMENT_REMINDER: Mark taken, snooze
   - STREAK_ALERT: Log now, remind later
   - WEEKLY_SUMMARY: View report

3. **Rich Notification Support**
   - Custom images for meal suggestions
   - Action buttons for quick interactions
   - Custom sounds per category (optional)
   - Notification grouping by category

4. **iOS-Specific Features (2025)**
   - Live Activities for real-time calorie/macro tracking
   - Focus Mode filtering (interruptionLevel)
   - Time Sensitive for health alerts
   - Provisional authorization flow
   - Critical alerts (if applicable for health metrics)

5. **User Preferences Screen**
   - Master toggle (all notifications)
   - Category toggles with descriptions:
     * Meal reminders (with time pickers per meal)
     * Goal progress updates
     * Health insights from ML
     * Supplement reminders
     * Streak alerts
     * Weekly summaries
     * Marketing/tips (separate consent)
   - Quiet hours configuration (start/end times)
   - Notification frequency controls

### Backend API (Express/Node.js)

1. **Database Schema (Prisma)**
   - DeviceToken model: userId, token, platform, createdAt, lastActiveAt
   - NotificationPreference model: userId, category, enabled, settings (JSON)
   - NotificationLog model: id, userId, type, title, body, sentAt, deliveredAt, openedAt, actionTaken
   - NotificationCampaign model: id, title, segment, content, scheduledAt, status, analytics

2. **Push Service Integration**
   - APNs provider configuration (jwt or certificate auth)
   - FCM HTTP v1 API integration
   - Unified send interface abstracting platform differences
   - Token refresh handling and cleanup of stale tokens
   - Delivery receipt tracking

3. **Scheduling System**
   - Timezone-aware scheduling (store user timezone)
   - Recurring notification jobs (meal reminders)
   - Campaign scheduling with queue (Bull/BullMQ)
   - Rate limiting to prevent notification fatigue
   - Smart send time optimization based on user engagement

4. **API Endpoints**
   - POST /api/notifications/register-device
   - DELETE /api/notifications/unregister-device
   - GET /api/notifications/preferences
   - PUT /api/notifications/preferences
   - POST /api/notifications/test (dev only)
   - GET /api/notifications/history

### Admin Panel Features

1. **Campaign Management**
   - Create notification campaigns with rich editor
   - Schedule future sends with timezone handling
   - User segmentation filters:
     * By activity level (active, dormant, churned)
     * By goals (weight loss, muscle gain, etc.)
     * By engagement metrics
     * By subscription status
   - A/B testing support (title, body, timing)
   - Preview on device mockups

2. **Analytics Dashboard**
   - Delivery rate metrics
   - Open rate tracking
   - Engagement metrics (action button taps)
   - Conversion tracking (e.g., notification → meal logged)
   - Unsubscribe/opt-out trends
   - Best performing notification types

3. **Individual User Management**
   - Send notification to specific user
   - View user's notification history
   - View/override user preferences (support use case)
   - Device token management

4. **Notification Templates**
   - Reusable templates for common notifications
   - Variable interpolation ({{userName}}, {{goalProgress}}, etc.)
   - Multi-language support preparation

5. **Audit & Compliance**
   - Full notification send history
   - GDPR-compliant consent tracking
   - Marketing opt-in separate from transactional
   - Data retention policies for notification logs

### Onboarding Integration

1. **Permission Request Timing**
   - Show pre-permission screen after user completes first meal log
   - Explain 3 key benefits with illustrations
   - Offer to customize preferences immediately
   - Skip option with in-app reminder after 3 days

2. **Initial Preference Setup**
   - Quick toggles for main categories during onboarding
   - Meal reminder time setup based on typical schedule
   - Smart defaults based on user goals

3. **Decline Handling**
   - Track permission status
   - Show contextual prompts when relevant (e.g., "Enable reminders to keep your streak!")
   - Never ask more than once per week
   - Deep link to iOS Settings if permanently declined

## Dependencies
- expo-notifications package
- APNs credentials (Apple Developer account)
- FCM project setup (Firebase Console)
- Redis for job queue (already in stack)
- Bull/BullMQ for scheduling

## Security Considerations
- Validate device tokens server-side
- Rate limit notification API endpoints
- Secure APNs/FCM credentials in environment
- Audit logging for admin actions

**Test Strategy:**

## Testing Strategy

### Unit Tests
- Notification preference CRUD operations
- Device token registration/validation
- Notification payload construction per platform
- Timezone conversion utilities
- Segmentation filter logic

### Integration Tests
- End-to-end device registration flow
- Preference sync between app and server
- Campaign creation and scheduling
- Admin API authorization

### E2E Tests (Maestro)
- Onboarding permission flow
- Preference screen interactions
- Notification tap handling and deep links

### Manual Testing
- Real device testing on iOS and Android
- APNs sandbox vs production
- Focus Mode behavior verification
- Background/foreground notification handling
- Rich notification rendering

### Load Testing
- Bulk notification send performance
- Queue processing under load

## Subtasks

### 40.1. Design and implement Prisma database schema for push notifications

**Status:** pending  
**Dependencies:** None  

Create Prisma models for DeviceToken, NotificationPreference, NotificationLog, and NotificationCampaign with proper indexes and relationships to the User model

**Details:**

Add the following models to server/prisma/schema.prisma:

1. DeviceToken model:
   - Fields: id (cuid), userId (relation), token (String), platform (enum: IOS/ANDROID), isActive (Boolean), lastActiveAt (DateTime)
   - Indexes: [userId, isActive], [token], [userId, platform]
   - Unique constraint on [userId, token, platform]

2. NotificationPreference model:
   - Fields: id (cuid), userId (unique relation), enabledCategories (String[]), quietHoursStart (String nullable), quietHoursEnd (String nullable), mealReminderTimes (Json), settings (Json)
   - Categories: MEAL_REMINDER, GOAL_PROGRESS, HEALTH_INSIGHT, SUPPLEMENT_REMINDER, STREAK_ALERT, WEEKLY_SUMMARY, MARKETING
   - Index: [userId]

3. NotificationLog model:
   - Fields: id (cuid), userId, notificationType (String), category (String), title, body, data (Json), platform (enum), sentAt, deliveredAt (nullable), openedAt (nullable), actionTaken (String nullable)
   - Indexes: [userId, sentAt], [notificationType], [category]

4. NotificationCampaign model:
   - Fields: id (cuid), title, targetSegment (Json), content (Json), scheduledAt, sentAt (nullable), status (enum: DRAFT/SCHEDULED/SENDING/COMPLETED/FAILED), deliveryCount (Int default 0), openCount (Int default 0)
   - Indexes: [status], [scheduledAt]

After schema creation:
- Run `npm run db:generate` to generate Prisma client
- Run `npm run db:push` for development or create migration with `npm run db:migrate`
- Update server/src/types/index.ts if needed for TypeScript types

### 40.2. Implement backend push notification service with APNs and FCM integration

**Status:** pending  
**Dependencies:** 40.1  

Create a unified push notification service that abstracts APNs (iOS) and FCM (Android) provider implementations with token management and delivery tracking

**Details:**

Create server/src/services/pushNotificationService.ts:

1. Install dependencies:
   - `npm install @parse/node-apn` for APNs
   - `npm install firebase-admin` for FCM
   - Add types: `@types/node-apn`

2. Service structure:
   - Initialize APNs provider (production/sandbox based on NODE_ENV)
   - Initialize Firebase Admin SDK for FCM
   - Abstract send interface: sendNotification(userId, payload, options)
   - Platform detection from device token registration

3. Core methods:
   - `sendToUser(userId: string, notification: NotificationPayload)`: Send to all user's active devices
   - `sendToDevice(token: string, platform: Platform, notification: NotificationPayload)`: Send to specific device
   - `sendBatch(userIds: string[], notification: NotificationPayload)`: Batch sending
   - `registerDevice(userId: string, token: string, platform: Platform)`: Register/update device token
   - `unregisterDevice(userId: string, token: string)`: Mark device inactive
   - `cleanupStaleTokens()`: Remove tokens that have bounced

4. Notification payload structure:
   - title, body, data (custom payload)
   - category (for action buttons)
   - badge (iOS), sound
   - interruptionLevel (iOS: passive/active/timeSensitive/critical)
   - threadId (for grouping)

5. Error handling:
   - Handle invalid tokens (remove from database)
   - Retry logic for transient failures
   - Log delivery failures to NotificationLog

6. Environment variables (server/.env):
   - APNS_KEY_ID, APNS_TEAM_ID, APNS_KEY_PATH, APNS_BUNDLE_ID
   - FIREBASE_SERVICE_ACCOUNT_PATH or FIREBASE_PROJECT_ID
   - APNS_PRODUCTION=false (for development)

### 40.3. Create backend API endpoints for device registration and notification preferences

**Status:** pending  
**Dependencies:** 40.1, 40.2  

Implement REST API endpoints following existing controller/service pattern for managing device tokens and user notification preferences

**Details:**

Create server/src/routes/notificationRoutes.ts and server/src/controllers/notificationController.ts:

1. POST /api/notifications/register-device
   - Body: { token: string, platform: 'ios' | 'android' }
   - Validates token format
   - Calls pushNotificationService.registerDevice()
   - Returns: { success: boolean, deviceId: string }

2. DELETE /api/notifications/unregister-device
   - Body: { token: string }
   - Marks device as inactive
   - Returns: { success: boolean }

3. GET /api/notifications/preferences
   - Returns user's notification preferences
   - Returns: { enabledCategories: string[], quietHours: {...}, settings: {...} }

4. PUT /api/notifications/preferences
   - Body: { enabledCategories?: string[], quietHoursStart?: string, quietHoursEnd?: string, settings?: object }
   - Updates preferences (partial update)
   - Returns updated preferences

5. GET /api/notifications/history
   - Query params: ?page=1&limit=20&category=MEAL_REMINDER
   - Returns paginated notification history
   - Follows pagination pattern from existing controllers

6. POST /api/notifications/test (dev only)
   - Sends test notification to user
   - Only enabled when NODE_ENV=development
   - Returns: { success: boolean, sentTo: number }

Validation (server/src/validation/schemas.ts):
- Create registerDeviceSchema, updatePreferencesSchema
- Use existing patterns from authController.ts and mealController.ts

Middleware:
- All endpoints require authentication (use requireAuth)
- Apply rate limiting (100 req/15min for registration, standard for others)
- Input sanitization via existing sanitize middleware

Register routes in server/src/index.ts following existing pattern

### 40.4. Implement notification scheduling system with Bull/BullMQ and Redis

**Status:** pending  
**Dependencies:** 40.2  

Create a job queue system for scheduling recurring notifications (meal reminders, weekly summaries) and campaign sends with timezone-aware scheduling

**Details:**

Create server/src/services/notificationScheduler.ts using Bull:

1. Install dependencies:
   - `npm install bull` (BullMQ requires Bull v4+ and may have breaking changes)
   - `npm install @types/bull`
   - Redis is already in the stack (verify connection in docker-compose)

2. Queue setup:
   - Create queue: notificationQueue = new Bull('notifications', redisConfig)
   - Define job types: MEAL_REMINDER, WEEKLY_SUMMARY, CAMPAIGN_SEND, HEALTH_INSIGHT

3. Job processors:
   - processMealReminder(job): Get users with enabled MEAL_REMINDER, check quiet hours, send notification
   - processWeeklySummary(job): Generate weekly stats, send to users with enabled WEEKLY_SUMMARY
   - processCampaignSend(job): Get user segment, batch send notifications
   - processHealthInsight(job): Check for new ML insights, send notifications

4. Scheduling methods:
   - `scheduleMealReminder(userId: string, time: string)`: Schedule daily recurring job
   - `scheduleWeeklySummary(userId: string, dayOfWeek: number, time: string)`: Schedule weekly job
   - `scheduleCampaign(campaignId: string, sendAt: Date)`: Schedule one-time campaign
   - `cancelUserJobs(userId: string)`: Remove all scheduled jobs for user
   - `updateMealReminderSchedule(userId: string, times: object)`: Update user's meal reminder times

5. Timezone handling:
   - Store user timezone in User model (add timezone field to schema)
   - Use moment-timezone to convert user local time to UTC for scheduling
   - Schedule jobs in UTC, convert back to user timezone when processing

6. Smart send time optimization:
   - Track user's typical engagement times from NotificationLog.openedAt
   - Adjust send times to user's active hours
   - Implement in future iteration (add TODO comment)

7. Rate limiting:
   - Max 5 notifications per user per day
   - Implement check in job processor
   - Skip if limit exceeded, log to NotificationLog

8. Queue monitoring:
   - Add Bull Board for queue UI: `npm install @bull-board/express`
   - Mount at /admin/queues (admin auth required)

Environment variables:
- REDIS_URL (already configured for ml-service)
- NOTIFICATION_QUEUE_CONCURRENCY=5

### 40.5. Setup expo-notifications in mobile app with permission handling

**Status:** pending  
**Dependencies:** None  

Install and configure expo-notifications, implement permission request flow, and set up notification handlers for foreground/background/tap events

**Details:**

1. Install expo-notifications (check if already in package.json dependencies - it's not currently installed):
   - `npx expo install expo-notifications`
   - `npx expo install expo-device` (for checking if physical device)

2. Create lib/services/notifications/index.ts:
   - Configure notification channel for Android (high priority, sound enabled)
   - Set notification handler for foreground notifications
   - registerForPushNotificationsAsync(): Main permission request function
   - getExpoPushToken(): Get Expo push token (or native APNs/FCM token)

3. Permission flow:
   - Check current permissions with Notifications.getPermissionsAsync()
   - Request permissions with Notifications.requestPermissionsAsync()
   - For iOS: Check for provisional authorization support
   - Handle all permission states: granted, denied, undetermined
   - Return token and permission status

4. Create lib/context/NotificationContext.tsx:
   - Provide notification state across app
   - Store: permissionStatus, expoPushToken, notificationListener refs
   - Methods: requestPermission(), registerDevice(), unregisterDevice()

5. Notification handlers:
   - Foreground listener: Notifications.addNotificationReceivedListener()
   - Background/quit handler: Notifications.addNotificationResponseReceivedListener()
   - Handle tap on notification: route to appropriate screen based on data.screen

6. Update app/_layout.tsx:
   - Wrap app with NotificationProvider
   - Initialize notification listeners on mount
   - Clean up listeners on unmount

7. Notification categories (for action buttons):
   - Define category identifiers matching backend
   - Configure action buttons: "View", "Dismiss", "Snooze" (iOS only supports 4 actions max)
   - Set categories with Notifications.setNotificationCategoryAsync()

8. Testing utilities (lib/services/notifications/test.ts):
   - scheduleTestNotification(): Schedule local notification for testing
   - Only available in __DEV__ mode

Note: expo-notifications currently not in package.json - needs to be installed first

### 40.6. Implement onboarding pre-permission priming flow

**Status:** pending  
**Dependencies:** 40.5  

Create contextual permission request screens shown after user logs first meal, explaining notification benefits with option to customize preferences immediately

**Details:**

Create app/notifications/priming.tsx (new screen):

1. Screen design:
   - Header: "Stay on track with smart reminders"
   - 3 benefit cards with icons:
     * "Never miss a meal" - Customizable meal reminders
     * "Track your progress" - Weekly summary reports
     * "Discover insights" - ML-powered health correlations
   - Primary button: "Enable Notifications"
   - Secondary button: "Customize Settings" (shows preference toggles inline)
   - Tertiary link: "Maybe Later"

2. Trigger logic (update lib/context/AuthContext.tsx or create notification trigger service):
   - Track if user has seen priming screen (AsyncStorage: 'hasSeenNotificationPriming')
   - Track user's first meal log (add firstMealLoggedAt to User model or track locally)
   - Show priming screen when:
     * User completes first meal log AND
     * Has not seen priming screen before AND
     * Notification permission is 'undetermined'
   - Use modal presentation or stack navigation

3. Permission request flow:
   - On "Enable Notifications" tap:
     * Call NotificationContext.requestPermission()
     * If granted: register device token with backend, show success message
     * If denied: show instructions to enable in Settings, mark as seen
   - On "Customize Settings" tap:
     * Show inline category toggles (MEAL_REMINDER, GOAL_PROGRESS, etc.)
     * Show meal reminder time pickers
     * Save to backend preferences endpoint
     * Then request permission
   - On "Maybe Later" tap:
     * Mark as seen, schedule in-app reminder for 3 days later
     * Store: 'notificationReminderAt' in AsyncStorage

4. Decline handling flow:
   - If permission denied, store: 'notificationPermissionDenied' = true
   - Show contextual in-app prompts when relevant:
     * After logging 3-day streak: "Enable reminders to maintain your streak!"
     * When viewing health insights: "Get notified about new insights"
   - Prompts include deep link to iOS Settings if permanently declined
   - Never ask more than once per week (track last prompt date)

5. Navigation integration:
   - Add route to app/_layout.tsx with headerShown: false
   - Can be presented modally from anywhere using router.push('/notifications/priming')

6. Illustrations/Assets:
   - Add simple SVG illustrations or use expo-symbols for icons
   - Use theme colors from lib/theme/colors.ts
   - Responsive layout using lib/responsive utilities

Follow existing screen patterns from app/auth/welcome.tsx for styling

### 40.7. Build user notification preferences settings screen

**Status:** pending  
**Dependencies:** 40.3, 40.5  

Create a settings screen with master toggle, category-specific toggles, quiet hours configuration, and meal reminder time pickers integrated with backend preferences API

**Details:**

Create app/notifications/preferences.tsx:

1. Screen structure:
   - Header: "Notification Preferences" with save button
   - Master toggle: "Enable All Notifications" (disables all categories when off)
   - Category sections:

2. Category toggles with descriptions:
   - MEAL_REMINDER: "Meal Reminders" - "Get reminded to log your meals"
     * Sub-settings: Time pickers for breakfast/lunch/dinner (conditionally shown when enabled)
   - GOAL_PROGRESS: "Goal Progress" - "Daily and weekly progress updates"
   - HEALTH_INSIGHT: "Health Insights" - "ML-powered nutrition-health correlations"
   - SUPPLEMENT_REMINDER: "Supplement Reminders" - "Remember to take your supplements"
   - STREAK_ALERT: "Streak Alerts" - "Don't lose your logging streak"
   - WEEKLY_SUMMARY: "Weekly Summary" - "Your week in review every Sunday"
   - MARKETING: "Tips & Updates" - "Product tips and feature updates" (separate consent)

3. Quiet hours configuration:
   - Toggle: "Enable Quiet Hours"
   - Start time picker (default: 10:00 PM)
   - End time picker (default: 8:00 AM)
   - Helper text: "No notifications during these hours"

4. State management:
   - Use local state for UI (useState)
   - Load preferences from API on mount: GET /api/notifications/preferences
   - Save on button tap or auto-save on toggle: PUT /api/notifications/preferences
   - Show loading states during API calls
   - Handle errors with Alert.alert

5. Meal reminder time pickers:
   - Use @react-native-community/datetimepicker (already in dependencies)
   - Store times as strings (HH:mm format)
   - Update backend when changed
   - Trigger reschedule of meal reminder jobs on backend

6. Permission status indicator:
   - Show current notification permission status at top
   - If denied: Show "Notifications Disabled" banner with "Open Settings" button
   - Deep link to iOS Settings using Linking.openSettings()

7. Styling:
   - Follow existing patterns from app/health-settings.tsx
   - Use theme colors from lib/theme/colors.ts
   - Responsive layout with lib/responsive utilities
   - Section headers with separators
   - Toggle switches with platform-specific styling

8. Navigation:
   - Add to app/(tabs)/profile.tsx as a list item: "Notification Preferences"
   - Register route in app/_layout.tsx with headerShown: false
   - Custom header component with back button and save button

API integration:
- Use lib/api/client.ts for authenticated requests
- Follow error handling pattern from lib/utils/errorHandling.ts
- Show success feedback after save (subtle checkmark or toast)

### 40.8. Implement notification handling for foreground, background, and tap actions

**Status:** pending  
**Dependencies:** 40.5, 40.3  

Handle notifications in all app states (foreground/background/quit), implement navigation based on notification data, and track notification opens

**Details:**

Update lib/services/notifications/index.ts and create lib/services/notifications/handlers.ts:

1. Foreground notification handler:
   - Notifications.setNotificationHandler({ handleNotification: async () => {...} })
   - Return: { shouldShowAlert: true, shouldPlaySound: true, shouldSetBadge: false }
   - Show in-app notification banner (use custom component or native)
   - Add to notification center

2. Background/tap notification handler:
   - Notifications.addNotificationResponseReceivedListener((response) => {...})
   - Extract notification data: response.notification.request.content.data
   - Route to appropriate screen based on data.screen and data.params

3. Routing logic based on notification type:
   - MEAL_REMINDER → Navigate to /add-meal with optional data.mealType
   - GOAL_PROGRESS → Navigate to /(tabs)/ (dashboard)
   - HEALTH_INSIGHT → Navigate to /health/[metricType] with data.metricType
   - SUPPLEMENT_REMINDER → Navigate to /supplements with data.supplementId
   - STREAK_ALERT → Navigate to /add-meal
   - WEEKLY_SUMMARY → Navigate to /(tabs)/health with data.weekStart

4. Action button handling (iOS notification categories):
   - Check response.actionIdentifier
   - Handle actions: VIEW, DISMISS, SNOOZE, LOG_NOW, MARK_TAKEN
   - SNOOZE: Schedule local notification for 15 minutes later
   - LOG_NOW: Open appropriate logging screen with pre-filled data
   - MARK_TAKEN: Call supplement API to mark as taken

5. Track notification opens:
   - Call backend: POST /api/notifications/track with { notificationId, action: 'opened' }
   - Include actionIdentifier if action button was tapped
   - Update NotificationLog.openedAt and actionTaken

6. Badge management:
   - Get current badge count: Notifications.getBadgeCountAsync()
   - Set badge count: Notifications.setBadgeCountAsync(count)
   - Clear badge on app launch: call setBadgeCountAsync(0) in app/_layout.tsx
   - Decrement badge when notification is read

7. Deep linking integration:
   - Use expo-router's router.push() for navigation
   - Handle edge cases: app not fully initialized, auth required
   - Queue navigation until app is ready if needed

8. Notification state persistence:
   - Store last notification ID in AsyncStorage to prevent duplicate handling
   - Clear on app launch

9. Error handling:
   - Wrap all handlers in try-catch
   - Log errors but don't crash app
   - Fallback to dashboard if navigation fails

10. Platform differences:
    - iOS: Action buttons work, badge count works
    - Android: Limited action buttons, different badge behavior
    - Test both platforms thoroughly

Integration in app/_layout.tsx:
- Initialize handlers on app mount
- Clean up listeners on unmount
- Handle navigation from notification when app is quit (check Notifications.getLastNotificationResponseAsync())

### 40.9. Implement iOS-specific notification features (Live Activities, Focus Mode, provisional auth)

**Status:** pending  
**Dependencies:** 40.5, 40.8  

Add support for iOS 2025 best practices including provisional authorization, Focus Mode respect via interruptionLevel, Time Sensitive notifications, and Live Activities for real-time tracking

**Details:**

1. Provisional Authorization (iOS 12+):
   - Update lib/services/notifications/index.ts permission request
   - First attempt provisional authorization: { ios: { allowProvisional: true } }
   - Deliver notifications silently to notification center without interrupting
   - After user engages with notification, request full permission
   - Track provisional state separately from full authorization

2. Focus Mode filtering (iOS 15+):
   - Set interruptionLevel in notification payload:
     * passive: Appears in notification center only (default for most)
     * active: Normal notification with sound (meal reminders)
     * timeSensitive: Breaks through Focus Mode (health alerts)
     * critical: Requires special entitlement (skip for now)
   - Update pushNotificationService.ts to set interruptionLevel based on category:
     * MEAL_REMINDER → active
     * HEALTH_INSIGHT → timeSensitive (if critical health metric)
     * WEEKLY_SUMMARY → passive
     * SUPPLEMENT_REMINDER → active

3. Time Sensitive notifications:
   - Request permission for Time Sensitive: { ios: { allowTimeSensitive: true } }
   - Use for health alerts that need immediate attention
   - Configure in notification payload: { interruptionLevel: 'timeSensitive' }

4. Live Activities (iOS 16.1+) - Basic implementation:
   - Install dependencies: `npx expo install expo-live-activities` (if available) or use native iOS module
   - Note: Live Activities require native iOS development and ActivityKit
   - Plan for future native module implementation
   - Add TODO comments for:
     * Real-time calorie/macro tracking widget
     * Live progress bar during meal logging
     * Daily goal progress ring
   - Document requirements in CLAUDE.md for future implementation

5. Notification relevance score (iOS 15+):
   - Set relevanceScore (0.0 to 1.0) for notification ranking
   - Higher scores appear more prominently
   - Set based on category importance:
     * STREAK_ALERT → 0.9
     * HEALTH_INSIGHT → 0.8
     * MEAL_REMINDER → 0.7
     * WEEKLY_SUMMARY → 0.5

6. Update notification payload structure:
   - Add iOS-specific fields in pushNotificationService.ts:
     * interruptionLevel
     * relevanceScore
     * threadId (for grouping)
     * targetContentId (for replacing notifications)

7. Notification grouping:
   - Use threadId to group related notifications
   - Examples:
     * "meal-reminders" for all meal reminders
     * "health-insights-{metricType}" for health insights by metric
     * "weekly-summary" for weekly reports
   - Configure summary format for notification stacks

8. Critical Alerts (future consideration):
   - Requires Apple approval and special entitlement
   - Add TODO for critical health alerts (e.g., dangerous blood glucose)
   - Document entitlement request process

9. Platform-specific UI updates:
   - Update priming screen to mention Focus Mode compatibility
   - Add explanation of Time Sensitive notifications for health alerts
   - Show current iOS notification settings (can't read programmatically, provide instructions)

10. Testing:
    - Test on iOS 15+ devices (provisional auth not supported on simulator)
    - Test with different Focus Modes enabled
    - Verify Time Sensitive notifications break through Do Not Disturb
    - Test notification grouping with multiple notifications

Note: Live Activities require native development - document for future iteration

### 40.10. Build admin panel campaign creation and management UI

**Status:** pending  
**Dependencies:** 40.1, 40.3  

Create Next.js admin panel pages for creating notification campaigns, scheduling sends, defining user segments, and A/B testing notification variants

**Details:**

Create admin-panel/app/(dashboard)/dashboard/notifications/ directory:

1. Campaign list page (page.tsx):
   - Table showing all campaigns: title, status, scheduled date, delivery/open rates
   - Filters: status (all/draft/scheduled/completed/failed), date range
   - Actions: Edit, Duplicate, Delete, View Analytics
   - "Create Campaign" button
   - Pagination matching existing admin pages pattern

2. Create/Edit campaign page (create/page.tsx and [id]/edit/page.tsx):
   - Form fields:
     * Campaign title (required)
     * Notification title (max 50 chars, show preview)
     * Notification body (max 150 chars, show preview)
     * Category dropdown (MEAL_REMINDER, GOAL_PROGRESS, etc.)
     * Data payload (JSON editor for custom data)
     * Schedule: Now, Specific Date/Time, or Save as Draft
     * Timezone selector (user's timezone or UTC)
   - User segmentation section:
     * Filters: Activity level (active/dormant/churned)
     * Goal type (weight loss/muscle gain/maintain)
     * Subscription tier (free/pro)
     * Last active date range
     * Custom SQL filter (admin only)
     * Show estimated recipient count
   - A/B testing section (optional):
     * Enable A/B test toggle
     * Variant A/B titles and bodies
     * Split percentage (default 50/50)
     * Test duration before auto-selecting winner
   - Device preview:
     * iOS and Android notification mockups
     * Update in real-time as fields change
     * Use react-device-frameset or similar library

3. Campaign analytics page ([id]/analytics/page.tsx):
   - Overview cards:
     * Total sent, Delivered, Opened, Clicked (if action buttons)
     * Delivery rate %, Open rate %, Click-through rate %
   - Time-series chart: Opens over time (hourly for first 24h, then daily)
   - Breakdown tables:
     * By platform (iOS vs Android)
     * By user segment
     * By A/B variant (if test)
   - Action button performance (if applicable)
   - Export to CSV button

4. Backend API endpoints (create server/src/routes/adminNotificationRoutes.ts):
   - GET /api/admin/notifications/campaigns - List campaigns with pagination
   - POST /api/admin/notifications/campaigns - Create campaign
   - GET /api/admin/notifications/campaigns/:id - Get campaign details
   - PUT /api/admin/notifications/campaigns/:id - Update campaign
   - DELETE /api/admin/notifications/campaigns/:id - Delete campaign (only drafts)
   - POST /api/admin/notifications/campaigns/:id/send - Send campaign immediately
   - GET /api/admin/notifications/campaigns/:id/analytics - Get analytics
   - POST /api/admin/notifications/campaigns/:id/test - Send test to admin's device
   - POST /api/admin/notifications/segment/estimate - Estimate recipient count

5. Backend controller (server/src/controllers/adminNotificationController.ts):
   - Implement controllers for each endpoint
   - User segmentation logic using Prisma queries
   - Schedule campaign using notificationScheduler
   - Audit logging for all campaign actions
   - Permission checks (SUPER_ADMIN or SUPPORT role)

6. Campaign send logic (update notificationScheduler.ts):
   - Add processCampaignSend job
   - Load campaign and segment users
   - Batch send notifications (1000 users per batch)
   - Update campaign status to SENDING, then COMPLETED
   - Update deliveryCount and openCount as notifications are sent/opened
   - Handle failures and retry logic

7. A/B testing logic:
   - Split users randomly into variants
   - Track metrics per variant
   - Auto-select winner after test duration based on open rate
   - Send winning variant to remaining users (if applicable)

8. UI Components (follow existing admin-panel patterns):
   - Use admin-panel/components/ui/button.tsx and other shadcn components
   - Follow layout from admin-panel/app/(dashboard)/layout.tsx
   - Match styling with existing admin pages
   - Use admin-panel/lib/api.ts for API calls

9. Device preview component:
   - Create components/NotificationPreview.tsx
   - iOS and Android mockup frames
   - Display title, body, icon, time
   - Show action buttons if configured

10. Form validation:
    - Client-side: React Hook Form with zod schemas
    - Server-side: Zod schemas in validation/adminSchemas.ts
    - Validate segment filters, schedule times, character limits

Security:
- All endpoints require admin authentication
- Audit log all campaign actions (create, edit, send)
- Rate limit campaign sending to prevent spam

### 40.11. Create admin panel analytics dashboard for notification metrics

**Status:** pending  
**Dependencies:** 40.10  

Build comprehensive analytics views showing delivery rates, open rates, engagement metrics, conversion tracking, and opt-out trends with charts and exportable reports

**Details:**

Create admin-panel/app/(dashboard)/dashboard/notifications/analytics/page.tsx:

1. Overview metrics (top cards):
   - Total notifications sent (last 30 days)
   - Average delivery rate %
   - Average open rate %
   - Average engagement rate (actions taken / delivered)
   - Trend indicators (↑/↓ vs previous period)

2. Time-series charts:
   - Notifications sent over time (daily, weekly, monthly views)
   - Open rate trend line
   - Delivery rate trend line
   - Use recharts or chart.js library
   - Date range selector (7d, 30d, 90d, custom)

3. Breakdown tables:
   - By notification category:
     * Category name, Sent count, Delivery %, Open %, Avg time to open
     * Sort by any column
   - By platform:
     * iOS vs Android metrics
     * Device model breakdown (from NotificationLog metadata)
   - Top performing notifications:
     * Highest open rates
     * Most actions taken
     * Show campaign title, sent date, metrics

4. Conversion tracking:
   - Track post-notification actions:
     * MEAL_REMINDER → Meal logged within 1 hour
     * GOAL_PROGRESS → App opened within 30 minutes
     * HEALTH_INSIGHT → Metric detail viewed
   - Conversion rate % for each category
   - Time-to-action histogram
   - Implementation: Track in NotificationLog with timestamps, correlate with user actions

5. User engagement analysis:
   - Notification frequency distribution (avg notifications per user per day)
   - Most engaged users (highest open rates)
   - Unresponsive users (0% open rate last 30 days)
   - Engagement heatmap by time of day and day of week

6. Opt-out/unsubscribe trends:
   - Chart: Opt-outs over time
   - Breakdown by category (which categories have highest opt-out)
   - Opt-out reasons (if we add reason collection)
   - Percentage of users with all notifications disabled

7. Best performing send times:
   - Heatmap: Open rate by hour of day and day of week
   - Identify optimal send times per category
   - Recommendations for campaign scheduling

8. Platform comparison:
   - Side-by-side metrics: iOS vs Android
   - Delivery issues by platform
   - Average time to delivery
   - Action button engagement (iOS only)

9. Export functionality:
   - Export to CSV button for each table/chart
   - PDF report generation (entire dashboard)
   - Scheduled email reports (future feature - add TODO)

10. Backend API endpoints (server/src/routes/adminNotificationRoutes.ts):
    - GET /api/admin/notifications/analytics/overview?startDate&endDate
    - GET /api/admin/notifications/analytics/by-category?startDate&endDate
    - GET /api/admin/notifications/analytics/by-platform?startDate&endDate
    - GET /api/admin/notifications/analytics/conversions?startDate&endDate
    - GET /api/admin/notifications/analytics/engagement?startDate&endDate
    - GET /api/admin/notifications/analytics/opt-outs?startDate&endDate
    - GET /api/admin/notifications/analytics/send-times?startDate&endDate
    - POST /api/admin/notifications/analytics/export - Generate CSV/PDF

11. Backend service (server/src/services/adminNotificationAnalyticsService.ts):
    - Implement complex Prisma aggregations
    - Calculate derived metrics (rates, averages, trends)
    - Optimize queries with indexes (NotificationLog indexes already in schema)
    - Cache results for expensive queries (use Redis)
    - Conversion tracking logic: correlate NotificationLog with Meal/Activity/HealthMetric timestamps

12. UI components:
    - Reuse admin panel patterns and components
    - Responsive charts (mobile-friendly)
    - Loading skeletons during data fetch
    - Error states with retry buttons
    - Empty states with helpful messages

13. Performance optimization:
    - Paginate large result sets
    - Server-side aggregations (don't load all logs to client)
    - Cache dashboard data (5-minute TTL)
    - Lazy load charts below the fold

14. Real-time updates (optional future feature):
    - WebSocket connection for live metrics
    - Auto-refresh every 30 seconds
    - Add TODO for implementation

Styling:
- Match existing admin panel design
- Use tailwind classes
- Responsive grid layout
- Print-friendly styles for PDF export

### 40.12. Implement admin panel individual user notification management and templates

**Status:** pending  
**Dependencies:** 40.10, 40.11  

Create admin interface for sending notifications to specific users, viewing user notification history, managing reusable notification templates, and overriding user preferences for support cases

**Details:**

Create admin-panel pages and backend endpoints:

1. User notification management page (admin-panel/app/(dashboard)/dashboard/users/[id]/notifications/page.tsx):
   - Accessible from user detail page
   - Sections:
     * Send notification to this user
     * User's notification preferences (view/override)
     * User's notification history
     * User's device tokens

2. Send notification to user:
   - Form fields:
     * Template dropdown (load from NotificationTemplate model)
     * Or custom: Title, Body, Category, Data payload
     * Send immediately or schedule
   - Preview notification mockup
   - Send button with confirmation
   - Use cases: Support messages, account alerts, custom offers

3. User preferences view/override:
   - Display current preferences (enabled categories, quiet hours)
   - Override toggle (support use case: re-enable critical health alerts)
   - Reason field (required for audit log)
   - Save override with admin audit logging
   - Show override indicator in user preferences screen

4. User notification history table:
   - Columns: Date, Type, Title, Status (sent/delivered/opened), Action
   - Filters: Date range, Category, Status
   - Pagination (20 per page)
   - Click row to view full notification details and user response

5. User device tokens:
   - Table: Platform, Token (truncated), Registered date, Last active, Status
   - Actions: Revoke token (for troubleshooting)
   - Show push token validity status

6. Notification templates page (admin-panel/app/(dashboard)/dashboard/notifications/templates/page.tsx):
   - List all templates: Name, Category, Usage count, Last updated
   - Create/Edit template form:
     * Template name (internal)
     * Category
     * Title (supports variables: {{userName}}, {{goalProgress}}, etc.)
     * Body (supports variables)
     * Default data payload (JSON)
     * Action buttons configuration (iOS)
   - Variable interpolation preview
   - Delete template (only if not used in any campaigns)

7. Template variable system:
   - Supported variables: {{userName}}, {{goalCalories}}, {{currentWeight}}, {{goalWeight}}, {{streakDays}}, {{weeklyProgress}}, {{metricName}}, {{metricValue}}
   - Backend service: interpolateTemplate(template, userData)
   - Validation: Ensure all variables are valid before saving

8. Backend models (add to schema.prisma):
   - NotificationTemplate model:
     * id, name, category, title, body, dataPayload (Json), variables (String[]), createdAt, updatedAt
     * Index: [category]
   - NotificationPreferenceOverride model (optional):
     * id, userId, overriddenBy (adminUserId), reason, settings (Json), createdAt, expiresAt

9. Backend API endpoints (server/src/routes/adminNotificationRoutes.ts):
   - POST /api/admin/notifications/send-to-user - Send to specific user
   - GET /api/admin/notifications/users/:userId/history - User's notification history
   - GET /api/admin/notifications/users/:userId/devices - User's device tokens
   - DELETE /api/admin/notifications/users/:userId/devices/:tokenId - Revoke token
   - PUT /api/admin/notifications/users/:userId/preferences/override - Override preferences
   - GET /api/admin/notifications/templates - List templates
   - POST /api/admin/notifications/templates - Create template
   - PUT /api/admin/notifications/templates/:id - Update template
   - DELETE /api/admin/notifications/templates/:id - Delete template
   - GET /api/admin/notifications/templates/:id/preview - Preview with sample data

10. Backend service (server/src/services/adminNotificationService.ts):
    - sendToUser(userId, notification, adminUserId): Send and log admin action
    - getUserNotificationHistory(userId, pagination): Get NotificationLog with pagination
    - getUserDeviceTokens(userId): Get active device tokens
    - overrideUserPreferences(userId, settings, reason, adminUserId): Update and log
    - Template CRUD operations
    - interpolateTemplate(template, userData): Replace variables with user data

11. Audit logging:
    - Log all admin actions in AdminAuditLog:
      * SEND_NOTIFICATION_TO_USER
      * OVERRIDE_NOTIFICATION_PREFERENCES
      * REVOKE_DEVICE_TOKEN
      * CREATE_NOTIFICATION_TEMPLATE
    - Include userId, templateId, reason in details (Json)

12. Multi-language support (preparation):
    - Add locale field to NotificationTemplate
    - Support template variants by language
    - Fallback to default language if user's language not available
    - Add TODO for full i18n implementation

13. UI components:
    - Template editor with syntax highlighting for variables
    - Variable picker dropdown
    - Real-time preview with sample data
    - Confirmation modals for destructive actions
    - Success/error toasts

14. Security:
    - Require SUPER_ADMIN or SUPPORT role for sending to individual users
    - Require SUPER_ADMIN for preference overrides
    - Rate limit individual sends (max 10 per hour per admin)
    - Audit log all actions with reason field

15. Data retention (GDPR compliance):
    - Document data retention policy for NotificationLog
    - Add script to delete notification logs older than 90 days
    - Exclude from deletion: Audit logs, campaign analytics
    - Add TODO for automated cleanup job
