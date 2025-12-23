# Task ID: 9

**Title:** Build Insights Feed Mobile UI

**Status:** pending

**Dependencies:** 8

**Priority:** medium

**Description:** Create mobile screens to display ML-generated insights, recommendations, and allow user feedback.

**Details:**

1. Create new screens:
   - `app/insights/index.tsx` - Insights feed/dashboard
   - `app/insights/[id].tsx` - Detailed insight view

2. Insights Feed (`app/insights/index.tsx`):
   - List of insight cards sorted by priority and recency
   - Card components per insight type:
     - Correlation: Show correlation strength badge
     - Prediction: Show predicted value and arrow
     - Anomaly: Show warning indicator
     - Recommendation: Show actionable tip
   - Swipe to dismiss functionality
   - Pull-to-refresh to generate new insights
   - Empty state: 'Keep logging meals to unlock insights'

3. Insight Card Design:
```typescript
interface InsightCard {
  icon: string;           // Based on insightType
  title: string;          // From insight.title
  description: string;    // Truncated insight.description
  priority: 'high' | 'medium' | 'low'; // Color coding
  correlation?: number;   // Show badge if correlation insight
  timestamp: Date;        // When generated
}
```

4. Detailed Insight View (`app/insights/[id].tsx`):
   - Full description text
   - Recommendation with call-to-action
   - Supporting chart/data if applicable
   - 'Was this helpful?' feedback buttons
   - Share insight button (future)

5. Create API client in `lib/api/insights.ts`:
```typescript
export const insightsApi = {
  getAll: () => apiClient.get('/api/insights'),
  getById: (id: string) => apiClient.get(`/api/insights/${id}`),
  markViewed: (id: string) => apiClient.put(`/api/insights/${id}/viewed`),
  dismiss: (id: string) => apiClient.put(`/api/insights/${id}/dismissed`),
  submitFeedback: (id: string, helpful: boolean) =>
    apiClient.put(`/api/insights/${id}/feedback`, { helpful }),
}
```

6. Add insights badge to tab bar showing unread count

**Test Strategy:**

1. Component tests for insight cards
2. Test swipe-to-dismiss interaction
3. Test feedback submission flow
4. Test empty and loading states
5. Test priority-based sorting
6. Visual regression tests for card styles

## Subtasks

### 9.1. Create insights API client and TypeScript types in lib/api/insights.ts

**Status:** pending  
**Dependencies:** None  

Create the API client module for insights with all required endpoints and TypeScript interfaces for Insight types

**Details:**

Create lib/api/insights.ts following the existing pattern from lib/api/health-metrics.ts. Define TypeScript interfaces: InsightType (enum: CORRELATION, PREDICTION, ANOMALY, RECOMMENDATION), InsightPriority (high/medium/low), Insight interface with fields (id, userId, insightType, title, description, priority, correlation?, predictedValue?, confidenceScore?, recommendation?, viewed, dismissed, helpful?, createdAt, expiresAt). Implement insightsApi object with methods: getAll() returning paginated insights sorted by priority and recency, getById(id), markViewed(id), dismiss(id), submitFeedback(id, helpful: boolean), getUnreadCount(). Use the shared api client from lib/api/client.ts with proper error handling pattern (isNotFoundError helper). Export types for use in components.

### 9.2. Create insight card components for each insight type (Correlation, Prediction, Anomaly, Recommendation)

**Status:** pending  
**Dependencies:** 9.1  

Build reusable InsightCard component with type-specific rendering for correlation strength badges, prediction arrows, anomaly warnings, and recommendation tips

**Details:**

Create lib/components/insights/InsightCard.tsx following SwipeableMealCard.tsx patterns. Props interface: InsightCardProps { insight: Insight; onPress: () => void; onDismiss: () => void }. Implement type-specific rendering: CORRELATION type shows correlation strength badge with color coding (strong >0.7 green, moderate 0.4-0.7 yellow, weak <0.4 gray); PREDICTION type shows predicted value with up/down arrow indicator based on trend; ANOMALY type shows warning icon with amber/red background based on severity; RECOMMENDATION type shows lightbulb icon with actionable tip preview. All cards show: icon based on insightType, truncated title/description, priority color band (high=red, medium=amber, low=gray), relative timestamp. Use colors from lib/theme/colors.ts, Ionicons for icons, and LinearGradient for priority bands. Include accessibilityLabel and accessibilityRole props.

### 9.3. Build insights feed screen at app/insights/index.tsx with FlatList and pull-to-refresh

**Status:** pending  
**Dependencies:** 9.1, 9.2  

Create the main insights dashboard screen with sorted insight cards, pull-to-refresh, empty state, and navigation to detail view

**Details:**

Create app/insights/index.tsx following app/(tabs)/health.tsx patterns. Use SafeAreaView, FlatList for virtualized list rendering. State: insights[], isLoading, refreshing, error. Use useFocusEffect to reload on screen focus. Implement pull-to-refresh with RefreshControl (tintColor from theme). Sort insights by priority (high first) then by createdAt (newest first). Filter out dismissed insights. Render InsightCard for each item with onPress navigating to /insights/[id]. Empty state component: heart-pulse icon, 'Keep logging meals to unlock insights' message, styled per emptyContainer pattern. Loading state with ActivityIndicator. Error state with retry button. Header with 'Insights' title and date subtitle. Use useResponsive hook for tablet/phone layouts. Register screen in app/_layout.tsx with headerShown: false and slide_from_right animation.

### 9.4. Create insight detail view at app/insights/[id].tsx with full content, feedback buttons, and charts

**Status:** pending  
**Dependencies:** 9.1, 9.2  

Build detailed insight view screen showing full description, recommendation with CTA, supporting data visualization, and 'Was this helpful?' feedback interaction

**Details:**

Create app/insights/[id].tsx following app/health/[metricType].tsx patterns. Use useLocalSearchParams to get insight id. Fetch insight via insightsApi.getById(), call markViewed on mount. Layout: SafeAreaView with custom header (back button, insight type icon, title). ScrollView content: full description text, recommendation section with highlighted call-to-action button if applicable, supporting chart using LineChart from react-native-chart-kit if correlation/prediction type (show related metric data), metadata card showing confidence score and expiration. Feedback section at bottom: 'Was this helpful?' with thumbs up/down TouchableOpacity buttons that call submitFeedback API and show confirmation state. Style feedback buttons with colors.status.success/error on selection. Handle loading, error states. Use responsive spacing from useResponsive hook. Register in app/_layout.tsx.

### 9.5. Implement swipe-to-dismiss on InsightCard and add unread badge to tab bar

**Status:** pending  
**Dependencies:** 9.1, 9.2, 9.3  

Add swipe gesture to dismiss insights from feed and integrate unread insight count badge on the tab bar navigation

**Details:**

Update lib/components/insights/InsightCard.tsx to wrap content in Swipeable from react-native-gesture-handler following SwipeableMealCard.tsx pattern. Implement renderRightActions showing dismiss action (X icon, 'Dismiss' label, error color background). On swipe complete, call insightsApi.dismiss(id) and trigger parent refresh via onDismiss callback. For tab bar badge: modify app/(tabs)/_layout.tsx to add Insights tab with IconSymbol 'lightbulb.fill'. Create useInsightsBadge hook that polls insightsApi.getUnreadCount() every 60 seconds and on focus. Pass badge count to tabBarBadge option (null if 0). Style badge with colors.status.error background. Add long-press handler on InsightCard showing action sheet (View Details, Dismiss, Cancel) using showAlert utility. Ensure proper cleanup of polling interval on unmount.
