import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { supplementLogsApi } from '@/lib/api/supplements';
import { SupplementLog, SUPPLEMENT_CATEGORIES } from '@/lib/types/supplements';
import { useAuth } from '@/lib/context/AuthContext';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

interface GroupedLogs {
  date: string;
  dateLabel: string;
  logs: SupplementLog[];
}

export default function SupplementHistoryScreen() {
  const [logs, setLogs] = useState<SupplementLog[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { user } = useAuth();
  const router = useRouter();

  const loadLogs = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      // Get logs from the last 30 days
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 30);

      const data = await supplementLogsApi.getByDateRange(startDate, endDate);
      setLogs(data);
    } catch (error) {
      console.error('Failed to load logs:', error);
      showAlert('Error', getErrorMessage(error, 'Failed to load history'));
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadLogs();
    setRefreshing(false);
  }, [loadLogs]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  // Group logs by date
  const groupedLogs: GroupedLogs[] = logs.reduce((groups: GroupedLogs[], log) => {
    const dateObj = new Date(log.takenAt);
    const dateKey = dateObj.toDateString();

    let group = groups.find((g) => g.date === dateKey);
    if (!group) {
      const today = new Date().toDateString();
      const yesterday = new Date(Date.now() - 86400000).toDateString();

      let dateLabel = dateObj.toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'short',
        day: 'numeric',
      });

      if (dateKey === today) {
        dateLabel = 'Today';
      } else if (dateKey === yesterday) {
        dateLabel = 'Yesterday';
      }

      group = { date: dateKey, dateLabel, logs: [] };
      groups.push(group);
    }

    group.logs.push(log);
    return groups;
  }, []);

  // Sort groups by date (most recent first)
  groupedLogs.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  const getCategoryLabel = (category: string) => {
    return SUPPLEMENT_CATEGORIES.find((c) => c.value === category)?.label || category;
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const handleDeleteLog = (logId: string) => {
    showAlert(
      'Delete Log',
      'Are you sure you want to delete this log entry?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await supplementLogsApi.delete(logId);
              setLogs(logs.filter((l) => l.id !== logId));
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to delete log'));
            }
          },
        },
      ]
    );
  };

  const renderLogItem = ({ item }: { item: SupplementLog }) => (
    <TouchableOpacity
      style={styles.logItem}
      onLongPress={() => handleDeleteLog(item.id)}
    >
      <View style={styles.logIcon}>
        <Ionicons name="medical" size={20} color={colors.primary.main} />
      </View>
      <View style={styles.logInfo}>
        <Text style={styles.logName}>{item.supplement.name}</Text>
        <Text style={styles.logDetails}>
          {item.dosage} {item.unit}
          {item.source === 'QUICK_LOG' && ' • Quick Log'}
          {item.source === 'SCHEDULED' && ' • Scheduled'}
        </Text>
      </View>
      <View style={styles.logTime}>
        <Text style={styles.logTimeText}>{formatTime(item.takenAt)}</Text>
      </View>
    </TouchableOpacity>
  );

  const renderGroup = ({ item }: { item: GroupedLogs }) => (
    <View style={styles.group}>
      <Text style={styles.groupDate}>{item.dateLabel}</Text>
      {item.logs.map((log) => (
        <View key={log.id}>{renderLogItem({ item: log })}</View>
      ))}
    </View>
  );

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={24} color={colors.text.secondary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>History</Text>
          <View style={{ width: 24 }} />
        </View>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color={colors.text.secondary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>History</Text>
        <View style={{ width: 24 }} />
      </View>

      {groupedLogs.length === 0 ? (
        <View style={styles.emptyState}>
          <Ionicons name="time-outline" size={64} color={colors.text.disabled} />
          <Text style={styles.emptyTitle}>No History Yet</Text>
          <Text style={styles.emptyText}>
            Your supplement intake history will appear here
          </Text>
        </View>
      ) : (
        <FlatList
          data={groupedLogs}
          keyExtractor={(item) => item.date}
          renderItem={renderGroup}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={colors.primary.main}
              colors={[colors.primary.main]}
            />
          }
        />
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>Long press on a log to delete</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listContent: {
    padding: spacing.lg,
    paddingBottom: spacing['3xl'],
  },

  // Groups
  group: {
    marginBottom: spacing.xl,
  },
  groupDate: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.3,
  },

  // Log Items
  logItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  logIcon: {
    width: 40,
    height: 40,
    borderRadius: borderRadius.full,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  logInfo: {
    flex: 1,
  },
  logName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  logDetails: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  logTime: {
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  logTimeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },

  // Empty State
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  emptyTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
  },

  // Footer
  footer: {
    padding: spacing.md,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  footerText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
});
