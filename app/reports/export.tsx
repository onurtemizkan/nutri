import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, Share } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { reportsApi } from '@/lib/api/reports';
import { ReportExportFormat } from '@/lib/types/reports';
import { getErrorMessage } from '@/lib/utils/errorHandling';

type ExportType = 'weekly' | 'monthly';

export default function ExportScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ type: ExportType; date?: string; month?: string }>();
  const [isExporting, setIsExporting] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<ReportExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const reportType = params.type || 'weekly';
  const dateParam = params.type === 'monthly' ? params.month : params.date;

  const handleClose = useCallback(() => {
    router.back();
  }, [router]);

  const handleExport = useCallback(
    async (format: ReportExportFormat) => {
      setSelectedFormat(format);
      setIsExporting(true);
      setError(null);
      setSuccess(false);

      try {
        let result;
        if (reportType === 'weekly') {
          result = await reportsApi.exportWeeklyReport(format, dateParam);
        } else {
          result = await reportsApi.exportMonthlyReport(format, dateParam);
        }

        if (result.success) {
          if (format === 'json') {
            // For JSON, offer to share the data
            const jsonString = JSON.stringify(result.data, null, 2);
            await Share.share({
              message: jsonString,
              title: result.filename,
            });
            setSuccess(true);
          } else {
            // For PDF/image, show success message (actual export will be implemented with Puppeteer)
            setSuccess(true);
          }
        } else {
          setError(result.message || 'Export failed');
        }
      } catch (err) {
        setError(getErrorMessage(err, 'Failed to export report'));
      } finally {
        setIsExporting(false);
      }
    },
    [reportType, dateParam]
  );

  const formatOptions: {
    format: ReportExportFormat;
    icon: string;
    label: string;
    description: string;
  }[] = [
    {
      format: 'json',
      icon: 'code-slash-outline',
      label: 'JSON Data',
      description: 'Export raw data for analysis',
    },
    {
      format: 'pdf',
      icon: 'document-text-outline',
      label: 'PDF Document',
      description: 'Print-ready report (Coming Soon)',
    },
    {
      format: 'image',
      icon: 'image-outline',
      label: 'Image',
      description: 'Share on social media (Coming Soon)',
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleClose} style={styles.closeButton}>
          <Ionicons name="close" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Export Report</Text>
        <View style={styles.headerRight} />
      </View>

      <View style={styles.content}>
        <Text style={styles.subtitle}>
          Export your {reportType} report in your preferred format
        </Text>

        {/* Format Options */}
        <View style={styles.formatList}>
          {formatOptions.map((option) => {
            const isSelected = selectedFormat === option.format;
            const isDisabled = option.format !== 'json'; // Only JSON is currently functional

            return (
              <TouchableOpacity
                key={option.format}
                style={[
                  styles.formatCard,
                  isSelected && styles.formatCardSelected,
                  isDisabled && styles.formatCardDisabled,
                ]}
                onPress={() => !isDisabled && handleExport(option.format)}
                disabled={isExporting || isDisabled}
              >
                <View style={styles.formatIcon}>
                  <Ionicons
                    name={option.icon as keyof typeof Ionicons.glyphMap}
                    size={28}
                    color={isDisabled ? colors.text.disabled : colors.primary.main}
                  />
                </View>
                <View style={styles.formatInfo}>
                  <Text style={[styles.formatLabel, isDisabled && styles.formatLabelDisabled]}>
                    {option.label}
                  </Text>
                  <Text style={styles.formatDescription}>{option.description}</Text>
                </View>
                {isSelected && isExporting && (
                  <ActivityIndicator size="small" color={colors.primary.main} />
                )}
                {isSelected && success && (
                  <Ionicons name="checkmark-circle" size={24} color={colors.semantic.success} />
                )}
                {!isDisabled && !isSelected && (
                  <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
                )}
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Error Message */}
        {error && (
          <View style={styles.errorContainer}>
            <Ionicons name="warning" size={20} color={colors.semantic.error} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Success Message */}
        {success && (
          <View style={styles.successContainer}>
            <Ionicons name="checkmark-circle" size={20} color={colors.semantic.success} />
            <Text style={styles.successText}>Export successful!</Text>
          </View>
        )}

        {/* Info Card */}
        <View style={styles.infoCard}>
          <Ionicons name="information-circle-outline" size={20} color={colors.text.tertiary} />
          <Text style={styles.infoText}>
            PDF and image exports are coming soon. They will allow you to create beautiful visual
            reports to share with others.
          </Text>
        </View>
      </View>

      {/* Close Button */}
      <View style={styles.footer}>
        <TouchableOpacity style={styles.doneButton} onPress={handleClose}>
          <LinearGradient
            colors={gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.doneButtonGradient}
          >
            <Text style={styles.doneButtonText}>Done</Text>
          </LinearGradient>
        </TouchableOpacity>
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
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  closeButton: {
    padding: spacing.sm,
  },
  headerTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  headerRight: {
    width: 40,
  },
  content: {
    flex: 1,
    padding: spacing.lg,
  },
  subtitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  formatList: {
    gap: spacing.md,
  },
  formatCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  formatCardSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  formatCardDisabled: {
    opacity: 0.6,
  },
  formatIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.background.elevated,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  formatInfo: {
    flex: 1,
  },
  formatLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  formatLabelDisabled: {
    color: colors.text.tertiary,
  },
  formatDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  errorContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.special.errorLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.lg,
  },
  errorText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.semantic.error,
  },
  successContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.special.successLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.lg,
  },
  successText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.semantic.success,
    fontWeight: typography.fontWeight.medium,
  },
  infoCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.xl,
  },
  infoText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: 20,
  },
  footer: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  doneButton: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
  },
  doneButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  doneButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
});
