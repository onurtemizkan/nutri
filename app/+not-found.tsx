import { Link } from 'expo-router';
import { StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useResponsive } from '@/hooks/useResponsive';
import { spacing } from '@/lib/theme/colors';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function NotFoundScreen() {
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  return (
    <ThemedView style={styles.container}>
        <View style={[
          styles.content,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.contentTablet
        ]}>
          <ThemedText type="title">This screen doesn't exist.</ThemedText>
          <Link href="/" style={styles.link}>
            <ThemedText type="link">Go to home screen!</ThemedText>
          </Link>
        </View>
      </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  contentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    width: '100%',
  },
  link: {
    marginTop: spacing.md,
    paddingVertical: spacing.md,
  },
});
