import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '@/lib/context/AuthContext';
import { useRouter } from 'expo-router';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

export default function ProfileScreen() {
  const { user, logout, updateUser } = useAuth();
  const router = useRouter();
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [goalCalories, setGoalCalories] = useState(user?.goalCalories.toString() || '2000');
  const [goalProtein, setGoalProtein] = useState(user?.goalProtein.toString() || '150');
  const [goalCarbs, setGoalCarbs] = useState(user?.goalCarbs.toString() || '200');
  const [goalFat, setGoalFat] = useState(user?.goalFat.toString() || '65');

  const handleSaveGoals = async () => {
    setIsLoading(true);
    try {
      await updateUser({
        goalCalories: parseInt(goalCalories),
        goalProtein: parseFloat(goalProtein),
        goalCarbs: parseFloat(goalCarbs),
        goalFat: parseFloat(goalFat),
      });

      showAlert('Success', 'Goals updated successfully!');
      setIsEditing(false);
    } catch {
      showAlert('Error', 'Failed to update goals');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    showAlert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        style: 'destructive',
        onPress: async () => {
          await logout();
          router.replace('/auth/welcome');
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.container} testID="profile-screen">
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          {/* Profile Header */}
          <View style={styles.profileHeader} testID="profile-header">
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>
                {user?.name.charAt(0).toUpperCase()}
              </Text>
            </LinearGradient>
            <Text style={styles.name}>{user?.name}</Text>
            <Text style={styles.email}>{user?.email}</Text>
          </View>

          {/* Daily Goals */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Daily Goals</Text>
              {!isEditing && (
                <TouchableOpacity onPress={() => setIsEditing(true)} testID="profile-edit-goals-button">
                  <Text style={styles.editButton}>Edit</Text>
                </TouchableOpacity>
              )}
            </View>

            {isEditing ? (
              <>
                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Calorie Goal</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={goalCalories}
                      onChangeText={setGoalCalories}
                      keyboardType="numeric"
                      editable={!isLoading}
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Protein Goal (g)</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={goalProtein}
                      onChangeText={setGoalProtein}
                      keyboardType="numeric"
                      editable={!isLoading}
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Carbs Goal (g)</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={goalCarbs}
                      onChangeText={setGoalCarbs}
                      keyboardType="numeric"
                      editable={!isLoading}
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Fat Goal (g)</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      value={goalFat}
                      onChangeText={setGoalFat}
                      keyboardType="numeric"
                      editable={!isLoading}
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                </View>

                <View style={styles.buttonRow}>
                  <TouchableOpacity
                    style={styles.cancelButton}
                    onPress={() => {
                      setIsEditing(false);
                      setGoalCalories(user?.goalCalories.toString() || '2000');
                      setGoalProtein(user?.goalProtein.toString() || '150');
                      setGoalCarbs(user?.goalCarbs.toString() || '200');
                      setGoalFat(user?.goalFat.toString() || '65');
                    }}
                    disabled={isLoading}
                    activeOpacity={0.8}
                    testID="profile-cancel-edit-button"
                  >
                    <Text style={styles.cancelButtonText}>Cancel</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.saveButton}
                    onPress={handleSaveGoals}
                    disabled={isLoading}
                    activeOpacity={0.8}
                    testID="profile-save-goals-button"
                  >
                    <LinearGradient
                      colors={isLoading ? [colors.text.disabled, colors.text.disabled] : gradients.primary}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={styles.saveButtonGradient}
                    >
                      {isLoading ? (
                        <ActivityIndicator color={colors.text.primary} />
                      ) : (
                        <Text style={styles.saveButtonText}>Save</Text>
                      )}
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              </>
            ) : (
              <>
                <View style={styles.goalItem}>
                  <Text style={styles.goalLabel}>Calories</Text>
                  <Text style={styles.goalValue}>{user?.goalCalories}</Text>
                </View>

                <View style={styles.goalItem}>
                  <Text style={styles.goalLabel}>Protein</Text>
                  <Text style={styles.goalValue}>{user?.goalProtein}g</Text>
                </View>

                <View style={styles.goalItem}>
                  <Text style={styles.goalLabel}>Carbs</Text>
                  <Text style={styles.goalValue}>{user?.goalCarbs}g</Text>
                </View>

                <View style={styles.goalItem}>
                  <Text style={styles.goalLabel}>Fat</Text>
                  <Text style={styles.goalValue}>{user?.goalFat}g</Text>
                </View>
              </>
            )}
          </View>

          {/* Health Integration */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Health Integration</Text>

            <TouchableOpacity
              style={styles.menuItemWithArrow}
              onPress={() => router.push('/health-settings')}
              accessibilityLabel="Open health settings"
              testID="health-integration-button"
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuItemIcon}>
                  <Ionicons
                    name="heart"
                    size={20}
                    color={Platform.OS === 'ios' ? '#FF2D55' : colors.text.tertiary}
                  />
                </View>
                <View>
                  <Text style={styles.menuItemText}>Apple Health</Text>
                  <Text style={styles.menuItemSubtext}>
                    {Platform.OS === 'ios' ? 'Sync health metrics' : 'iOS only'}
                  </Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
          </View>

          {/* Account */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Account</Text>

            <TouchableOpacity style={styles.menuItem} onPress={handleLogout} testID="profile-logout-button">
              <Text style={[styles.menuItemText, styles.logoutText]}>Logout</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
  },

  // Profile Header
  profileHeader: {
    alignItems: 'center',
    paddingVertical: spacing['2xl'],
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
    ...shadows.glow,
  },
  avatarText: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  name: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    letterSpacing: -0.5,
  },
  email: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },

  // Section
  section: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    letterSpacing: -0.3,
  },
  editButton: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Goals
  goalItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  goalLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  goalValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Input
  inputContainer: {
    marginBottom: spacing.md,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.3,
  },
  inputWrapper: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  input: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 48,
  },

  // Buttons
  buttonRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.sm,
  },
  cancelButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: colors.border.primary,
    height: 48,
  },
  cancelButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  saveButton: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    ...shadows.md,
  },
  saveButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    height: 48,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Menu
  menuItem: {
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  menuItemWithArrow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  menuItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  menuItemIcon: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.elevated,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  menuItemText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  menuItemSubtext: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  logoutText: {
    color: colors.status.error,
    fontWeight: typography.fontWeight.semibold,
  },
});
