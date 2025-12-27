import { useState, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Platform,
  Modal,
  Image,
  ActionSheetIOS,
} from 'react-native';
import type { SubscriptionTier } from '@/lib/types';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { useAuth } from '@/lib/context/AuthContext';
import { useRouter } from 'expo-router';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { useResponsive } from '@/hooks/useResponsive';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function ProfileScreen() {
  const { user, logout, updateUser, deleteAccount } = useAuth();
  const router = useRouter();
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { isTablet, getResponsiveValue } = useResponsive();

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });
  const avatarSize = getResponsiveValue({
    small: 70,
    medium: 80,
    large: 90,
    tablet: 100,
    default: 80,
  });
  const maxContentWidth = getResponsiveValue({
    small: undefined,
    medium: undefined,
    large: 600,
    tablet: 500,
    default: undefined,
  });

  const [goalCalories, setGoalCalories] = useState(user?.goalCalories.toString() || '2000');
  const [goalProtein, setGoalProtein] = useState(user?.goalProtein.toString() || '150');
  const [goalCarbs, setGoalCarbs] = useState(user?.goalCarbs.toString() || '200');
  const [goalFat, setGoalFat] = useState(user?.goalFat.toString() || '65');

  // Delete account state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirmation, setDeleteConfirmation] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  // Profile picture state
  const [isUploadingPicture, setIsUploadingPicture] = useState(false);
  const [showPictureModal, setShowPictureModal] = useState(false);

  const processAndUploadImage = async (uri: string) => {
    setIsUploadingPicture(true);
    try {
      // Resize and compress the image
      const manipulatedImage = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 400, height: 400 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );

      if (!manipulatedImage.base64) {
        throw new Error('Failed to process image');
      }

      // Create data URL
      const dataUrl = `data:image/jpeg;base64,${manipulatedImage.base64}`;

      // Update user profile with the new picture
      await updateUser({ profilePicture: dataUrl });
      showAlert('Success', 'Profile picture updated!');
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to update profile picture'));
    } finally {
      setIsUploadingPicture(false);
      setShowPictureModal(false);
    }
  };

  const pickImageFromLibrary = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      showAlert('Permission Required', 'Please allow access to your photo library to select a profile picture.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      await processAndUploadImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      showAlert('Permission Required', 'Please allow access to your camera to take a profile picture.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      await processAndUploadImage(result.assets[0].uri);
    }
  };

  const removeProfilePicture = async () => {
    setIsUploadingPicture(true);
    try {
      await updateUser({ profilePicture: null });
      showAlert('Success', 'Profile picture removed!');
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to remove profile picture'));
    } finally {
      setIsUploadingPicture(false);
      setShowPictureModal(false);
    }
  };

  const handleAvatarPress = () => {
    if (Platform.OS === 'ios') {
      const options = user?.profilePicture
        ? ['Take Photo', 'Choose from Library', 'Remove Photo', 'Cancel']
        : ['Take Photo', 'Choose from Library', 'Cancel'];
      const destructiveButtonIndex = user?.profilePicture ? 2 : undefined;
      const cancelButtonIndex = user?.profilePicture ? 3 : 2;

      ActionSheetIOS.showActionSheetWithOptions(
        {
          options,
          cancelButtonIndex,
          destructiveButtonIndex,
        },
        (buttonIndex) => {
          if (buttonIndex === 0) {
            takePhoto();
          } else if (buttonIndex === 1) {
            pickImageFromLibrary();
          } else if (buttonIndex === 2 && user?.profilePicture) {
            removeProfilePicture();
          }
        }
      );
    } else {
      setShowPictureModal(true);
    }
  };

  const handleDeleteAccount = async () => {
    if (deleteConfirmation !== 'DELETE') {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteAccount();
      setShowDeleteModal(false);
      router.replace('/auth/signin');
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to delete account. Please try again.'));
    } finally {
      setIsDeleting(false);
    }
  };

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

  // Subscription info
  const subscriptionInfo = useMemo(() => {
    const tier: SubscriptionTier = user?.subscriptionTier || 'FREE';
    const endDate = user?.subscriptionEndDate ? new Date(user.subscriptionEndDate) : null;
    const now = new Date();

    // Calculate remaining days for trial
    let remainingDays = 0;
    if (tier === 'PRO_TRIAL' && endDate) {
      remainingDays = Math.max(0, Math.ceil((endDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24)));
    }

    // Format end date
    const formattedEndDate = endDate
      ? endDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      : null;

    // Format price
    const formattedPrice = user?.subscriptionPrice
      ? `$${user.subscriptionPrice.toFixed(2)}`
      : null;

    // Billing cycle label
    const billingCycleLabel = user?.subscriptionBillingCycle === 'ANNUAL' ? 'Annual' : 'Monthly';

    return {
      tier,
      remainingDays,
      formattedEndDate,
      formattedPrice,
      billingCycleLabel,
      isExpired: endDate ? endDate < now : false,
    };
  }, [user?.subscriptionTier, user?.subscriptionEndDate, user?.subscriptionPrice, user?.subscriptionBillingCycle]);

  const getSubscriptionBadgeStyle = (tier: SubscriptionTier) => {
    switch (tier) {
      case 'PRO':
        return { backgroundColor: colors.primary.main };
      case 'PRO_TRIAL':
        return { backgroundColor: colors.status.warning };
      default:
        return { backgroundColor: colors.text.disabled };
    }
  };

  const getSubscriptionLabel = (tier: SubscriptionTier) => {
    switch (tier) {
      case 'PRO':
        return 'Pro';
      case 'PRO_TRIAL':
        return 'Pro Trial';
      default:
        return 'Free';
    }
  };

  return (
    <SafeAreaView style={styles.container} testID="profile-screen">
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={[
          styles.content,
          { padding: contentPadding },
          maxContentWidth && { maxWidth: maxContentWidth, alignSelf: 'center', width: '100%' }
        ]}>
          {/* Profile Header */}
          <View style={styles.profileHeader} testID="profile-header">
            <TouchableOpacity
              onPress={handleAvatarPress}
              disabled={isUploadingPicture}
              activeOpacity={0.8}
              accessibilityRole="button"
              accessibilityLabel="Change profile picture"
              accessibilityHint="Double tap to take or choose a new profile picture"
              accessibilityState={{ disabled: isUploadingPicture, busy: isUploadingPicture }}
              testID="profile-avatar-button"
            >
              <View style={styles.avatarContainer}>
                {user?.profilePicture ? (
                  <Image
                    source={{ uri: user.profilePicture }}
                    style={[
                      styles.avatarImage,
                      { width: avatarSize, height: avatarSize, borderRadius: avatarSize / 2 }
                    ]}
                  />
                ) : (
                  <LinearGradient
                    colors={gradients.primary}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[
                      styles.avatar,
                      { width: avatarSize, height: avatarSize, borderRadius: avatarSize / 2 }
                    ]}
                  >
                    <Text style={styles.avatarText}>
                      {user?.name.charAt(0).toUpperCase()}
                    </Text>
                  </LinearGradient>
                )}
                {isUploadingPicture ? (
                  <View style={[styles.avatarOverlay, { borderRadius: avatarSize / 2 }]}>
                    <ActivityIndicator color={colors.text.primary} />
                  </View>
                ) : (
                  <View style={styles.avatarEditBadge}>
                    <Ionicons name="camera" size={14} color={colors.text.primary} />
                  </View>
                )}
              </View>
            </TouchableOpacity>
            <Text style={styles.name}>{user?.name}</Text>
            <Text style={styles.email}>{user?.email}</Text>
          </View>

          {/* Subscription */}
          <View style={styles.section} testID="subscription-section">
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Subscription</Text>
            </View>

            <View style={styles.subscriptionCard}>
              <View style={styles.subscriptionHeader}>
                <View style={[styles.subscriptionBadge, getSubscriptionBadgeStyle(subscriptionInfo.tier)]}>
                  <Text style={styles.subscriptionBadgeText}>
                    {getSubscriptionLabel(subscriptionInfo.tier)}
                  </Text>
                </View>
                {subscriptionInfo.tier === 'PRO' && (
                  <View style={styles.subscriptionCycleBadge}>
                    <Text style={styles.subscriptionCycleText}>{subscriptionInfo.billingCycleLabel}</Text>
                  </View>
                )}
              </View>

              {/* Free tier */}
              {subscriptionInfo.tier === 'FREE' && (
                <Text style={styles.subscriptionDescription}>
                  Basic features included
                </Text>
              )}

              {/* Pro Trial */}
              {subscriptionInfo.tier === 'PRO_TRIAL' && (
                <>
                  <View style={styles.subscriptionDetailRow}>
                    <Ionicons name="time-outline" size={16} color={colors.text.tertiary} />
                    <Text style={styles.subscriptionDetailText}>
                      {subscriptionInfo.remainingDays > 0
                        ? `${subscriptionInfo.remainingDays} day${subscriptionInfo.remainingDays !== 1 ? 's' : ''} remaining`
                        : 'Trial expired'}
                    </Text>
                  </View>
                  {subscriptionInfo.formattedEndDate && (
                    <View style={styles.subscriptionDetailRow}>
                      <Ionicons name="calendar-outline" size={16} color={colors.text.tertiary} />
                      <Text style={styles.subscriptionDetailText}>
                        Ends {subscriptionInfo.formattedEndDate}
                      </Text>
                    </View>
                  )}
                </>
              )}

              {/* Pro */}
              {subscriptionInfo.tier === 'PRO' && (
                <>
                  {subscriptionInfo.formattedEndDate && (
                    <View style={styles.subscriptionDetailRow}>
                      <Ionicons name="calendar-outline" size={16} color={colors.text.tertiary} />
                      <Text style={styles.subscriptionDetailText}>
                        {subscriptionInfo.isExpired ? 'Expired' : 'Renews'} {subscriptionInfo.formattedEndDate}
                      </Text>
                    </View>
                  )}
                  {subscriptionInfo.formattedPrice && (
                    <View style={styles.subscriptionDetailRow}>
                      <Ionicons name="card-outline" size={16} color={colors.text.tertiary} />
                      <Text style={styles.subscriptionDetailText}>
                        {subscriptionInfo.formattedPrice}/{subscriptionInfo.billingCycleLabel === 'Annual' ? 'year' : 'month'}
                      </Text>
                    </View>
                  )}
                </>
              )}
            </View>
          </View>

          {/* Daily Goals */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Daily Goals</Text>
              {!isEditing && (
                <TouchableOpacity
                  onPress={() => setIsEditing(true)}
                  accessibilityRole="button"
                  accessibilityLabel="Edit daily goals"
                  accessibilityHint="Double tap to edit your nutrition goals"
                  testID="profile-edit-goals-button"
                >
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
                      accessibilityLabel="Calorie goal"
                      accessibilityHint="Enter your daily calorie goal"
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
                      accessibilityLabel="Protein goal in grams"
                      accessibilityHint="Enter your daily protein goal"
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
                      accessibilityLabel="Carbohydrates goal in grams"
                      accessibilityHint="Enter your daily carbohydrates goal"
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
                      accessibilityLabel="Fat goal in grams"
                      accessibilityHint="Enter your daily fat goal"
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
                    accessibilityRole="button"
                    accessibilityLabel="Cancel editing"
                    accessibilityHint="Double tap to discard changes and stop editing"
                    accessibilityState={{ disabled: isLoading }}
                    testID="profile-cancel-edit-button"
                  >
                    <Text style={styles.cancelButtonText}>Cancel</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.saveButton}
                    onPress={handleSaveGoals}
                    disabled={isLoading}
                    activeOpacity={0.8}
                    accessibilityRole="button"
                    accessibilityLabel={isLoading ? 'Saving goals' : 'Save goals'}
                    accessibilityHint="Double tap to save your nutrition goals"
                    accessibilityState={{ disabled: isLoading, busy: isLoading }}
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
              accessibilityRole="button"
              accessibilityLabel="Apple Health settings"
              accessibilityHint="Double tap to configure health data sync"
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

          {/* Notifications */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Notifications</Text>

            <TouchableOpacity
              style={[styles.menuItemWithArrow, styles.menuItemLast]}
              onPress={() => router.push('/notification-settings')}
              accessibilityRole="button"
              accessibilityLabel="Notification preferences"
              accessibilityHint="Double tap to manage reminders and alerts"
              testID="notification-settings-button"
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuItemIcon}>
                  <Ionicons
                    name="notifications"
                    size={20}
                    color={colors.primary.main}
                  />
                </View>
                <View>
                  <Text style={styles.menuItemText}>Notification Preferences</Text>
                  <Text style={styles.menuItemSubtext}>Manage reminders and alerts</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
          </View>

          {/* Supplements */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Supplements</Text>

            <TouchableOpacity
              style={[styles.menuItemWithArrow, styles.menuItemLast]}
              onPress={() => router.push('/supplements')}
              accessibilityRole="button"
              accessibilityLabel="Manage supplements"
              accessibilityHint="Double tap to track your daily supplements"
              testID="supplements-button"
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuItemIcon}>
                  <Ionicons
                    name="medical"
                    size={20}
                    color={colors.primary.main}
                  />
                </View>
                <View>
                  <Text style={styles.menuItemText}>Manage Supplements</Text>
                  <Text style={styles.menuItemSubtext}>Track your daily supplements</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
          </View>

          {/* Legal */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Legal</Text>

            <TouchableOpacity
              style={styles.menuItemWithArrow}
              onPress={() => router.push('/terms')}
              accessibilityRole="link"
              accessibilityLabel="Terms and Conditions"
              accessibilityHint="Double tap to view terms of service"
              testID="terms-conditions-button"
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuItemIcon}>
                  <Ionicons
                    name="document-text"
                    size={20}
                    color={colors.text.tertiary}
                  />
                </View>
                <View>
                  <Text style={styles.menuItemText}>Terms & Conditions</Text>
                  <Text style={styles.menuItemSubtext}>View terms of service</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.menuItemWithArrow}
              onPress={() => router.push('/privacy')}
              accessibilityRole="link"
              accessibilityLabel="Privacy Policy"
              accessibilityHint="Double tap to view how we handle your data"
              testID="privacy-policy-button"
            >
              <View style={styles.menuItemLeft}>
                <View style={styles.menuItemIcon}>
                  <Ionicons
                    name="shield-checkmark"
                    size={20}
                    color={colors.text.tertiary}
                  />
                </View>
                <View>
                  <Text style={styles.menuItemText}>Privacy Policy</Text>
                  <Text style={styles.menuItemSubtext}>How we handle your data</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
          </View>

          {/* Account */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Account</Text>

            <TouchableOpacity
              style={styles.menuItem}
              onPress={handleLogout}
              accessibilityRole="button"
              accessibilityLabel="Logout"
              accessibilityHint="Double tap to sign out of your account"
              testID="profile-logout-button"
            >
              <Text style={[styles.menuItemText, styles.logoutText]}>Logout</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.menuItem, styles.menuItemLast]}
              onPress={() => setShowDeleteModal(true)}
              accessibilityRole="button"
              accessibilityLabel="Delete account"
              accessibilityHint="Double tap to permanently delete your account and all data"
              testID="profile-delete-account-button"
            >
              <Text style={[styles.menuItemText, styles.deleteText]}>Delete Account</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>

      {/* Delete Account Confirmation Modal */}
      <Modal
        visible={showDeleteModal}
        transparent
        animationType="fade"
        onRequestClose={() => {
          if (!isDeleting) {
            setShowDeleteModal(false);
            setDeleteConfirmation('');
          }
        }}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <View style={styles.modalWarningIcon}>
                <Ionicons name="warning" size={32} color={colors.status.error} />
              </View>
              <Text style={styles.modalTitle}>Delete Account</Text>
            </View>

            <Text style={styles.modalDescription}>
              This action is permanent and cannot be undone. All your data will be permanently deleted, including:
            </Text>

            <View style={styles.modalList}>
              <Text style={styles.modalListItem}>• All your meals and nutrition logs</Text>
              <Text style={styles.modalListItem}>• Health metrics and activity data</Text>
              <Text style={styles.modalListItem}>• ML insights and predictions</Text>
              <Text style={styles.modalListItem}>• Account settings and preferences</Text>
            </View>

            <Text style={styles.modalConfirmText}>
              Type <Text style={styles.modalConfirmHighlight}>DELETE</Text> to confirm:
            </Text>

            <View style={styles.modalInputWrapper}>
              <TextInput
                style={styles.modalInput}
                value={deleteConfirmation}
                onChangeText={setDeleteConfirmation}
                placeholder="Type DELETE"
                placeholderTextColor={colors.text.disabled}
                autoCapitalize="characters"
                autoCorrect={false}
                editable={!isDeleting}
                accessibilityLabel="Delete confirmation"
                accessibilityHint="Type DELETE in capital letters to confirm account deletion"
                testID="delete-confirmation-input"
              />
            </View>

            <View style={styles.modalButtonRow}>
              <TouchableOpacity
                style={styles.modalCancelButton}
                onPress={() => {
                  setShowDeleteModal(false);
                  setDeleteConfirmation('');
                }}
                disabled={isDeleting}
                accessibilityRole="button"
                accessibilityLabel="Cancel"
                accessibilityHint="Double tap to cancel and close this dialog"
                accessibilityState={{ disabled: isDeleting }}
                testID="delete-cancel-button"
              >
                <Text style={styles.modalCancelButtonText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.modalDeleteButton,
                  deleteConfirmation !== 'DELETE' && styles.modalDeleteButtonDisabled,
                ]}
                onPress={handleDeleteAccount}
                disabled={deleteConfirmation !== 'DELETE' || isDeleting}
                accessibilityRole="button"
                accessibilityLabel={isDeleting ? 'Deleting account' : 'Delete account forever'}
                accessibilityHint={deleteConfirmation !== 'DELETE' ? 'Type DELETE above to enable this button' : 'Double tap to permanently delete your account'}
                accessibilityState={{ disabled: deleteConfirmation !== 'DELETE' || isDeleting, busy: isDeleting }}
                testID="delete-confirm-button"
              >
                {isDeleting ? (
                  <ActivityIndicator color={colors.text.primary} size="small" />
                ) : (
                  <Text style={styles.modalDeleteButtonText}>Delete Forever</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Profile Picture Modal (Android) */}
      <Modal
        visible={showPictureModal}
        transparent
        animationType="fade"
        onRequestClose={() => {
          if (!isUploadingPicture) {
            setShowPictureModal(false);
          }
        }}
      >
        <TouchableOpacity
          style={styles.pictureModalOverlay}
          activeOpacity={1}
          onPress={() => !isUploadingPicture && setShowPictureModal(false)}
        >
          <View style={styles.pictureModalContent}>
            <Text style={styles.pictureModalTitle}>Change Profile Picture</Text>

            <TouchableOpacity
              style={styles.pictureModalOption}
              onPress={() => {
                setShowPictureModal(false);
                takePhoto();
              }}
              disabled={isUploadingPicture}
              accessibilityRole="button"
              accessibilityLabel="Take photo"
              accessibilityHint="Double tap to take a new profile picture with camera"
              accessibilityState={{ disabled: isUploadingPicture }}
            >
              <View style={styles.pictureModalOptionIcon}>
                <Ionicons name="camera" size={22} color={colors.primary.main} />
              </View>
              <Text style={styles.pictureModalOptionText}>Take Photo</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.pictureModalOption}
              onPress={() => {
                setShowPictureModal(false);
                pickImageFromLibrary();
              }}
              disabled={isUploadingPicture}
              accessibilityRole="button"
              accessibilityLabel="Choose from library"
              accessibilityHint="Double tap to select a photo from your library"
              accessibilityState={{ disabled: isUploadingPicture }}
            >
              <View style={styles.pictureModalOptionIcon}>
                <Ionicons name="images" size={22} color={colors.primary.main} />
              </View>
              <Text style={styles.pictureModalOptionText}>Choose from Library</Text>
            </TouchableOpacity>

            {user?.profilePicture && (
              <TouchableOpacity
                style={styles.pictureModalOption}
                onPress={removeProfilePicture}
                disabled={isUploadingPicture}
                accessibilityRole="button"
                accessibilityLabel="Remove photo"
                accessibilityHint="Double tap to remove your current profile picture"
                accessibilityState={{ disabled: isUploadingPicture }}
              >
                <View style={styles.pictureModalOptionIcon}>
                  <Ionicons name="trash" size={22} color={colors.status.error} />
                </View>
                <Text style={[styles.pictureModalOptionText, styles.pictureModalRemoveText]}>
                  Remove Photo
                </Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity
              style={styles.pictureModalCancelButton}
              onPress={() => setShowPictureModal(false)}
              disabled={isUploadingPicture}
              accessibilityRole="button"
              accessibilityLabel="Cancel"
              accessibilityHint="Double tap to close this menu"
              accessibilityState={{ disabled: isUploadingPicture }}
            >
              <Text style={styles.pictureModalCancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>
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
    // padding applied dynamically
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

  // Subscription
  subscriptionCard: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    padding: spacing.md,
  },
  subscriptionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  subscriptionBadge: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  subscriptionBadgeText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  subscriptionCycleBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.tertiary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  subscriptionCycleText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
  subscriptionDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  subscriptionDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.xs,
  },
  subscriptionDetailText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
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
  deleteText: {
    color: colors.status.error,
    fontWeight: typography.fontWeight.semibold,
  },
  menuItemLast: {
    borderBottomWidth: 0,
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  modalContent: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    width: '100%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  modalHeader: {
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  modalWarningIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: `${colors.status.error}20`,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  modalTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
  },
  modalDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
    marginBottom: spacing.md,
  },
  modalList: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  modalListItem: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  modalConfirmText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  modalConfirmHighlight: {
    color: colors.status.error,
    fontWeight: typography.fontWeight.bold,
  },
  modalInputWrapper: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
  },
  modalInput: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 48,
    textAlign: 'center',
    letterSpacing: 2,
  },
  modalButtonRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  modalCancelButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: colors.border.primary,
    height: 48,
    justifyContent: 'center',
  },
  modalCancelButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  modalDeleteButton: {
    flex: 1,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    backgroundColor: colors.status.error,
    height: 48,
    justifyContent: 'center',
  },
  modalDeleteButtonDisabled: {
    backgroundColor: colors.text.disabled,
    opacity: 0.5,
  },
  modalDeleteButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Avatar styles
  avatarContainer: {
    position: 'relative',
    marginBottom: spacing.md,
  },
  avatarImage: {
    ...shadows.glow,
  },
  avatarOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarEditBadge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.primary.main,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: colors.background.tertiary,
  },

  // Picture Modal styles
  pictureModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'flex-end',
  },
  pictureModalContent: {
    backgroundColor: colors.background.secondary,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    padding: spacing.lg,
    paddingBottom: spacing['2xl'],
  },
  pictureModalTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  pictureModalOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  pictureModalOptionIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  pictureModalOptionText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  pictureModalRemoveText: {
    color: colors.status.error,
  },
  pictureModalCancelButton: {
    marginTop: spacing.lg,
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  pictureModalCancelText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
});
