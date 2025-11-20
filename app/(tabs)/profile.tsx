import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '@/lib/context/AuthContext';
import { useRouter } from 'expo-router';

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

      Alert.alert('Success', 'Goals updated successfully!');
      setIsEditing(false);
    } catch {
      Alert.alert('Error', 'Failed to update goals');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
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
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.content}>
          {/* Profile Header */}
          <View style={styles.profileHeader}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>
                {user?.name.charAt(0).toUpperCase()}
              </Text>
            </View>
            <Text style={styles.name}>{user?.name}</Text>
            <Text style={styles.email}>{user?.email}</Text>
          </View>

          {/* Daily Goals */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Daily Goals</Text>
              {!isEditing && (
                <TouchableOpacity onPress={() => setIsEditing(true)}>
                  <Text style={styles.editButton}>Edit</Text>
                </TouchableOpacity>
              )}
            </View>

            {isEditing ? (
              <>
                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Calorie Goal</Text>
                  <TextInput
                    style={styles.input}
                    value={goalCalories}
                    onChangeText={setGoalCalories}
                    keyboardType="numeric"
                    editable={!isLoading}
                  />
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Protein Goal (g)</Text>
                  <TextInput
                    style={styles.input}
                    value={goalProtein}
                    onChangeText={setGoalProtein}
                    keyboardType="numeric"
                    editable={!isLoading}
                  />
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Carbs Goal (g)</Text>
                  <TextInput
                    style={styles.input}
                    value={goalCarbs}
                    onChangeText={setGoalCarbs}
                    keyboardType="numeric"
                    editable={!isLoading}
                  />
                </View>

                <View style={styles.inputContainer}>
                  <Text style={styles.label}>Fat Goal (g)</Text>
                  <TextInput
                    style={styles.input}
                    value={goalFat}
                    onChangeText={setGoalFat}
                    keyboardType="numeric"
                    editable={!isLoading}
                  />
                </View>

                <View style={styles.buttonRow}>
                  <TouchableOpacity
                    style={[styles.button, styles.cancelButton]}
                    onPress={() => {
                      setIsEditing(false);
                      setGoalCalories(user?.goalCalories.toString() || '2000');
                      setGoalProtein(user?.goalProtein.toString() || '150');
                      setGoalCarbs(user?.goalCarbs.toString() || '200');
                      setGoalFat(user?.goalFat.toString() || '65');
                    }}
                    disabled={isLoading}
                  >
                    <Text style={styles.cancelButtonText}>Cancel</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.button, styles.saveButton]}
                    onPress={handleSaveGoals}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <ActivityIndicator color="#fff" />
                    ) : (
                      <Text style={styles.saveButtonText}>Save</Text>
                    )}
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

          {/* Account */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Account</Text>

            <TouchableOpacity style={styles.menuItem} onPress={handleLogout}>
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
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  profileHeader: {
    alignItems: 'center',
    paddingVertical: 32,
    backgroundColor: '#fff',
    borderRadius: 16,
    marginBottom: 24,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#3b5998',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: '700',
    color: '#fff',
  },
  name: {
    fontSize: 24,
    fontWeight: '700',
    color: '#000',
    marginBottom: 4,
  },
  email: {
    fontSize: 14,
    color: '#666',
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  editButton: {
    fontSize: 14,
    fontWeight: '600',
    color: '#3b5998',
  },
  goalItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  goalLabel: {
    fontSize: 16,
    color: '#666',
  },
  goalValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  button: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#f5f5f5',
  },
  cancelButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
  },
  saveButton: {
    backgroundColor: '#3b5998',
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  menuItem: {
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  menuItemText: {
    fontSize: 16,
    color: '#000',
  },
  logoutText: {
    color: '#F44336',
    fontWeight: '600',
  },
});
