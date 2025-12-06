import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Modal,
  FlatList,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { supplementsApi, supplementLogsApi } from '@/lib/api/supplements';
import {
  Supplement,
  SUPPLEMENT_CATEGORIES,
  COMMON_UNITS,
} from '@/lib/types/supplements';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

export default function QuickLogSupplementScreen() {
  // Supplement selection
  const [supplements, setSupplements] = useState<Supplement[]>([]);
  const [selectedSupplement, setSelectedSupplement] = useState<Supplement | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSupplementPicker, setShowSupplementPicker] = useState(false);
  const [loadingSupplements, setLoadingSupplements] = useState(true);

  // Form fields
  const [dosage, setDosage] = useState('');
  const [unit, setUnit] = useState('mg');
  const [notes, setNotes] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    loadSupplements();
  }, []);

  const loadSupplements = async () => {
    try {
      const data = await supplementsApi.getSupplements();
      setSupplements(data);
    } catch (error) {
      console.error('Failed to load supplements:', error);
      showAlert('Error', getErrorMessage(error, 'Failed to load supplements'));
    } finally {
      setLoadingSupplements(false);
    }
  };

  const filteredSupplements = supplements.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectSupplement = (supplement: Supplement) => {
    setSelectedSupplement(supplement);
    if (supplement.defaultDosage) setDosage(supplement.defaultDosage);
    if (supplement.defaultUnit) setUnit(supplement.defaultUnit);
    setShowSupplementPicker(false);
  };

  const handleLog = async () => {
    if (!selectedSupplement) {
      showAlert('Error', 'Please select a supplement');
      return;
    }
    if (!dosage) {
      showAlert('Error', 'Please enter a dosage');
      return;
    }

    setIsLoading(true);
    try {
      await supplementLogsApi.quickLog(selectedSupplement.id, dosage, unit);
      showAlert('Success', `${selectedSupplement.name} logged!`, [
        { text: 'Log Another', onPress: () => resetForm() },
        { text: 'Done', onPress: () => router.back() },
      ]);
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to log supplement'));
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedSupplement(null);
    setDosage('');
    setUnit('mg');
    setNotes('');
  };

  const getCategoryLabel = (category: string) => {
    return SUPPLEMENT_CATEGORIES.find((c) => c.value === category)?.label || category;
  };

  const renderSupplementItem = ({ item }: { item: Supplement }) => (
    <TouchableOpacity
      style={styles.supplementItem}
      onPress={() => handleSelectSupplement(item)}
    >
      <View style={styles.supplementItemInfo}>
        <Text style={styles.supplementItemName}>{item.name}</Text>
        <Text style={styles.supplementItemCategory}>
          {getCategoryLabel(item.category)}
        </Text>
      </View>
      {item.defaultDosage && (
        <Text style={styles.supplementItemDosage}>
          {item.defaultDosage} {item.defaultUnit}
        </Text>
      )}
    </TouchableOpacity>
  );

  // Recent supplements (if we want to show them)
  const recentCategories = ['AMINO_ACID', 'VITAMIN', 'MINERAL', 'PERFORMANCE'];
  const quickSelectSupplements = supplements.filter((s) =>
    recentCategories.includes(s.category)
  ).slice(0, 8);

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="close" size={24} color={colors.text.secondary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Quick Log</Text>
          <TouchableOpacity onPress={handleLog} disabled={isLoading || !selectedSupplement}>
            {isLoading ? (
              <ActivityIndicator color={colors.primary.main} />
            ) : (
              <Text
                style={[
                  styles.logButton,
                  !selectedSupplement && styles.logButtonDisabled,
                ]}
              >
                Log
              </Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          <View style={styles.content}>
            {/* Quick Select */}
            {!selectedSupplement && quickSelectSupplements.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Quick Select</Text>
                <View style={styles.quickSelectGrid}>
                  {quickSelectSupplements.map((supp) => (
                    <TouchableOpacity
                      key={supp.id}
                      style={styles.quickSelectItem}
                      onPress={() => handleSelectSupplement(supp)}
                    >
                      <Text style={styles.quickSelectName} numberOfLines={1}>
                        {supp.name}
                      </Text>
                      {supp.defaultDosage && (
                        <Text style={styles.quickSelectDosage}>
                          {supp.defaultDosage} {supp.defaultUnit}
                        </Text>
                      )}
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            {/* Supplement Selection */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Supplement</Text>
              <TouchableOpacity
                style={styles.pickerButton}
                onPress={() => setShowSupplementPicker(true)}
              >
                {selectedSupplement ? (
                  <View style={styles.selectedSupplement}>
                    <View style={styles.selectedSupplementIcon}>
                      <Ionicons name="medical" size={24} color={colors.primary.main} />
                    </View>
                    <View style={styles.selectedSupplementInfo}>
                      <Text style={styles.selectedSupplementName}>
                        {selectedSupplement.name}
                      </Text>
                      <Text style={styles.selectedSupplementCategory}>
                        {getCategoryLabel(selectedSupplement.category)}
                      </Text>
                    </View>
                    <TouchableOpacity onPress={() => setSelectedSupplement(null)}>
                      <Ionicons name="close-circle" size={24} color={colors.text.tertiary} />
                    </TouchableOpacity>
                  </View>
                ) : (
                  <>
                    <Ionicons name="search" size={20} color={colors.text.tertiary} />
                    <Text style={styles.pickerPlaceholder}>Search supplements...</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>

            {/* Dosage */}
            {selectedSupplement && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Dosage</Text>
                <View style={styles.dosageRow}>
                  <View style={styles.dosageInputWrapper}>
                    <TextInput
                      style={styles.dosageInput}
                      placeholder="Amount"
                      placeholderTextColor={colors.text.disabled}
                      value={dosage}
                      onChangeText={setDosage}
                      keyboardType="numeric"
                      editable={!isLoading}
                    />
                  </View>
                  <View style={styles.unitSelector}>
                    {COMMON_UNITS.slice(0, 4).map((u) => (
                      <TouchableOpacity
                        key={u}
                        style={[styles.unitButton, unit === u && styles.unitButtonActive]}
                        onPress={() => setUnit(u)}
                      >
                        <Text
                          style={[
                            styles.unitButtonText,
                            unit === u && styles.unitButtonTextActive,
                          ]}
                        >
                          {u}
                        </Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
              </View>
            )}

            {/* Notes */}
            {selectedSupplement && (
              <View style={styles.section}>
                <Text style={styles.label}>Notes (optional)</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={[styles.input, styles.notesInput]}
                    placeholder="Add any notes..."
                    placeholderTextColor={colors.text.disabled}
                    value={notes}
                    onChangeText={setNotes}
                    multiline
                    numberOfLines={2}
                    textAlignVertical="top"
                    editable={!isLoading}
                  />
                </View>
              </View>
            )}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Supplement Picker Modal */}
      <Modal visible={showSupplementPicker} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Select Supplement</Text>
              <TouchableOpacity onPress={() => setShowSupplementPicker(false)}>
                <Ionicons name="close" size={24} color={colors.text.secondary} />
              </TouchableOpacity>
            </View>
            <View style={styles.searchContainer}>
              <Ionicons name="search" size={20} color={colors.text.tertiary} />
              <TextInput
                style={styles.searchInput}
                placeholder="Search supplements..."
                placeholderTextColor={colors.text.disabled}
                value={searchQuery}
                onChangeText={setSearchQuery}
                autoFocus
              />
              {searchQuery.length > 0 && (
                <TouchableOpacity onPress={() => setSearchQuery('')}>
                  <Ionicons name="close-circle" size={20} color={colors.text.tertiary} />
                </TouchableOpacity>
              )}
            </View>
            {loadingSupplements ? (
              <ActivityIndicator size="large" color={colors.primary.main} style={styles.loader} />
            ) : (
              <FlatList
                data={filteredSupplements}
                keyExtractor={(item) => item.id}
                renderItem={renderSupplementItem}
                contentContainerStyle={styles.supplementList}
                ItemSeparatorComponent={() => <View style={styles.separator} />}
                ListEmptyComponent={
                  <View style={styles.emptyList}>
                    <Text style={styles.emptyListText}>No supplements found</Text>
                  </View>
                }
              />
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  keyboardView: {
    flex: 1,
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
  logButton: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  logButtonDisabled: {
    color: colors.text.disabled,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing['3xl'],
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.3,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.3,
  },

  // Quick Select
  quickSelectGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  quickSelectItem: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    minWidth: '45%',
    flex: 1,
  },
  quickSelectName: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  quickSelectDosage: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },

  // Picker
  pickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    padding: spacing.md,
    minHeight: 56,
    gap: spacing.sm,
  },
  pickerPlaceholder: {
    fontSize: typography.fontSize.md,
    color: colors.text.disabled,
    flex: 1,
  },
  selectedSupplement: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  selectedSupplementIcon: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.full,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  selectedSupplementInfo: {
    flex: 1,
  },
  selectedSupplementName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  selectedSupplementCategory: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },

  // Dosage
  dosageRow: {
    flexDirection: 'row',
    gap: spacing.md,
    alignItems: 'flex-start',
  },
  dosageInputWrapper: {
    flex: 1,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dosageInput: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
  },
  unitSelector: {
    flexDirection: 'row',
    gap: spacing.xs,
    flexWrap: 'wrap',
    maxWidth: 120,
  },
  unitButton: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.tertiary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  unitButtonActive: {
    backgroundColor: colors.primary.main,
    borderColor: colors.primary.main,
  },
  unitButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  unitButtonTextActive: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
  },

  // Input
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
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
  },
  notesInput: {
    height: 60,
    textAlignVertical: 'top',
    paddingTop: spacing.md,
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background.primary,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    maxHeight: '80%',
    paddingBottom: spacing.xl,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  modalTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    margin: spacing.md,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  searchInput: {
    flex: 1,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  supplementList: {
    paddingHorizontal: spacing.md,
  },
  supplementItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
  },
  supplementItemInfo: {
    flex: 1,
  },
  supplementItemName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  supplementItemCategory: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  supplementItemDosage: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  separator: {
    height: 1,
    backgroundColor: colors.border.secondary,
  },
  emptyList: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  emptyListText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  loader: {
    padding: spacing.xl,
  },
});
