import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';

const EFFECTIVE_DATE = 'December 12, 2025';
const LAST_UPDATED = 'December 12, 2025';

export default function PrivacyScreen() {
  const router = useRouter();
  const { getResponsiveValue } = useResponsive();

  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.xl,
    tablet: spacing['2xl'],
    default: spacing.lg,
  });

  const maxContentWidth = getResponsiveValue({
    small: undefined,
    medium: undefined,
    large: 700,
    tablet: 650,
    default: undefined,
  });

  return (
    <SafeAreaView style={styles.container} testID="privacy-screen">
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => {
            if (router.canGoBack()) {
              router.back();
            } else {
              router.replace('/(tabs)/profile');
            }
          }}
          style={styles.backButton}
          accessibilityLabel="Go back"
          testID="privacy-back-button"
        >
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Privacy Policy</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.content,
          { padding: contentPadding },
          maxContentWidth ? { maxWidth: maxContentWidth, alignSelf: 'center' as const, width: '100%' } : null,
        ]}
        showsVerticalScrollIndicator={false}
        testID="privacy-scroll-view"
      >
        {/* Last Updated */}
        <View style={styles.dateContainer}>
          <Text style={styles.dateText}>Effective Date: {EFFECTIVE_DATE}</Text>
          <Text style={styles.dateText}>Last Updated: {LAST_UPDATED}</Text>
        </View>

        {/* Introduction */}
        <Section title="1. Introduction">
          <Paragraph>
            Welcome to Nutri ("App," "we," "us," or "our"). We are committed to protecting your
            privacy and ensuring the security of your personal information. This Privacy Policy
            explains how we collect, use, disclose, and safeguard your information when you use our
            nutrition tracking and health analytics mobile application.
          </Paragraph>
          <Paragraph>
            By using Nutri, you consent to the data practices described in this Privacy Policy. If
            you do not agree with our policies and practices, please do not use our App.
          </Paragraph>
        </Section>

        {/* Information We Collect */}
        <Section title="2. Information We Collect">
          <SubSection title="2.1 Information You Provide">
            <Paragraph>We collect information that you voluntarily provide, including:</Paragraph>
            <BulletList
              items={[
                'Account information: Name, email address, password',
                'Profile data: Age, gender, height, weight, activity level',
                'Nutrition goals: Calorie targets, macronutrient goals',
                'Meal data: Food items, portions, meal times, nutritional content',
                'Health metrics: Heart rate, HRV, sleep data, weight measurements',
                'Photos: Food images for AI analysis (processed locally or via secure API)',
                'Notes and comments: Personal notes attached to meals or metrics',
              ]}
            />
          </SubSection>

          <SubSection title="2.2 Information Collected Automatically">
            <Paragraph>
              When you use Nutri, we automatically collect certain information:
            </Paragraph>
            <BulletList
              items={[
                'Device information: Device type, operating system, unique device identifiers',
                'App usage data: Features used, screens viewed, interaction patterns',
                'Log data: Access times, error logs, performance metrics',
                'Analytics data: Aggregated usage statistics (anonymized)',
              ]}
            />
          </SubSection>

          <SubSection title="2.3 Information from Third Parties">
            <Paragraph>
              With your permission, we may receive data from integrated health platforms:
            </Paragraph>
            <BulletList
              items={[
                'Apple Health (HealthKit): Health metrics, activity data, sleep data',
                'Google Fit: Activity and health measurements',
                'Wearable devices: Fitbit, Garmin, Oura, Whoop data',
                'Food databases: Open Food Facts, USDA nutritional data',
              ]}
            />
          </SubSection>
        </Section>

        {/* How We Use Your Information */}
        <Section title="3. How We Use Your Information">
          <SubSection title="3.1 Primary Uses">
            <Paragraph>We use your information to:</Paragraph>
            <BulletList
              items={[
                'Provide and maintain the App services',
                'Track your nutrition intake and health metrics',
                'Generate personalized insights and recommendations',
                'Display historical trends and correlations',
                'Sync data across your devices',
                'Send important service notifications',
              ]}
            />
          </SubSection>

          <SubSection title="3.2 Service Improvement">
            <Paragraph>We may also use your information to:</Paragraph>
            <BulletList
              items={[
                'Improve and optimize App performance',
                'Develop new features and functionality',
                'Train and improve our machine learning models (using anonymized data)',
                'Conduct research and analysis',
                'Fix bugs and technical issues',
              ]}
            />
          </SubSection>

          <SubSection title="3.3 Communications">
            <Paragraph>With your consent, we may send you:</Paragraph>
            <BulletList
              items={[
                'Service updates and announcements',
                'Tips and recommendations for better nutrition tracking',
                'Reminders for meal logging or supplement intake',
                'Promotional materials (you can opt out at any time)',
              ]}
            />
          </SubSection>
        </Section>

        {/* Data Storage and Security */}
        <Section title="4. Data Storage and Security">
          <SubSection title="4.1 Where We Store Your Data">
            <Paragraph>Your data is stored on secure servers located in:</Paragraph>
            <BulletList
              items={[
                'Cloud infrastructure with enterprise-grade security',
                'Encrypted databases with access controls',
                'Redundant backup systems for data protection',
              ]}
            />
          </SubSection>

          <SubSection title="4.2 Security Measures">
            <Paragraph>We implement comprehensive security measures including:</Paragraph>
            <BulletList
              items={[
                'End-to-end encryption for data in transit (TLS/SSL)',
                'Encryption at rest for stored data (AES-256)',
                'Secure authentication with hashed passwords',
                'Regular security audits and penetration testing',
                'Access controls and employee training',
                'Intrusion detection and monitoring systems',
              ]}
            />
          </SubSection>

          <SubSection title="4.3 Data Retention">
            <Paragraph>
              We retain your personal data for as long as your account is active or as needed to
              provide services. You may request deletion of your data at any time. Some data may be
              retained for legal compliance, dispute resolution, or fraud prevention.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Data Sharing */}
        <Section title="5. How We Share Your Information">
          <View style={styles.highlightBox}>
            <Ionicons
              name="shield-checkmark"
              size={24}
              color={colors.status.success}
              style={styles.highlightIcon}
            />
            <Text style={styles.highlightText}>
              We do NOT sell your personal information to third parties.
            </Text>
          </View>

          <SubSection title="5.1 Service Providers">
            <Paragraph>
              We may share data with trusted service providers who assist us in operating the App:
            </Paragraph>
            <BulletList
              items={[
                'Cloud hosting providers (encrypted data storage)',
                'Analytics services (anonymized usage data)',
                'Customer support platforms',
                'Payment processors (if applicable)',
              ]}
            />
          </SubSection>

          <SubSection title="5.2 Legal Requirements">
            <Paragraph>We may disclose your information if required by law to:</Paragraph>
            <BulletList
              items={[
                'Comply with legal obligations or court orders',
                'Protect our rights, privacy, safety, or property',
                'Enforce our Terms and Conditions',
                'Respond to government or regulatory requests',
              ]}
            />
          </SubSection>

          <SubSection title="5.3 Business Transfers">
            <Paragraph>
              In the event of a merger, acquisition, or sale of assets, your data may be
              transferred. We will notify you of any such change and your choices regarding your
              information.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Your Rights */}
        <Section title="6. Your Rights and Choices">
          <SubSection title="6.1 Access and Portability">
            <Paragraph>You have the right to:</Paragraph>
            <BulletList
              items={[
                'Access your personal data at any time through the App',
                'Export your data in a portable format',
                'Request a copy of all data we hold about you',
              ]}
            />
          </SubSection>

          <SubSection title="6.2 Correction and Deletion">
            <Paragraph>You can:</Paragraph>
            <BulletList
              items={[
                'Update or correct your personal information',
                'Delete individual meals, metrics, or other entries',
                'Request complete deletion of your account and data',
              ]}
            />
          </SubSection>

          <SubSection title="6.3 Opt-Out Rights">
            <Paragraph>You may opt out of:</Paragraph>
            <BulletList
              items={[
                'Marketing communications and promotional emails',
                'Push notifications (through device settings)',
                'Analytics and usage tracking',
                'Third-party integrations (disconnect at any time)',
              ]}
            />
          </SubSection>

          <SubSection title="6.4 Regional Rights">
            <Paragraph>
              Depending on your location, you may have additional rights under laws such as GDPR
              (Europe), CCPA (California), or other privacy regulations. Contact us to exercise
              these rights.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Health Data */}
        <Section title="7. Health Data Special Considerations">
          <View style={styles.warningBox}>
            <Ionicons
              name="medical"
              size={24}
              color={colors.primary.main}
              style={styles.warningIcon}
            />
            <Text style={styles.warningText}>SENSITIVE HEALTH INFORMATION</Text>
          </View>

          <SubSection title="7.1 Health Data Protection">
            <Paragraph>
              We treat health-related data with extra care and implement additional safeguards:
            </Paragraph>
            <BulletList
              items={[
                'Health data is stored separately with additional encryption',
                'Access to health data is strictly controlled',
                'Health data is never used for advertising purposes',
                'We comply with applicable health data regulations',
              ]}
            />
          </SubSection>

          <SubSection title="7.2 HealthKit/Google Fit Data">
            <Paragraph>
              Data accessed from Apple Health or Google Fit is only used to provide App
              functionality. This data is not shared with third parties for marketing or advertising
              purposes, in compliance with platform requirements.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Children's Privacy */}
        <Section title="8. Children's Privacy">
          <Paragraph>
            Nutri is not intended for children under 13 years of age (or 16 in some jurisdictions).
            We do not knowingly collect personal information from children. If you believe we have
            inadvertently collected data from a child, please contact us immediately and we will
            delete it.
          </Paragraph>
        </Section>

        {/* International Transfers */}
        <Section title="9. International Data Transfers">
          <Paragraph>
            Your information may be transferred to and processed in countries other than your own.
            These countries may have different data protection laws. We ensure appropriate
            safeguards are in place, including:
          </Paragraph>
          <BulletList
            items={[
              'Standard contractual clauses approved by regulators',
              'Compliance with international data transfer frameworks',
              'Ensuring recipients maintain adequate security measures',
            ]}
          />
        </Section>

        {/* Changes to Privacy Policy */}
        <Section title="10. Changes to This Privacy Policy">
          <Paragraph>
            We may update this Privacy Policy from time to time. We will notify you of significant
            changes through:
          </Paragraph>
          <BulletList
            items={[
              'In-app notifications',
              'Email to your registered address',
              'Prominent notice in the App',
            ]}
          />
          <Paragraph>
            Your continued use of the App after changes become effective constitutes acceptance of
            the updated Privacy Policy.
          </Paragraph>
        </Section>

        {/* Contact */}
        <Section title="11. Contact Us">
          <Paragraph>
            If you have questions, concerns, or requests regarding this Privacy Policy or your
            personal data, please contact us:
          </Paragraph>
          <View style={styles.contactBox}>
            <Text style={styles.contactLabel}>Privacy Inquiries:</Text>
            <Text style={styles.contactValue}>privacy@nutriapp.com</Text>
            <Text style={styles.contactLabel}>Data Protection Officer:</Text>
            <Text style={styles.contactValue}>dpo@nutriapp.com</Text>
            <Text style={styles.contactLabel}>General Support:</Text>
            <Text style={styles.contactValue}>support@nutriapp.com</Text>
          </View>
          <Paragraph>
            We will respond to your privacy-related inquiries within 30 days.
          </Paragraph>
        </Section>

        {/* Acknowledgment */}
        <View style={styles.acknowledgmentBox}>
          <Text style={styles.acknowledgmentText}>
            BY USING NUTRI, YOU ACKNOWLEDGE THAT YOU HAVE READ AND UNDERSTOOD THIS PRIVACY POLICY
            AND AGREE TO THE COLLECTION, USE, AND SHARING OF YOUR INFORMATION AS DESCRIBED HEREIN.
          </Text>
        </View>

        {/* Bottom spacing */}
        <View style={styles.bottomSpacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

// Component helpers
interface SectionProps {
  title: string;
  children: React.ReactNode;
}

function Section({ title, children }: SectionProps) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

interface SubSectionProps {
  title: string;
  children: React.ReactNode;
}

function SubSection({ title, children }: SubSectionProps) {
  return (
    <View style={styles.subSection}>
      <Text style={styles.subSectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

interface ParagraphProps {
  children: React.ReactNode;
}

function Paragraph({ children }: ParagraphProps) {
  return <Text style={styles.paragraph}>{children}</Text>;
}

interface BulletListProps {
  items: string[];
}

function BulletList({ items }: BulletListProps) {
  return (
    <View style={styles.bulletList}>
      {items.map((item, index) => (
        <View key={index} style={styles.bulletItem}>
          <Text style={styles.bullet}>{'\u2022'}</Text>
          <Text style={styles.bulletText}>{item}</Text>
        </View>
      ))}
    </View>
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
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    letterSpacing: -0.3,
  },
  headerSpacer: {
    width: 40,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
  },

  // Date container
  dateContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dateText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },

  // Sections
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.3,
  },
  subSection: {
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  subSectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },

  // Paragraphs
  paragraph: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
    marginBottom: spacing.md,
  },

  // Bullet lists
  bulletList: {
    marginBottom: spacing.md,
    paddingLeft: spacing.sm,
  },
  bulletItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing.xs,
  },
  bullet: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    marginRight: spacing.sm,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
  },
  bulletText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
  },

  // Highlight box (for positive messages)
  highlightBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${colors.status.success}15`,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: `${colors.status.success}30`,
  },
  highlightIcon: {
    marginRight: spacing.sm,
  },
  highlightText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.status.success,
    flex: 1,
  },

  // Warning box
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${colors.primary.main}15`,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: `${colors.primary.main}30`,
  },
  warningIcon: {
    marginRight: spacing.sm,
  },
  warningText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
    flex: 1,
  },

  // Contact box
  contactBox: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginVertical: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  contactLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  contactValue: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    marginBottom: spacing.sm,
  },

  // Acknowledgment box
  acknowledgmentBox: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    marginTop: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  acknowledgmentText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
  },

  bottomSpacer: {
    height: spacing.xl,
  },
});
