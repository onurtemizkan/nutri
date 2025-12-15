import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';

const EFFECTIVE_DATE = 'December 11, 2025';
const LAST_UPDATED = 'December 11, 2025';

export default function TermsScreen() {
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
    <SafeAreaView style={styles.container} testID="terms-screen">
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
          testID="terms-back-button"
        >
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Terms & Conditions</Text>
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
        testID="terms-scroll-view"
      >
        {/* Last Updated */}
        <View style={styles.dateContainer}>
          <Text style={styles.dateText}>Effective Date: {EFFECTIVE_DATE}</Text>
          <Text style={styles.dateText}>Last Updated: {LAST_UPDATED}</Text>
        </View>

        {/* Introduction */}
        <Section title="1. Introduction and Acceptance of Terms">
          <Paragraph>
            Welcome to Nutri ("App," "we," "us," or "our"). Nutri is a nutrition tracking and
            health analytics mobile application designed to help you monitor your dietary intake,
            track health metrics, and gain insights into how your nutrition affects your overall
            well-being.
          </Paragraph>
          <Paragraph>
            By downloading, installing, accessing, or using the Nutri application, you ("User,"
            "you," or "your") acknowledge that you have read, understood, and agree to be bound by
            these Terms and Conditions ("Terms"). If you do not agree to these Terms, you must
            immediately discontinue use of the App.
          </Paragraph>
          <Paragraph>
            These Terms constitute a legally binding agreement between you and Nutri. Your
            continued use of the App following any modifications to these Terms constitutes your
            acceptance of those changes.
          </Paragraph>
        </Section>

        {/* Description of Service */}
        <Section title="2. Description of Service">
          <SubSection title="2.1 Core Features">
            <Paragraph>Nutri provides the following services:</Paragraph>
            <BulletList
              items={[
                'Meal logging and nutritional tracking (calories, protein, carbohydrates, fats, fiber, sugar)',
                'Health metric monitoring (resting heart rate, heart rate variability, sleep duration, recovery scores)',
                'AI-powered food analysis and recognition',
                'Barcode scanning for packaged food products',
                'Integration with third-party health platforms (Apple Health, Google Fit)',
                'Machine learning-based insights correlating nutrition with health outcomes',
                'Personalized nutrition goals and recommendations',
                'Historical data analysis and trend visualization',
              ]}
            />
          </SubSection>

          <SubSection title="2.2 Service Availability">
            <Paragraph>
              We strive to maintain continuous service availability but do not guarantee
              uninterrupted access. The App may be temporarily unavailable due to:
            </Paragraph>
            <BulletList
              items={[
                'Scheduled maintenance and updates',
                'Server or infrastructure issues',
                'Force majeure events',
                'Third-party service disruptions',
              ]}
            />
          </SubSection>
        </Section>

        {/* User Accounts */}
        <Section title="3. User Accounts and Registration">
          <SubSection title="3.1 Account Creation">
            <Paragraph>
              To access certain features of Nutri, you must create an account. You may register
              using:
            </Paragraph>
            <BulletList
              items={[
                'Email address and password',
                'Apple Sign In (iOS devices)',
                'Other supported authentication providers',
              ]}
            />
          </SubSection>

          <SubSection title="3.2 Account Responsibilities">
            <Paragraph>You agree to:</Paragraph>
            <BulletList
              items={[
                'Provide accurate, current, and complete registration information',
                'Maintain the confidentiality of your account credentials',
                'Notify us immediately of any unauthorized access or security breach',
                'Accept responsibility for all activities occurring under your account',
                'Not share your account with third parties',
                'Not create multiple accounts for deceptive purposes',
              ]}
            />
          </SubSection>

          <SubSection title="3.3 Account Termination">
            <Paragraph>
              We reserve the right to suspend or terminate your account at our sole discretion if
              you violate these Terms, engage in fraudulent activity, or for any other reason we
              deem appropriate. You may delete your account at any time through the App settings.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Health Data Disclaimer */}
        <Section title="4. Health Information Disclaimer">
          <View style={styles.warningBox}>
            <Ionicons
              name="warning"
              size={24}
              color={colors.status.warning}
              style={styles.warningIcon}
            />
            <Text style={styles.warningText}>IMPORTANT HEALTH DISCLAIMER</Text>
          </View>

          <SubSection title="4.1 Not Medical Advice">
            <Paragraph>
              NUTRI IS NOT A MEDICAL DEVICE, AND THE INFORMATION PROVIDED THROUGH THE APP DOES NOT
              CONSTITUTE MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT. The App is intended for general
              informational and educational purposes only.
            </Paragraph>
            <Paragraph>
              The nutritional information, health metrics, correlations, and insights provided by
              Nutri should NOT be used as a substitute for professional medical advice, diagnosis,
              or treatment. Always seek the advice of your physician or other qualified health
              provider with any questions you may have regarding a medical condition or dietary
              changes.
            </Paragraph>
          </SubSection>

          <SubSection title="4.2 Accuracy of Information">
            <Paragraph>
              While we strive to provide accurate nutritional data, we cannot guarantee the
              accuracy, completeness, or reliability of:
            </Paragraph>
            <BulletList
              items={[
                'Nutritional information from food databases',
                'AI-generated food analysis and portion estimates',
                'Barcode-scanned product information',
                'Health metric calculations and correlations',
                'Machine learning predictions and recommendations',
              ]}
            />
            <Paragraph>
              Users should verify nutritional information independently, especially for medical
              conditions such as diabetes, food allergies, or eating disorders.
            </Paragraph>
          </SubSection>

          <SubSection title="4.3 Health Conditions">
            <Paragraph>
              If you have any health conditions, including but not limited to diabetes, heart
              disease, eating disorders, food allergies, or are pregnant or nursing, you should
              consult with a healthcare professional before using this App to guide dietary
              decisions.
            </Paragraph>
          </SubSection>

          <SubSection title="4.4 Emergency Situations">
            <Paragraph>
              NUTRI IS NOT DESIGNED FOR EMERGENCY SITUATIONS. If you are experiencing a medical
              emergency, call your local emergency services immediately. Do not rely on the App for
              any emergency medical needs.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Third-Party Integrations */}
        <Section title="5. Third-Party Integrations and Services">
          <SubSection title="5.1 Health Platform Integration">
            <Paragraph>
              Nutri may integrate with third-party health platforms including:
            </Paragraph>
            <BulletList
              items={[
                'Apple Health (HealthKit)',
                'Google Fit',
                'Fitbit',
                'Garmin Connect',
                'Oura',
                'Whoop',
                'Other compatible health data providers',
              ]}
            />
            <Paragraph>
              Your use of these integrations is subject to the respective third-party terms of
              service and privacy policies. We are not responsible for the accuracy of data
              received from third-party services.
            </Paragraph>
          </SubSection>

          <SubSection title="5.2 Food Database Providers">
            <Paragraph>
              Nutritional information is sourced from various databases including but not limited
              to Open Food Facts, USDA FoodData Central, and proprietary sources. We do not
              guarantee the accuracy of third-party nutritional data.
            </Paragraph>
          </SubSection>

          <SubSection title="5.3 Third-Party Terms">
            <Paragraph>
              By using third-party integrations, you agree to comply with their respective terms of
              service. We are not liable for any issues arising from third-party services.
            </Paragraph>
          </SubSection>
        </Section>

        {/* User Content */}
        <Section title="6. User Content and Data">
          <SubSection title="6.1 User-Generated Content">
            <Paragraph>
              You retain ownership of all content you submit to Nutri, including meal photos, food
              logs, notes, and personal health data ("User Content"). By submitting User Content,
              you grant Nutri a non-exclusive, worldwide, royalty-free license to use, process, and
              store such content solely for the purpose of providing and improving our services.
            </Paragraph>
          </SubSection>

          <SubSection title="6.2 Data Accuracy">
            <Paragraph>
              You are responsible for the accuracy of the data you enter. Inaccurate data entry may
              result in incorrect nutritional calculations and insights.
            </Paragraph>
          </SubSection>

          <SubSection title="6.3 Prohibited Content">
            <Paragraph>You agree not to submit content that:</Paragraph>
            <BulletList
              items={[
                'Violates any applicable laws or regulations',
                'Infringes on intellectual property rights',
                'Contains malicious code or harmful software',
                'Is fraudulent, deceptive, or misleading',
                'Promotes eating disorders or unhealthy behaviors',
                'Harasses, threatens, or harms others',
              ]}
            />
          </SubSection>
        </Section>

        {/* Privacy */}
        <Section title="7. Privacy and Data Collection">
          <SubSection title="7.1 Data We Collect">
            <Paragraph>Nutri collects and processes the following categories of data:</Paragraph>
            <BulletList
              items={[
                'Account information (name, email, authentication data)',
                'Nutritional data (meals, foods, portions, calories, macronutrients)',
                'Health metrics (heart rate, HRV, sleep, weight, body measurements)',
                'Device information (device type, operating system, app version)',
                'Usage data (feature usage, interaction patterns)',
                'Photos (when used for food analysis)',
                'Location data (only if explicitly permitted for local food suggestions)',
              ]}
            />
          </SubSection>

          <SubSection title="7.2 How We Use Your Data">
            <Paragraph>Your data is used to:</Paragraph>
            <BulletList
              items={[
                'Provide and improve the App services',
                'Generate personalized insights and recommendations',
                'Train and improve our machine learning models (in anonymized form)',
                'Communicate important service updates',
                'Ensure security and prevent fraud',
                'Comply with legal obligations',
              ]}
            />
          </SubSection>

          <SubSection title="7.3 Data Security">
            <Paragraph>
              We implement industry-standard security measures including encryption, secure data
              transmission (HTTPS/TLS), and secure storage. However, no method of transmission over
              the Internet or electronic storage is 100% secure.
            </Paragraph>
          </SubSection>

          <SubSection title="7.4 Data Retention">
            <Paragraph>
              We retain your data for as long as your account is active or as needed to provide
              services. You may request data deletion at any time, subject to legal retention
              requirements.
            </Paragraph>
          </SubSection>

          <SubSection title="7.5 Your Rights">
            <Paragraph>
              Depending on your jurisdiction, you may have rights to access, correct, delete, or
              port your personal data. Contact us to exercise these rights.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Intellectual Property */}
        <Section title="8. Intellectual Property">
          <SubSection title="8.1 Our Intellectual Property">
            <Paragraph>
              All content, features, functionality, design, graphics, logos, and trademarks in
              Nutri are owned by us or our licensors and are protected by copyright, trademark, and
              other intellectual property laws. You may not copy, modify, distribute, sell, or
              lease any part of our services without express written permission.
            </Paragraph>
          </SubSection>

          <SubSection title="8.2 License Grant">
            <Paragraph>
              We grant you a limited, non-exclusive, non-transferable, revocable license to use
              Nutri for personal, non-commercial purposes in accordance with these Terms.
            </Paragraph>
          </SubSection>

          <SubSection title="8.3 Restrictions">
            <Paragraph>You may not:</Paragraph>
            <BulletList
              items={[
                'Reverse engineer, decompile, or disassemble the App',
                'Remove any copyright or proprietary notices',
                'Use the App for commercial purposes without authorization',
                'Create derivative works based on the App',
                'Use automated systems to access the App (bots, scrapers)',
                'Interfere with or disrupt the App or servers',
              ]}
            />
          </SubSection>
        </Section>

        {/* Disclaimers */}
        <Section title="9. Disclaimers and Limitation of Liability">
          <SubSection title="9.1 Disclaimer of Warranties">
            <Paragraph>
              THE APP IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER
              EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF
              MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, OR ACCURACY.
            </Paragraph>
            <Paragraph>We do not warrant that:</Paragraph>
            <BulletList
              items={[
                'The App will meet your specific requirements',
                'The App will be uninterrupted, timely, secure, or error-free',
                'The results obtained from using the App will be accurate or reliable',
                'Any errors in the App will be corrected',
              ]}
            />
          </SubSection>

          <SubSection title="9.2 Limitation of Liability">
            <Paragraph>
              TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, NUTRI AND ITS OFFICERS,
              DIRECTORS, EMPLOYEES, AND AGENTS SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
              SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO:
            </Paragraph>
            <BulletList
              items={[
                'Loss of profits, data, or goodwill',
                'Personal injury or health issues',
                'Interruption of service',
                'Computer damage or system failure',
                'Cost of substitute services',
              ]}
            />
            <Paragraph>
              ARISING OUT OF OR RELATED TO YOUR USE OF OR INABILITY TO USE THE APP, EVEN IF WE HAVE
              BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
            </Paragraph>
          </SubSection>

          <SubSection title="9.3 Maximum Liability">
            <Paragraph>
              IN NO EVENT SHALL OUR TOTAL LIABILITY TO YOU EXCEED THE AMOUNT YOU PAID TO US, IF
              ANY, IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.
            </Paragraph>
          </SubSection>

          <SubSection title="9.4 Essential Purpose">
            <Paragraph>
              THE LIMITATIONS AND EXCLUSIONS IN THIS SECTION APPLY EVEN IF ANY LIMITED REMEDY FAILS
              OF ITS ESSENTIAL PURPOSE.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Indemnification */}
        <Section title="10. Indemnification">
          <Paragraph>
            You agree to indemnify, defend, and hold harmless Nutri and its officers, directors,
            employees, agents, licensors, and suppliers from and against any claims, actions,
            demands, liabilities, damages, losses, costs, and expenses (including reasonable
            attorneys' fees) arising out of or related to:
          </Paragraph>
          <BulletList
            items={[
              'Your use of the App',
              'Your violation of these Terms',
              'Your violation of any third-party rights',
              'Any content you submit to the App',
              'Your negligent or wrongful conduct',
            ]}
          />
        </Section>

        {/* Termination */}
        <Section title="11. Termination">
          <SubSection title="11.1 Termination by You">
            <Paragraph>
              You may terminate your account at any time by deleting your account through the App
              settings or by contacting us. Upon termination, your right to use the App will
              immediately cease.
            </Paragraph>
          </SubSection>

          <SubSection title="11.2 Termination by Us">
            <Paragraph>
              We may terminate or suspend your account immediately, without prior notice or
              liability, for any reason, including if you breach these Terms.
            </Paragraph>
          </SubSection>

          <SubSection title="11.3 Effect of Termination">
            <Paragraph>Upon termination:</Paragraph>
            <BulletList
              items={[
                'Your license to use the App is immediately revoked',
                'We may delete your account data (subject to legal retention requirements)',
                'Provisions that by their nature should survive termination will survive',
              ]}
            />
          </SubSection>
        </Section>

        {/* Governing Law */}
        <Section title="12. Governing Law and Dispute Resolution">
          <SubSection title="12.1 Governing Law">
            <Paragraph>
              These Terms shall be governed by and construed in accordance with the laws of the
              jurisdiction in which Nutri operates, without regard to its conflict of law
              provisions.
            </Paragraph>
          </SubSection>

          <SubSection title="12.2 Dispute Resolution">
            <Paragraph>
              Any dispute arising from these Terms or your use of the App shall first be attempted
              to be resolved through informal negotiation. If informal resolution fails, disputes
              shall be resolved through binding arbitration in accordance with the rules of the
              relevant arbitration authority.
            </Paragraph>
          </SubSection>

          <SubSection title="12.3 Class Action Waiver">
            <Paragraph>
              YOU AGREE THAT ANY DISPUTE RESOLUTION PROCEEDINGS WILL BE CONDUCTED ONLY ON AN
              INDIVIDUAL BASIS AND NOT IN A CLASS, CONSOLIDATED, OR REPRESENTATIVE ACTION.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Changes to Terms */}
        <Section title="13. Changes to Terms">
          <Paragraph>
            We reserve the right to modify these Terms at any time. We will notify you of
            significant changes through:
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
            the modified Terms. If you do not agree to the new Terms, you must stop using the App.
          </Paragraph>
        </Section>

        {/* Miscellaneous */}
        <Section title="14. Miscellaneous Provisions">
          <SubSection title="14.1 Entire Agreement">
            <Paragraph>
              These Terms, together with our Privacy Policy, constitute the entire agreement
              between you and Nutri regarding your use of the App.
            </Paragraph>
          </SubSection>

          <SubSection title="14.2 Severability">
            <Paragraph>
              If any provision of these Terms is found to be unenforceable or invalid, that
              provision will be limited or eliminated to the minimum extent necessary, and the
              remaining provisions will remain in full force and effect.
            </Paragraph>
          </SubSection>

          <SubSection title="14.3 Waiver">
            <Paragraph>
              Our failure to enforce any right or provision of these Terms will not be considered a
              waiver of such right or provision.
            </Paragraph>
          </SubSection>

          <SubSection title="14.4 Assignment">
            <Paragraph>
              You may not assign or transfer these Terms without our prior written consent. We may
              assign our rights and obligations under these Terms without restriction.
            </Paragraph>
          </SubSection>

          <SubSection title="14.5 Force Majeure">
            <Paragraph>
              We shall not be liable for any failure to perform our obligations where such failure
              results from circumstances beyond our reasonable control, including natural disasters,
              war, terrorism, riots, government actions, or internet/telecommunications failures.
            </Paragraph>
          </SubSection>
        </Section>

        {/* Contact */}
        <Section title="15. Contact Information">
          <Paragraph>
            If you have any questions, concerns, or requests regarding these Terms and Conditions,
            please contact us at:
          </Paragraph>
          <View style={styles.contactBox}>
            <Text style={styles.contactLabel}>Email:</Text>
            <Text style={styles.contactValue}>legal@nutriapp.com</Text>
            <Text style={styles.contactLabel}>Support:</Text>
            <Text style={styles.contactValue}>support@nutriapp.com</Text>
          </View>
          <Paragraph>
            We will respond to your inquiry within a reasonable timeframe.
          </Paragraph>
        </Section>

        {/* Acknowledgment */}
        <View style={styles.acknowledgmentBox}>
          <Text style={styles.acknowledgmentText}>
            BY USING NUTRI, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND AGREE TO BE BOUND
            BY THESE TERMS AND CONDITIONS.
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

  // Warning box
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${colors.status.warning}15`,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: `${colors.status.warning}30`,
  },
  warningIcon: {
    marginRight: spacing.sm,
  },
  warningText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.status.warning,
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
