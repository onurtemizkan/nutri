/**
 * E2E Test Data Seed Script
 *
 * Creates test users with comprehensive data for E2E testing.
 * Run with: npm run db:seed
 */

import { PrismaClient, HealthMetricType, ActivityType, ActivityIntensity, SubscriptionTier, SupplementFrequency, SupplementTimeOfDay, AdminRole } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

// ============================================================================
// ADMIN USER DEFINITIONS
// ============================================================================

interface TestAdminUser {
  email: string;
  password: string;
  name: string;
  role: AdminRole;
}

const TEST_ADMIN_USERS: TestAdminUser[] = [
  {
    email: 'admin@nutri.app',
    password: 'AdminPass123!',
    name: 'Super Admin',
    role: AdminRole.SUPER_ADMIN,
  },
  {
    email: 'support@nutri.app',
    password: 'SupportPass123!',
    name: 'Support Admin',
    role: AdminRole.SUPPORT,
  },
  {
    email: 'analyst@nutri.app',
    password: 'AnalystPass123!',
    name: 'Analytics Admin',
    role: AdminRole.ANALYST,
  },
];

async function seedAdminUsers() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('🔐 SEEDING ADMIN USERS');
  console.log('═══════════════════════════════════════════════════════════════\n');

  for (const admin of TEST_ADMIN_USERS) {
    const passwordHash = await bcrypt.hash(admin.password, 10);

    await prisma.adminUser.upsert({
      where: { email: admin.email },
      update: {
        passwordHash,
        name: admin.name,
        role: admin.role,
        isActive: true,
      },
      create: {
        email: admin.email,
        passwordHash,
        name: admin.name,
        role: admin.role,
        mfaEnabled: false,
        isActive: true,
      },
    });

    console.log(`  ✅ Created admin user: ${admin.email} (${admin.role})`);
  }

  console.log('\n📋 ADMIN USER CREDENTIALS:');
  console.log('---');
  for (const admin of TEST_ADMIN_USERS) {
    console.log(`📧 ${admin.email}`);
    console.log(`🔑 ${admin.password}`);
    console.log(`👤 ${admin.name} (${admin.role})`);
    console.log('---');
  }
  console.log('');
}

// ============================================================================
// TEST USER DEFINITIONS
// ============================================================================

interface TestUser {
  email: string;
  password: string;
  name: string;
  subscriptionTier: SubscriptionTier;
  goalCalories: number;
  goalProtein: number;
  goalCarbs: number;
  goalFat: number;
  currentWeight: number;
  goalWeight: number;
  height: number;
  activityLevel: string;
}

const TEST_USERS: TestUser[] = [
  {
    // Primary test user - Free tier with basic data
    email: 'test@nutri-e2e.local',
    password: 'TestPass123!',
    name: 'E2E Test User',
    subscriptionTier: SubscriptionTier.FREE,
    goalCalories: 2000,
    goalProtein: 150,
    goalCarbs: 200,
    goalFat: 65,
    currentWeight: 75,
    goalWeight: 70,
    height: 175,
    activityLevel: 'moderate',
  },
  {
    // Pro user with full data
    email: 'pro@nutri-e2e.local',
    password: 'ProPass123!',
    name: 'Pro Test User',
    subscriptionTier: SubscriptionTier.PRO,
    goalCalories: 2500,
    goalProtein: 180,
    goalCarbs: 250,
    goalFat: 80,
    currentWeight: 82,
    goalWeight: 78,
    height: 182,
    activityLevel: 'active',
  },
  {
    // Trial user
    email: 'trial@nutri-e2e.local',
    password: 'TrialPass123!',
    name: 'Trial Test User',
    subscriptionTier: SubscriptionTier.PRO_TRIAL,
    goalCalories: 1800,
    goalProtein: 120,
    goalCarbs: 180,
    goalFat: 55,
    currentWeight: 65,
    goalWeight: 60,
    height: 165,
    activityLevel: 'light',
  },
  {
    // Empty user - for testing empty states
    email: 'empty@nutri-e2e.local',
    password: 'EmptyPass123!',
    name: 'Empty Test User',
    subscriptionTier: SubscriptionTier.FREE,
    goalCalories: 2000,
    goalProtein: 150,
    goalCarbs: 200,
    goalFat: 65,
    currentWeight: 70,
    goalWeight: 68,
    height: 170,
    activityLevel: 'sedentary',
  },
  {
    // Athlete user - high activity
    email: 'athlete@nutri-e2e.local',
    password: 'AthletePass123!',
    name: 'Athlete Test User',
    subscriptionTier: SubscriptionTier.PRO,
    goalCalories: 3500,
    goalProtein: 250,
    goalCarbs: 400,
    goalFat: 100,
    currentWeight: 90,
    goalWeight: 88,
    height: 188,
    activityLevel: 'veryActive',
  },
];

// ============================================================================
// MEAL DATA GENERATORS
// ============================================================================

interface MealData {
  name: string;
  mealType: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  servingSize?: string;
  notes?: string;
}

const BREAKFAST_MEALS: MealData[] = [
  { name: 'Oatmeal with Berries', mealType: 'breakfast', calories: 350, protein: 12, carbs: 55, fat: 8, fiber: 6, sugar: 12, servingSize: '1 bowl', notes: 'Added honey and fresh blueberries' },
  { name: 'Greek Yogurt Parfait', mealType: 'breakfast', calories: 280, protein: 18, carbs: 35, fat: 6, fiber: 3, sugar: 20, servingSize: '200g', notes: 'With granola and strawberries' },
  { name: 'Scrambled Eggs with Toast', mealType: 'breakfast', calories: 420, protein: 22, carbs: 30, fat: 24, fiber: 2, sugar: 4, servingSize: '2 eggs + 2 slices', notes: 'Whole wheat toast' },
  { name: 'Avocado Toast', mealType: 'breakfast', calories: 380, protein: 10, carbs: 38, fat: 22, fiber: 8, sugar: 3, servingSize: '2 slices', notes: 'With cherry tomatoes and feta' },
  { name: 'Protein Smoothie', mealType: 'breakfast', calories: 320, protein: 30, carbs: 40, fat: 6, fiber: 4, sugar: 28, servingSize: '500ml', notes: 'Banana, protein powder, milk, spinach' },
  { name: 'Breakfast Burrito', mealType: 'breakfast', calories: 520, protein: 28, carbs: 45, fat: 26, fiber: 5, sugar: 6, servingSize: '1 burrito', notes: 'Eggs, beans, cheese, salsa' },
  { name: 'Overnight Oats', mealType: 'breakfast', calories: 340, protein: 14, carbs: 48, fat: 10, fiber: 7, sugar: 15, servingSize: '300g', notes: 'With chia seeds and maple syrup' },
];

const LUNCH_MEALS: MealData[] = [
  { name: 'Grilled Chicken Salad', mealType: 'lunch', calories: 450, protein: 42, carbs: 18, fat: 24, fiber: 6, sugar: 8, servingSize: '350g', notes: 'Mixed greens, cherry tomatoes, balsamic' },
  { name: 'Turkey Sandwich', mealType: 'lunch', calories: 480, protein: 32, carbs: 45, fat: 18, fiber: 4, sugar: 6, servingSize: '1 sandwich', notes: 'Whole grain bread, avocado, lettuce' },
  { name: 'Quinoa Buddha Bowl', mealType: 'lunch', calories: 520, protein: 22, carbs: 58, fat: 22, fiber: 10, sugar: 12, servingSize: '400g', notes: 'Roasted vegetables, chickpeas, tahini' },
  { name: 'Chicken Wrap', mealType: 'lunch', calories: 460, protein: 35, carbs: 40, fat: 18, fiber: 5, sugar: 5, servingSize: '1 wrap', notes: 'Whole wheat wrap, grilled chicken, hummus' },
  { name: 'Salmon Poke Bowl', mealType: 'lunch', calories: 550, protein: 38, carbs: 52, fat: 20, fiber: 6, sugar: 10, servingSize: '400g', notes: 'Brown rice, edamame, seaweed' },
  { name: 'Mediterranean Wrap', mealType: 'lunch', calories: 420, protein: 18, carbs: 48, fat: 18, fiber: 8, sugar: 6, servingSize: '1 wrap', notes: 'Falafel, hummus, vegetables' },
  { name: 'Tuna Salad', mealType: 'lunch', calories: 380, protein: 35, carbs: 12, fat: 22, fiber: 4, sugar: 4, servingSize: '300g', notes: 'Light mayo, celery, mixed greens' },
];

const DINNER_MEALS: MealData[] = [
  { name: 'Grilled Salmon with Rice', mealType: 'dinner', calories: 620, protein: 45, carbs: 48, fat: 28, fiber: 4, sugar: 2, servingSize: '400g', notes: 'Wild-caught salmon, brown rice, asparagus' },
  { name: 'Chicken Stir Fry', mealType: 'dinner', calories: 480, protein: 38, carbs: 35, fat: 22, fiber: 6, sugar: 10, servingSize: '350g', notes: 'Mixed vegetables, teriyaki sauce, jasmine rice' },
  { name: 'Beef Tacos', mealType: 'dinner', calories: 580, protein: 35, carbs: 42, fat: 30, fiber: 8, sugar: 6, servingSize: '3 tacos', notes: 'Ground beef, corn tortillas, fresh salsa' },
  { name: 'Pasta Primavera', mealType: 'dinner', calories: 520, protein: 18, carbs: 68, fat: 20, fiber: 8, sugar: 10, servingSize: '350g', notes: 'Whole wheat pasta, seasonal vegetables' },
  { name: 'Grilled Steak with Potatoes', mealType: 'dinner', calories: 720, protein: 52, carbs: 45, fat: 38, fiber: 5, sugar: 3, servingSize: '450g', notes: 'Sirloin steak, roasted sweet potatoes' },
  { name: 'Shrimp and Vegetable Curry', mealType: 'dinner', calories: 480, protein: 32, carbs: 42, fat: 22, fiber: 6, sugar: 8, servingSize: '400g', notes: 'Coconut milk curry, basmati rice' },
  { name: 'Turkey Meatballs with Zucchini Noodles', mealType: 'dinner', calories: 420, protein: 38, carbs: 22, fat: 22, fiber: 6, sugar: 12, servingSize: '350g', notes: 'Marinara sauce, parmesan' },
];

const SNACK_MEALS: MealData[] = [
  { name: 'Protein Bar', mealType: 'snack', calories: 220, protein: 20, carbs: 25, fat: 8, fiber: 3, sugar: 6, servingSize: '1 bar', notes: 'Quest Bar - Cookies and Cream' },
  { name: 'Apple with Almond Butter', mealType: 'snack', calories: 280, protein: 8, carbs: 30, fat: 16, fiber: 5, sugar: 22, servingSize: '1 apple + 2 tbsp', notes: 'Natural almond butter' },
  { name: 'Greek Yogurt', mealType: 'snack', calories: 150, protein: 15, carbs: 12, fat: 4, fiber: 0, sugar: 10, servingSize: '170g', notes: 'Plain, non-fat' },
  { name: 'Mixed Nuts', mealType: 'snack', calories: 200, protein: 6, carbs: 8, fat: 18, fiber: 2, sugar: 2, servingSize: '30g', notes: 'Almonds, cashews, walnuts' },
  { name: 'Cheese and Crackers', mealType: 'snack', calories: 240, protein: 10, carbs: 18, fat: 14, fiber: 1, sugar: 2, servingSize: '50g cheese + 6 crackers', notes: 'Cheddar with whole wheat crackers' },
  { name: 'Banana', mealType: 'snack', calories: 105, protein: 1, carbs: 27, fat: 0, fiber: 3, sugar: 14, servingSize: '1 medium', notes: 'Fresh' },
  { name: 'Cottage Cheese with Fruit', mealType: 'snack', calories: 180, protein: 18, carbs: 15, fat: 4, fiber: 2, sugar: 12, servingSize: '200g', notes: 'Low-fat with peaches' },
];

// ============================================================================
// HEALTH METRIC GENERATORS
// ============================================================================

function generateHealthMetrics(userId: string, daysBack: number = 30): Array<{
  userId: string;
  metricType: HealthMetricType;
  value: number;
  unit: string;
  source: string;
  recordedAt: Date;
}> {
  const metrics: Array<{
    userId: string;
    metricType: HealthMetricType;
    value: number;
    unit: string;
    source: string;
    recordedAt: Date;
  }> = [];

  const now = new Date();

  for (let i = 0; i < daysBack; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    date.setHours(8, 0, 0, 0); // Morning measurement time

    // Resting Heart Rate (55-75 bpm with some variation)
    metrics.push({
      userId,
      metricType: HealthMetricType.RESTING_HEART_RATE,
      value: 58 + Math.random() * 12,
      unit: 'bpm',
      source: 'apple_health',
      recordedAt: date,
    });

    // HRV RMSSD (20-60 ms)
    metrics.push({
      userId,
      metricType: HealthMetricType.HEART_RATE_VARIABILITY_RMSSD,
      value: 30 + Math.random() * 25,
      unit: 'ms',
      source: 'apple_health',
      recordedAt: date,
    });

    // Sleep Duration (6-9 hours)
    metrics.push({
      userId,
      metricType: HealthMetricType.SLEEP_DURATION,
      value: 6 + Math.random() * 3,
      unit: 'hours',
      source: 'apple_health',
      recordedAt: date,
    });

    // Deep Sleep (1-2.5 hours)
    metrics.push({
      userId,
      metricType: HealthMetricType.DEEP_SLEEP_DURATION,
      value: 1 + Math.random() * 1.5,
      unit: 'hours',
      source: 'apple_health',
      recordedAt: date,
    });

    // Sleep Efficiency (75-98%)
    metrics.push({
      userId,
      metricType: HealthMetricType.SLEEP_EFFICIENCY,
      value: 75 + Math.random() * 23,
      unit: '%',
      source: 'apple_health',
      recordedAt: date,
    });

    // Steps (3000-15000)
    const stepsDate = new Date(date);
    stepsDate.setHours(22, 0, 0, 0); // End of day
    metrics.push({
      userId,
      metricType: HealthMetricType.STEPS,
      value: Math.floor(3000 + Math.random() * 12000),
      unit: 'steps',
      source: 'apple_health',
      recordedAt: stepsDate,
    });

    // Active Calories (200-800)
    metrics.push({
      userId,
      metricType: HealthMetricType.ACTIVE_CALORIES,
      value: Math.floor(200 + Math.random() * 600),
      unit: 'kcal',
      source: 'apple_health',
      recordedAt: stepsDate,
    });

    // Recovery Score (40-100)
    metrics.push({
      userId,
      metricType: HealthMetricType.RECOVERY_SCORE,
      value: Math.floor(40 + Math.random() * 60),
      unit: '%',
      source: 'apple_health',
      recordedAt: date,
    });
  }

  return metrics;
}

// ============================================================================
// ACTIVITY GENERATORS
// ============================================================================

function generateActivities(userId: string, daysBack: number = 30): Array<{
  userId: string;
  activityType: ActivityType;
  intensity: ActivityIntensity;
  startedAt: Date;
  endedAt: Date;
  duration: number;
  caloriesBurned: number;
  averageHeartRate: number;
  maxHeartRate: number;
  distance?: number;
  steps?: number;
  source: string;
  notes?: string;
}> {
  const activities: Array<{
    userId: string;
    activityType: ActivityType;
    intensity: ActivityIntensity;
    startedAt: Date;
    endedAt: Date;
    duration: number;
    caloriesBurned: number;
    averageHeartRate: number;
    maxHeartRate: number;
    distance?: number;
    steps?: number;
    source: string;
    notes?: string;
  }> = [];

  const activityTypes: Array<{ type: ActivityType; intensity: ActivityIntensity; durationRange: [number, number]; caloriesPerMinute: number; hasDistance: boolean }> = [
    { type: ActivityType.RUNNING, intensity: ActivityIntensity.HIGH, durationRange: [20, 60], caloriesPerMinute: 12, hasDistance: true },
    { type: ActivityType.WEIGHT_TRAINING, intensity: ActivityIntensity.MODERATE, durationRange: [30, 75], caloriesPerMinute: 6, hasDistance: false },
    { type: ActivityType.CYCLING, intensity: ActivityIntensity.MODERATE, durationRange: [30, 90], caloriesPerMinute: 10, hasDistance: true },
    { type: ActivityType.YOGA, intensity: ActivityIntensity.LOW, durationRange: [30, 60], caloriesPerMinute: 3, hasDistance: false },
    { type: ActivityType.SWIMMING, intensity: ActivityIntensity.HIGH, durationRange: [30, 60], caloriesPerMinute: 11, hasDistance: true },
    { type: ActivityType.WALKING, intensity: ActivityIntensity.LOW, durationRange: [20, 60], caloriesPerMinute: 4, hasDistance: true },
    { type: ActivityType.CROSSFIT, intensity: ActivityIntensity.MAXIMUM, durationRange: [15, 30], caloriesPerMinute: 15, hasDistance: false },
  ];

  const now = new Date();

  // Generate 3-5 activities per week
  for (let i = 0; i < daysBack; i++) {
    // Skip some days (simulate realistic workout schedule)
    if (Math.random() > 0.5) continue;

    const activityConfig = activityTypes[Math.floor(Math.random() * activityTypes.length)];
    const duration = activityConfig.durationRange[0] + Math.floor(Math.random() * (activityConfig.durationRange[1] - activityConfig.durationRange[0]));

    const startedAt = new Date(now);
    startedAt.setDate(startedAt.getDate() - i);
    startedAt.setHours(7 + Math.floor(Math.random() * 12), Math.floor(Math.random() * 60), 0, 0);

    const endedAt = new Date(startedAt);
    endedAt.setMinutes(endedAt.getMinutes() + duration);

    const activity: {
      userId: string;
      activityType: ActivityType;
      intensity: ActivityIntensity;
      startedAt: Date;
      endedAt: Date;
      duration: number;
      caloriesBurned: number;
      averageHeartRate: number;
      maxHeartRate: number;
      distance?: number;
      steps?: number;
      source: string;
      notes?: string;
    } = {
      userId,
      activityType: activityConfig.type,
      intensity: activityConfig.intensity,
      startedAt,
      endedAt,
      duration,
      caloriesBurned: Math.floor(duration * activityConfig.caloriesPerMinute * (0.8 + Math.random() * 0.4)),
      averageHeartRate: Math.floor(110 + Math.random() * 40),
      maxHeartRate: Math.floor(150 + Math.random() * 30),
      source: 'apple_health',
    };

    if (activityConfig.hasDistance) {
      // Running/cycling/walking distance in meters
      if (activityConfig.type === ActivityType.RUNNING) {
        activity.distance = Math.floor(duration * 150 * (0.8 + Math.random() * 0.4)); // ~9km/hr avg
        activity.steps = Math.floor(activity.distance * 1.3);
      } else if (activityConfig.type === ActivityType.CYCLING) {
        activity.distance = Math.floor(duration * 400 * (0.8 + Math.random() * 0.4)); // ~24km/hr avg
      } else if (activityConfig.type === ActivityType.SWIMMING) {
        activity.distance = Math.floor(duration * 40 * (0.8 + Math.random() * 0.4)); // ~2.4km/hr avg
      } else if (activityConfig.type === ActivityType.WALKING) {
        activity.distance = Math.floor(duration * 80 * (0.8 + Math.random() * 0.4)); // ~5km/hr avg
        activity.steps = Math.floor(activity.distance * 1.3);
      }
    }

    activities.push(activity);
  }

  return activities;
}

// ============================================================================
// SUPPLEMENT DATA
// ============================================================================

interface SupplementData {
  name: string;
  brand?: string;
  dosageAmount: number;
  dosageUnit: string;
  frequency: SupplementFrequency;
  timesPerDay: number;
  timeOfDay: SupplementTimeOfDay[];
  withFood: boolean;
  notes?: string;
  color?: string;
}

const SUPPLEMENTS: SupplementData[] = [
  {
    name: 'Vitamin D3',
    brand: 'NOW Foods',
    dosageAmount: 5000,
    dosageUnit: 'IU',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.MORNING],
    withFood: true,
    notes: 'Take with fatty meal for better absorption',
    color: '#FFD700',
  },
  {
    name: 'Omega-3 Fish Oil',
    brand: 'Nordic Naturals',
    dosageAmount: 1000,
    dosageUnit: 'mg',
    frequency: SupplementFrequency.TWICE_DAILY,
    timesPerDay: 2,
    timeOfDay: [SupplementTimeOfDay.WITH_BREAKFAST, SupplementTimeOfDay.WITH_DINNER],
    withFood: true,
    notes: 'EPA/DHA for heart and brain health',
    color: '#FFA500',
  },
  {
    name: 'Magnesium Glycinate',
    brand: 'Doctor\'s Best',
    dosageAmount: 200,
    dosageUnit: 'mg',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.BEFORE_BED],
    withFood: false,
    notes: 'Helps with sleep and muscle recovery',
    color: '#9370DB',
  },
  {
    name: 'Vitamin B Complex',
    brand: 'Thorne',
    dosageAmount: 1,
    dosageUnit: 'capsule',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.MORNING],
    withFood: true,
    notes: 'Energy and nervous system support',
    color: '#32CD32',
  },
  {
    name: 'Probiotics',
    brand: 'Garden of Life',
    dosageAmount: 50,
    dosageUnit: 'billion CFU',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.EMPTY_STOMACH],
    withFood: false,
    notes: 'Take 30 min before breakfast',
    color: '#00CED1',
  },
  {
    name: 'Creatine Monohydrate',
    brand: 'Optimum Nutrition',
    dosageAmount: 5,
    dosageUnit: 'g',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.AFTERNOON],
    withFood: false,
    notes: 'Mix with post-workout shake',
    color: '#FF6347',
  },
  {
    name: 'Zinc',
    brand: 'Life Extension',
    dosageAmount: 30,
    dosageUnit: 'mg',
    frequency: SupplementFrequency.DAILY,
    timesPerDay: 1,
    timeOfDay: [SupplementTimeOfDay.WITH_DINNER],
    withFood: true,
    notes: 'Immune support',
    color: '#708090',
  },
];

// ============================================================================
// MAIN SEED FUNCTION
// ============================================================================

async function seed() {
  console.log('🌱 Starting E2E test data seed...\n');

  // Clean existing test data
  console.log('🧹 Cleaning existing test data...');
  await prisma.supplementLog.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.supplement.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.mLInsight.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.mLPrediction.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.mLFeature.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.userMLProfile.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.activity.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.healthMetric.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.weightRecord.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.waterIntake.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.meal.deleteMany({
    where: { user: { email: { endsWith: '@nutri-e2e.local' } } },
  });
  await prisma.user.deleteMany({
    where: { email: { endsWith: '@nutri-e2e.local' } },
  });
  console.log('✅ Cleaned existing test data\n');

  // Seed admin users first
  await seedAdminUsers();

  // Create test users
  console.log('👥 Creating test users...');
  const createdUsers: { user: typeof TEST_USERS[0]; id: string }[] = [];

  for (const userData of TEST_USERS) {
    const hashedPassword = await bcrypt.hash(userData.password, 10);

    const user = await prisma.user.create({
      data: {
        email: userData.email,
        password: hashedPassword,
        name: userData.name,
        subscriptionTier: userData.subscriptionTier,
        goalCalories: userData.goalCalories,
        goalProtein: userData.goalProtein,
        goalCarbs: userData.goalCarbs,
        goalFat: userData.goalFat,
        currentWeight: userData.currentWeight,
        goalWeight: userData.goalWeight,
        height: userData.height,
        activityLevel: userData.activityLevel,
        subscriptionStartDate: userData.subscriptionTier !== SubscriptionTier.FREE ? new Date() : null,
        subscriptionEndDate: userData.subscriptionTier === SubscriptionTier.PRO_TRIAL
          ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days trial
          : userData.subscriptionTier === SubscriptionTier.PRO
            ? new Date(Date.now() + 365 * 24 * 60 * 60 * 1000) // 1 year
            : null,
      },
    });

    createdUsers.push({ user: userData, id: user.id });
    console.log(`  ✅ Created user: ${userData.name} (${userData.email})`);
  }
  console.log('');

  // Populate data for each user (except "empty" user)
  for (const { user, id: userId } of createdUsers) {
    if (user.email === 'empty@nutri-e2e.local') {
      console.log(`⏭️  Skipping data population for ${user.name} (empty test user)\n`);
      continue;
    }

    console.log(`📊 Populating data for ${user.name}...`);

    // Create meals for the past 14 days
    const now = new Date();
    let mealCount = 0;

    for (let i = 0; i < 14; i++) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      // Breakfast
      const breakfast = BREAKFAST_MEALS[Math.floor(Math.random() * BREAKFAST_MEALS.length)];
      date.setHours(8, 0, 0, 0);
      await prisma.meal.create({
        data: {
          userId,
          ...breakfast,
          consumedAt: new Date(date),
        },
      });
      mealCount++;

      // Lunch
      const lunch = LUNCH_MEALS[Math.floor(Math.random() * LUNCH_MEALS.length)];
      date.setHours(12, 30, 0, 0);
      await prisma.meal.create({
        data: {
          userId,
          ...lunch,
          consumedAt: new Date(date),
        },
      });
      mealCount++;

      // Dinner
      const dinner = DINNER_MEALS[Math.floor(Math.random() * DINNER_MEALS.length)];
      date.setHours(19, 0, 0, 0);
      await prisma.meal.create({
        data: {
          userId,
          ...dinner,
          consumedAt: new Date(date),
        },
      });
      mealCount++;

      // Snacks (1-2 per day)
      const numSnacks = 1 + Math.floor(Math.random() * 2);
      for (let s = 0; s < numSnacks; s++) {
        const snack = SNACK_MEALS[Math.floor(Math.random() * SNACK_MEALS.length)];
        date.setHours(10 + s * 5, 0, 0, 0);
        await prisma.meal.create({
          data: {
            userId,
            ...snack,
            consumedAt: new Date(date),
          },
        });
        mealCount++;
      }
    }
    console.log(`  ✅ Created ${mealCount} meals`);

    // Create health metrics
    const healthMetrics = generateHealthMetrics(userId, 30);
    await prisma.healthMetric.createMany({
      data: healthMetrics,
    });
    console.log(`  ✅ Created ${healthMetrics.length} health metrics`);

    // Create activities
    const activities = generateActivities(userId, 30);
    await prisma.activity.createMany({
      data: activities,
    });
    console.log(`  ✅ Created ${activities.length} activities`);

    // Create weight records (daily for past 30 days)
    const weightRecords = [];
    let weight = user.currentWeight;
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      date.setHours(7, 0, 0, 0);

      // Small daily fluctuation
      weight += (Math.random() - 0.5) * 0.3;
      weightRecords.push({
        userId,
        weight: Math.round(weight * 10) / 10,
        recordedAt: date,
      });
    }
    await prisma.weightRecord.createMany({
      data: weightRecords,
    });
    console.log(`  ✅ Created ${weightRecords.length} weight records`);

    // Create water intake records (3-8 per day for past 14 days)
    const waterIntakes = [];
    for (let i = 0; i < 14; i++) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);

      const numIntakes = 3 + Math.floor(Math.random() * 6);
      for (let w = 0; w < numIntakes; w++) {
        date.setHours(8 + w * 2, Math.floor(Math.random() * 60), 0, 0);
        waterIntakes.push({
          userId,
          amount: 200 + Math.floor(Math.random() * 300), // 200-500ml
          recordedAt: new Date(date),
        });
      }
    }
    await prisma.waterIntake.createMany({
      data: waterIntakes,
    });
    console.log(`  ✅ Created ${waterIntakes.length} water intake records`);

    // Create supplements for Pro/Trial users
    if (user.subscriptionTier !== SubscriptionTier.FREE) {
      const numSupplements = 3 + Math.floor(Math.random() * 3); // 3-5 supplements
      const selectedSupplements = SUPPLEMENTS.slice(0, numSupplements);

      for (const suppData of selectedSupplements) {
        const supplement = await prisma.supplement.create({
          data: {
            userId,
            ...suppData,
          },
        });

        // Create supplement logs for past 14 days
        for (let i = 0; i < 14; i++) {
          const date = new Date(now);
          date.setDate(date.getDate() - i);

          // Based on frequency, create appropriate number of logs
          const timesToday = suppData.frequency === SupplementFrequency.TWICE_DAILY ? 2 : 1;

          for (let t = 0; t < timesToday; t++) {
            // Skip some days randomly (85% compliance)
            if (Math.random() > 0.85) continue;

            const logTime = new Date(date);
            if (suppData.timeOfDay.includes(SupplementTimeOfDay.MORNING) || suppData.timeOfDay.includes(SupplementTimeOfDay.WITH_BREAKFAST)) {
              logTime.setHours(8 + t * 10, Math.floor(Math.random() * 30), 0, 0);
            } else if (suppData.timeOfDay.includes(SupplementTimeOfDay.BEFORE_BED)) {
              logTime.setHours(22, Math.floor(Math.random() * 30), 0, 0);
            } else {
              logTime.setHours(12 + t * 6, Math.floor(Math.random() * 30), 0, 0);
            }

            await prisma.supplementLog.create({
              data: {
                userId,
                supplementId: supplement.id,
                takenAt: logTime,
              },
            });
          }
        }
      }
      console.log(`  ✅ Created ${selectedSupplements.length} supplements with logs`);
    }

    // Create ML profile
    await prisma.userMLProfile.create({
      data: {
        userId,
        totalDataPoints: healthMetrics.length + activities.length + mealCount,
        dataQualityScore: 0.75 + Math.random() * 0.2,
        hasMinimumNutritionData: true,
        hasMinimumHealthData: true,
        hasMinimumActivityData: activities.length >= 10,
        modelsAvailable: {
          rhr_prediction: true,
          hrv_prediction: user.subscriptionTier !== SubscriptionTier.FREE,
          sleep_prediction: user.subscriptionTier !== SubscriptionTier.FREE,
        },
        enablePredictions: true,
        enableInsights: true,
      },
    });
    console.log(`  ✅ Created ML profile`);

    console.log('');
  }

  // Print summary
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('📋 TEST USER CREDENTIALS SUMMARY');
  console.log('═══════════════════════════════════════════════════════════════\n');

  for (const { user } of createdUsers) {
    console.log(`📧 ${user.email}`);
    console.log(`🔑 ${user.password}`);
    console.log(`👤 ${user.name}`);
    console.log(`💎 ${user.subscriptionTier}`);
    console.log('---');
  }

  console.log('\n✅ E2E test data seed completed successfully!\n');
}

// ============================================================================
// RUN SEED
// ============================================================================

seed()
  .catch((error) => {
    console.error('❌ Seed failed:', error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
