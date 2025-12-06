/**
 * Prisma Seed Script
 *
 * Seeds the database with initial supplement data.
 * Run with: npx prisma db seed
 */

import { PrismaClient, SupplementCategory } from '@prisma/client';

const prisma = new PrismaClient();

interface SupplementSeed {
  name: string;
  category: SupplementCategory;
  description: string;
  defaultDosage: string;
  defaultUnit: string;
  metadata?: Record<string, unknown>;
}

const supplements: SupplementSeed[] = [
  // AMINO_ACID
  {
    name: 'BCAA (Branched-Chain Amino Acids)',
    category: 'AMINO_ACID',
    description:
      'Essential amino acids (leucine, isoleucine, valine) that support muscle protein synthesis and recovery.',
    defaultDosage: '5',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Muscle recovery', 'Reduced fatigue', 'Protein synthesis'],
      timing: 'Before/during/after workout',
    },
  },
  {
    name: 'L-Lysine',
    category: 'AMINO_ACID',
    description:
      'Essential amino acid important for protein synthesis, calcium absorption, and immune function.',
    defaultDosage: '1000',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Immune support', 'Collagen production', 'Calcium absorption'],
      timing: 'With meals',
    },
  },
  {
    name: 'L-Glutamine',
    category: 'AMINO_ACID',
    description:
      'Most abundant amino acid in the body, supports gut health and immune function.',
    defaultDosage: '5',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Gut health', 'Immune support', 'Muscle recovery'],
      timing: 'Post-workout or before bed',
    },
  },
  {
    name: 'Beta-Alanine',
    category: 'AMINO_ACID',
    description:
      'Non-essential amino acid that helps buffer acid in muscles during high-intensity exercise.',
    defaultDosage: '3',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Endurance', 'Reduced fatigue', 'Performance'],
      timing: 'Pre-workout',
      warning: 'May cause harmless tingling sensation',
    },
  },
  {
    name: 'L-Citrulline',
    category: 'AMINO_ACID',
    description:
      'Amino acid that boosts nitric oxide production for improved blood flow and performance.',
    defaultDosage: '6',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Blood flow', 'Pump', 'Endurance'],
      timing: 'Pre-workout',
    },
  },
  {
    name: 'L-Arginine',
    category: 'AMINO_ACID',
    description:
      'Amino acid that serves as a precursor to nitric oxide, supporting cardiovascular health.',
    defaultDosage: '3',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Cardiovascular health', 'Blood flow', 'Immune function'],
      timing: 'Between meals',
    },
  },

  // VITAMIN
  {
    name: 'Vitamin D3',
    category: 'VITAMIN',
    description:
      'Essential vitamin for bone health, immune function, and mood regulation.',
    defaultDosage: '2000',
    defaultUnit: 'IU',
    metadata: {
      benefits: ['Bone health', 'Immune support', 'Mood'],
      timing: 'With fatty meal for absorption',
    },
  },
  {
    name: 'Vitamin C',
    category: 'VITAMIN',
    description:
      'Antioxidant vitamin essential for immune function and collagen synthesis.',
    defaultDosage: '1000',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Immune support', 'Antioxidant', 'Collagen production'],
      timing: 'With meals',
    },
  },
  {
    name: 'Vitamin B12',
    category: 'VITAMIN',
    description:
      'Essential for nerve function, red blood cell formation, and energy metabolism.',
    defaultDosage: '1000',
    defaultUnit: 'mcg',
    metadata: {
      benefits: ['Energy', 'Nerve health', 'Red blood cells'],
      timing: 'Morning',
    },
  },
  {
    name: 'Vitamin B Complex',
    category: 'VITAMIN',
    description:
      'Combination of all 8 B vitamins for energy metabolism and nervous system support.',
    defaultDosage: '1',
    defaultUnit: 'capsule',
    metadata: {
      benefits: ['Energy', 'Metabolism', 'Nervous system'],
      timing: 'Morning with food',
    },
  },
  {
    name: 'Vitamin E',
    category: 'VITAMIN',
    description: 'Fat-soluble antioxidant that protects cells from oxidative damage.',
    defaultDosage: '400',
    defaultUnit: 'IU',
    metadata: {
      benefits: ['Antioxidant', 'Skin health', 'Immune support'],
      timing: 'With fatty meal',
    },
  },
  {
    name: 'Vitamin K2',
    category: 'VITAMIN',
    description:
      'Essential for calcium metabolism and cardiovascular health, works synergistically with D3.',
    defaultDosage: '100',
    defaultUnit: 'mcg',
    metadata: {
      benefits: ['Bone health', 'Cardiovascular health', 'Calcium metabolism'],
      timing: 'With Vitamin D3',
    },
  },

  // MINERAL
  {
    name: 'Magnesium Glycinate',
    category: 'MINERAL',
    description:
      'Highly absorbable form of magnesium for muscle relaxation and sleep support.',
    defaultDosage: '400',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Muscle relaxation', 'Sleep', 'Stress relief'],
      timing: 'Evening/before bed',
    },
  },
  {
    name: 'Zinc',
    category: 'MINERAL',
    description:
      'Essential mineral for immune function, wound healing, and testosterone production.',
    defaultDosage: '30',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Immune support', 'Wound healing', 'Hormone production'],
      timing: 'With food to avoid nausea',
    },
  },
  {
    name: 'Calcium',
    category: 'MINERAL',
    description: 'Essential mineral for bone health, muscle function, and nerve transmission.',
    defaultDosage: '1000',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Bone health', 'Muscle function', 'Nerve transmission'],
      timing: 'Split doses with meals',
    },
  },
  {
    name: 'Iron',
    category: 'MINERAL',
    description:
      'Essential for oxygen transport and energy production. Important for women and athletes.',
    defaultDosage: '18',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Oxygen transport', 'Energy', 'Red blood cells'],
      timing: 'On empty stomach with Vitamin C',
      warning: 'Do not take with calcium or coffee',
    },
  },
  {
    name: 'Potassium',
    category: 'MINERAL',
    description: 'Electrolyte essential for fluid balance, muscle contractions, and heart health.',
    defaultDosage: '99',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Fluid balance', 'Muscle function', 'Heart health'],
      timing: 'With meals',
    },
  },
  {
    name: 'Selenium',
    category: 'MINERAL',
    description: 'Trace mineral with antioxidant properties, supports thyroid function.',
    defaultDosage: '200',
    defaultUnit: 'mcg',
    metadata: {
      benefits: ['Antioxidant', 'Thyroid support', 'Immune function'],
      timing: 'With food',
    },
  },

  // PERFORMANCE
  {
    name: 'Creatine Monohydrate',
    category: 'PERFORMANCE',
    description:
      'Most researched supplement for strength, power, and muscle gains. Safe and effective.',
    defaultDosage: '5',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Strength', 'Power', 'Muscle gains', 'Cognitive function'],
      timing: 'Any time, consistency matters most',
    },
  },
  {
    name: 'Pre-Workout',
    category: 'PERFORMANCE',
    description:
      'Energy and focus blend typically containing caffeine, beta-alanine, and nitric oxide boosters.',
    defaultDosage: '1',
    defaultUnit: 'scoop',
    metadata: {
      benefits: ['Energy', 'Focus', 'Performance'],
      timing: '20-30 minutes before workout',
      warning: 'Contains caffeine',
    },
  },
  {
    name: 'Caffeine',
    category: 'PERFORMANCE',
    description: 'Natural stimulant for energy, focus, and performance enhancement.',
    defaultDosage: '200',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Energy', 'Focus', 'Fat oxidation'],
      timing: 'Morning or pre-workout',
      warning: 'Avoid after 2pm for sleep quality',
    },
  },
  {
    name: 'Electrolytes',
    category: 'PERFORMANCE',
    description:
      'Essential minerals for hydration, muscle function, and performance during exercise.',
    defaultDosage: '1',
    defaultUnit: 'scoop',
    metadata: {
      benefits: ['Hydration', 'Performance', 'Recovery'],
      timing: 'During/after workout or in heat',
    },
  },
  {
    name: 'HMB (Beta-Hydroxy Beta-Methylbutyrate)',
    category: 'PERFORMANCE',
    description: 'Metabolite of leucine that may help preserve muscle during caloric deficit.',
    defaultDosage: '3',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Muscle preservation', 'Recovery'],
      timing: 'Split into 3 doses daily',
    },
  },

  // FATTY_ACID
  {
    name: 'Omega-3 Fish Oil',
    category: 'FATTY_ACID',
    description:
      'Essential fatty acids (EPA/DHA) for heart, brain, and joint health. Anti-inflammatory.',
    defaultDosage: '1000',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Heart health', 'Brain function', 'Joint health', 'Anti-inflammatory'],
      timing: 'With fatty meal',
    },
  },
  {
    name: 'Krill Oil',
    category: 'FATTY_ACID',
    description:
      'Omega-3 source with superior absorption due to phospholipid form. Contains astaxanthin.',
    defaultDosage: '500',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Heart health', 'Joint health', 'Antioxidant'],
      timing: 'With food',
    },
  },
  {
    name: 'Flaxseed Oil',
    category: 'FATTY_ACID',
    description: 'Plant-based omega-3 (ALA) source. Good for vegetarians/vegans.',
    defaultDosage: '1000',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Heart health', 'Plant-based omega-3'],
      timing: 'With food',
    },
  },
  {
    name: 'Algae Oil (Vegan Omega-3)',
    category: 'FATTY_ACID',
    description: 'Vegan source of EPA/DHA omega-3s derived from algae.',
    defaultDosage: '500',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Heart health', 'Brain function', 'Vegan-friendly'],
      timing: 'With food',
    },
  },

  // PROTEIN
  {
    name: 'Whey Protein',
    category: 'PROTEIN',
    description:
      'Fast-absorbing complete protein from milk. Ideal for post-workout muscle recovery.',
    defaultDosage: '30',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Muscle recovery', 'Protein synthesis', 'Convenient protein source'],
      timing: 'Post-workout or between meals',
    },
  },
  {
    name: 'Casein Protein',
    category: 'PROTEIN',
    description: 'Slow-digesting protein from milk. Ideal for overnight muscle recovery.',
    defaultDosage: '30',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Sustained amino acids', 'Overnight recovery'],
      timing: 'Before bed',
    },
  },
  {
    name: 'Plant Protein (Pea/Rice Blend)',
    category: 'PROTEIN',
    description: 'Complete plant-based protein blend suitable for vegans and those with dairy issues.',
    defaultDosage: '30',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Vegan-friendly', 'Complete protein', 'Easy digestion'],
      timing: 'Post-workout or between meals',
    },
  },
  {
    name: 'Collagen Peptides',
    category: 'PROTEIN',
    description: 'Hydrolyzed collagen for skin, hair, nail, and joint health.',
    defaultDosage: '10',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Skin health', 'Joint support', 'Hair and nails'],
      timing: 'Any time, often added to coffee',
    },
  },

  // HERBAL
  {
    name: 'Ashwagandha',
    category: 'HERBAL',
    description:
      'Adaptogenic herb that helps manage stress, supports sleep, and may boost testosterone.',
    defaultDosage: '600',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Stress relief', 'Sleep', 'Testosterone', 'Anxiety reduction'],
      timing: 'Evening or split doses',
    },
  },
  {
    name: 'Turmeric/Curcumin',
    category: 'HERBAL',
    description:
      'Powerful anti-inflammatory and antioxidant. Take with black pepper for absorption.',
    defaultDosage: '500',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Anti-inflammatory', 'Antioxidant', 'Joint health'],
      timing: 'With fatty meal and black pepper',
    },
  },
  {
    name: 'Rhodiola Rosea',
    category: 'HERBAL',
    description: 'Adaptogenic herb that helps with mental fatigue, stress, and physical performance.',
    defaultDosage: '500',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Mental clarity', 'Stress relief', 'Physical performance'],
      timing: 'Morning on empty stomach',
    },
  },
  {
    name: 'Ginkgo Biloba',
    category: 'HERBAL',
    description: 'Traditional herb for cognitive function and blood circulation.',
    defaultDosage: '120',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Cognitive function', 'Memory', 'Circulation'],
      timing: 'With food',
    },
  },
  {
    name: 'Milk Thistle',
    category: 'HERBAL',
    description: 'Herb traditionally used for liver health and detoxification support.',
    defaultDosage: '250',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Liver support', 'Detoxification', 'Antioxidant'],
      timing: 'With meals',
    },
  },

  // PROBIOTIC
  {
    name: 'Multi-Strain Probiotic',
    category: 'PROBIOTIC',
    description:
      'Blend of beneficial bacteria strains for gut health and immune function.',
    defaultDosage: '10',
    defaultUnit: 'billion CFU',
    metadata: {
      benefits: ['Gut health', 'Immune support', 'Digestion'],
      timing: 'Morning on empty stomach or with light meal',
    },
  },
  {
    name: 'Lactobacillus Acidophilus',
    category: 'PROBIOTIC',
    description: 'Common probiotic strain for digestive and immune health.',
    defaultDosage: '1',
    defaultUnit: 'capsule',
    metadata: {
      benefits: ['Digestion', 'Immune support', 'Gut balance'],
      timing: 'With food',
    },
  },

  // OTHER
  {
    name: 'Multivitamin',
    category: 'OTHER',
    description:
      'Comprehensive vitamin and mineral blend for overall health and filling nutritional gaps.',
    defaultDosage: '1',
    defaultUnit: 'tablet',
    metadata: {
      benefits: ['Nutritional insurance', 'Overall health'],
      timing: 'With breakfast',
    },
  },
  {
    name: 'CoQ10 (Coenzyme Q10)',
    category: 'OTHER',
    description:
      'Antioxidant that supports energy production and heart health. Levels decrease with age.',
    defaultDosage: '100',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Energy', 'Heart health', 'Antioxidant'],
      timing: 'With fatty meal',
    },
  },
  {
    name: 'Melatonin',
    category: 'OTHER',
    description: 'Sleep hormone supplement for improving sleep onset and quality.',
    defaultDosage: '3',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Sleep onset', 'Sleep quality', 'Jet lag'],
      timing: '30-60 minutes before bed',
      warning: 'Start with lower dose (0.5-1mg)',
    },
  },
  {
    name: 'Alpha-Lipoic Acid',
    category: 'OTHER',
    description: 'Powerful antioxidant that supports blood sugar metabolism and nerve health.',
    defaultDosage: '300',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Antioxidant', 'Blood sugar support', 'Nerve health'],
      timing: 'On empty stomach',
    },
  },
  {
    name: 'NAC (N-Acetyl Cysteine)',
    category: 'OTHER',
    description:
      'Precursor to glutathione, supports liver health and acts as a mucolytic.',
    defaultDosage: '600',
    defaultUnit: 'mg',
    metadata: {
      benefits: ['Liver support', 'Antioxidant', 'Respiratory health'],
      timing: 'On empty stomach',
    },
  },
  {
    name: 'Fiber (Psyllium Husk)',
    category: 'OTHER',
    description: 'Soluble fiber supplement for digestive health and regularity.',
    defaultDosage: '5',
    defaultUnit: 'g',
    metadata: {
      benefits: ['Digestive health', 'Regularity', 'Blood sugar support'],
      timing: 'With plenty of water, away from medications',
    },
  },
];

async function main() {
  console.log('🌱 Seeding supplements...\n');

  let created = 0;
  let skipped = 0;

  for (const supplement of supplements) {
    try {
      await prisma.supplement.upsert({
        where: { name: supplement.name },
        update: {
          category: supplement.category,
          description: supplement.description,
          defaultDosage: supplement.defaultDosage,
          defaultUnit: supplement.defaultUnit,
          metadata: supplement.metadata,
        },
        create: {
          name: supplement.name,
          category: supplement.category,
          description: supplement.description,
          defaultDosage: supplement.defaultDosage,
          defaultUnit: supplement.defaultUnit,
          metadata: supplement.metadata,
        },
      });
      created++;
      console.log(`✅ ${supplement.name}`);
    } catch (error) {
      skipped++;
      console.error(`❌ Failed to seed ${supplement.name}:`, error);
    }
  }

  console.log(`\n📊 Seeding complete: ${created} supplements created/updated, ${skipped} skipped`);

  // Print summary by category
  const categories = await prisma.supplement.groupBy({
    by: ['category'],
    _count: { id: true },
    orderBy: { category: 'asc' },
  });

  console.log('\n📈 Supplements by category:');
  for (const cat of categories) {
    console.log(`   ${cat.category}: ${cat._count.id}`);
  }
}

main()
  .catch((e) => {
    console.error('❌ Seeding failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
