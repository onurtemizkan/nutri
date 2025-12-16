# Food Allergy & Intolerance Detection via HRV Monitoring

## Executive Summary

This document presents comprehensive research on detecting food allergies, intolerances, and sensitivities through Heart Rate Variability (HRV) monitoring, combined with ingredient extraction strategies for the Nutri app.

**Key Discovery**: HRV changes can detect allergic reactions **17 minutes before clinical symptoms appear** with 90.5% sensitivity and 79.4% specificity.

---

## Part 1: Immunological Effects on HRV

### 1.1 The Allergy-Autonomic Connection

Food allergies create measurable autonomic nervous system (ANS) dysfunction:

| Phase | HRV Response | Mechanism |
|-------|--------------|-----------|
| **Initial (0-20 min)** | Parasympathetic predominance | Vagal activation from allergen exposure |
| **Reaction Phase** | Sympathetic withdrawal | LF/HF ratio decreases, LF power drops |
| **Anaphylaxis** | Severe HRV depression | RMS of HR change: 8.2% vs 5.6% (tolerant) |

**Research Evidence**:
- Allergic rhinitis patients show **significantly lower LF/HF ratio** than controls
- 24-hour cardiac autonomic activity shows sustained imbalance
- HRV indices predicting parasympathetic predominance are **increased** in allergic patients

### 1.2 Mast Cell-Vagus Nerve Interaction

A critical bidirectional relationship exists:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAST CELL ←→ VAGUS NERVE                     │
├─────────────────────────────────────────────────────────────────┤
│ Vagal Stimulation → Acetylcholine → Histamine Release (↑)       │
│ Higher Vagal Tone → α7 Nicotinic Activation → Mast Cell (↓)     │
│                                                                 │
│ Result: Complex feedback loop affecting HRV during reactions    │
└─────────────────────────────────────────────────────────────────┘
```

**Clinical Implication**: 44% of POTS patients also meet MCAS criteria, suggesting mast cell activation and dysautonomia are linked.

### 1.3 Inflammatory Cytokine-HRV Relationship

| Cytokine | HRV Correlation | Time Course |
|----------|-----------------|-------------|
| TNF-α | Strong inverse | Hours after mast cell activation |
| IL-6 | Strong inverse | Hours to next day |
| IL-1β | Moderate inverse | Hours after exposure |
| Histamine | Acute inverse | Minutes to hours |

**Effect**: Higher vagal activity = lower inflammatory cytokine production = better HRV

---

## Part 2: Specific Allergen Effects

### 2.1 High-Confidence Allergen-HRV Effects

#### Gluten (Celiac Disease)
| Metric | Effect | Population |
|--------|--------|------------|
| SDNN | Significantly lower | All celiac patients |
| SDANN | Significantly lower | Children with celiac |
| Resting HRV | Lower than controls | P < 0.05 |
| LF/HF correlation | Positive with disease duration | Longer celiac = worse |

**Key Finding**: 20% show parasympathetic dominance, 36% show **sympathetic dominance**, 44% balanced.

#### Dairy (Allergy vs Intolerance)
| Type | Mechanism | HRV Effect |
|------|-----------|------------|
| **Lactose Intolerance** | Histamine release from immune response | ↑ Heart rate, ↓ HRV |
| **Milk Allergy (IgE)** | Casein/whey protein reaction | Standard allergic pattern |

**Detection**: HRV decrease after dairy = sympathetic activation = inflammatory stress

#### Peanuts/Tree Nuts (Anaphylaxis Research)

**2020 Landmark Study** - Continuous cardiovascular monitoring during peanut challenges:

| Parameter | Mean Change | Clinical Significance |
|-----------|-------------|----------------------|
| Stroke Volume | ↓ 4.2% | Fluid redistribution |
| Heart Rate | ↑ 11.6% | Compensatory response |
| Peripheral Blood Flow | ↑ 19.7% | Vasodilation |
| HRV | Sympathetic activation pattern | Anxiety + epinephrine drive |

**Key Discovery**: Changes occurred **irrespective of reaction severity** - even mild reactions show cardiovascular changes.

#### Shellfish
- **Tropomyosin** is major allergen (>60% of shellfish-allergic sensitized)
- Cross-reacts with dust mites, cockroaches, insects
- Oral challenges show **higher severity scores** than tree nuts
- More cardiovascular involvement

### 2.2 HRV as Early Warning System

**Oral Food Challenge Study** (45 children):

| Group | RMS of HR Change | Significance |
|-------|------------------|--------------|
| Anaphylaxis | 8.2% | - |
| Mild Symptoms | 5.6% | p < 0.05 |
| Tolerant | 5.8% | p < 0.05 |

**Performance Metrics**:
- **ROC-AUC**: 0.89
- **Sensitivity**: 90.5%
- **Specificity**: 79.4%
- **Early Detection**: 17 minutes before clinical symptoms

---

## Part 3: Food Intolerances (Non-IgE Mediated)

### 3.1 FODMAP Intolerance

**Gut-Brain-Heart Axis**:
- Decreased parasympathetic activity frequently observed in IBS
- Decrease in vagal tone influences peripheral inflammation
- Low-FODMAP diet shown to reduce urinary histamine

| FODMAP Type | Primary Sources | HRV Effect Window |
|-------------|-----------------|-------------------|
| Fructans | Wheat, onion, garlic | 1-6 hours |
| Galactans | Beans, lentils, soy | 2-8 hours |
| Polyols | Sugar alcohols, stone fruits | 1-4 hours |
| Excess Fructose | Honey, HFCS, apples | 1-4 hours |
| Lactose | Dairy products | 30 min - 4 hours |

### 3.2 Histamine Intolerance

**Dose-Response**:
| Histamine Load | Effect |
|----------------|--------|
| 5-10 mg | Reaction in sensitive individuals |
| < 200 mg/kg | Should not exceed in food |
| > 1000 mg/meal | Intoxication possible |

**High-Histamine Foods**:
- Aged cheese (up to 2,500 mg/kg detected!)
- Wine, beer
- Fermented foods (sauerkraut, kimchi)
- Tuna, mackerel, anchovies
- Cured/smoked meats

**HRV Effects**:
- Vasodilation → hypotension → heart palpitations
- Vagal nerve stimulation → nausea
- Tachycardia in excess consumption

### 3.3 Tyramine Sensitivity

| Dose | Effect |
|------|--------|
| 3 mg | Migraine in sensitive individuals |
| 6 mg | Dangerous for MAO inhibitor users |
| 100-800 mg/kg | Toxic for general population |

**Cardiovascular Effects**:
- **Potent vasoconstrictor** → hypertension
- Increases cardiac frequency
- Can lead to heart failure or brain hemorrhage (severe cases)

### 3.4 Other Sensitivities

| Sensitivity | Mechanism | HRV Signature |
|-------------|-----------|---------------|
| **Salicylates** | Leukotriene overproduction | Inflammatory response |
| **Nightshades** | Alkaloid immune trigger | Systemic inflammation |
| **Oxalates** | Mineral binding, inflammation | Chronic HRV depression |
| **Lectins** | Gut barrier disruption | Leaky gut → inflammation |
| **Sulfites** | SO2 bronchospasm (asthmatics) | Rapid onset, wheezing |

---

## Part 4: Detection Time Windows

### 4.1 Multi-Phase Detection Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION TIME WINDOWS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IMMEDIATE (10-15 min)                                          │
│  ├─ IgE-mediated reactions                                      │
│  ├─ Severe allergies                                            │
│  └─ Anaphylaxis detection (RMS HR change ≥8%)                   │
│                                                                 │
│  SHORT-TERM (30-90 min)                                         │
│  ├─ General food sensitivities                                  │
│  ├─ Histamine reactions                                         │
│  └─ Moderate allergic reactions                                 │
│                                                                 │
│  MEDIUM-TERM (3-4 hours)                                        │
│  ├─ Delayed/biphasic reactions                                  │
│  ├─ De novo inflammatory mediators                              │
│  └─ FODMAP effects                                              │
│                                                                 │
│  NEXT-DAY (12-24 hours)                                         │
│  ├─ Chronic inflammation                                        │
│  ├─ IgG-mediated reactions                                      │
│  └─ Morning HRV assessment                                      │
│                                                                 │
│  MULTI-DAY (3-7 days)                                           │
│  ├─ Cumulative exposure effects                                 │
│  ├─ Alcohol recovery (4-5 days)                                 │
│  └─ Pattern confirmation                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Alert Thresholds

| Tier | Magnitude | Reproducibility | Confidence | Action |
|------|-----------|-----------------|------------|--------|
| **1 - High** | ≥10 points | 4+ consistent | 90-100% | Eliminate definitively |
| **2 - Moderate** | 7-10 points | 3-4 reactions | 75-89% | Eliminate and retest |
| **3 - Watch** | 5-7 points | 2-3 reactions | 50-74% | Monitor closely |
| **4 - Track** | 3-5 points | 1-2 reactions | 25-49% | Track for patterns |

---

## Part 5: Ingredient Extraction Strategy

### 5.1 Recommended Data Sources

| Source | Strengths | Use Case |
|--------|-----------|----------|
| **Open Food Facts** | Free, global, allergen tags, barcode | Primary for packaged foods |
| **USDA FoodData Central** | Accurate nutrients, whole foods | Primary for raw ingredients |
| **Nutritionix** | NLP queries, restaurant foods | Eating out scenarios |
| **FatSecret** | Global coverage, 10 allergens | International users |

### 5.2 Allergen Classification

**FDA Big 9 (USA, effective 2023)**:
1. Milk (incl. sheep, goat - 2025 update)
2. Eggs (incl. duck, quail - 2025 update)
3. Fish
4. Crustacean Shellfish
5. Tree Nuts (9 specific: almond, brazil, cashew, hazelnut, macadamia, pecan, pine nut, pistachio, walnut)
6. Peanuts
7. Wheat
8. Soybeans
9. Sesame (added 2023)

**EU Additions (Big 14)**:
- Gluten-containing cereals (broader than wheat)
- Molluscan shellfish
- Mustard
- Celery
- Lupin
- Sulfites (≥10mg/kg)

### 5.3 Hidden Allergen Detection

**Critical Ingredient Mappings**:

```python
HIDDEN_ALLERGENS = {
    "MILK": [
        "casein", "sodium caseinate", "calcium caseinate",
        "whey", "whey protein", "lactalbumin", "lactoglobulin",
        "lactose", "curds", "ghee", "paneer"
    ],
    "EGGS": [
        "albumin", "ovalbumin", "globulin", "livetin",
        "lysozyme", "lecithin (egg)"
    ],
    "WHEAT": [
        "modified food starch", "hydrolyzed vegetable protein",
        "malt extract", "malt flavoring", "soy sauce",
        "vegetable gum", "surimi"
    ],
    "SOY": [
        "soy lecithin", "hydrolyzed soy protein",
        "textured vegetable protein", "tvp"
    ]
}
```

### 5.4 Compound Quantification

**Histamine Content Database**:
```python
HISTAMINE_LEVELS = {
    # Very High (>100 mg/kg)
    "aged_cheese": {"min": 100, "max": 2500, "unit": "mg/kg"},
    "fermented_sausage": {"min": 100, "max": 400},
    "tuna_canned": {"min": 50, "max": 300},
    "sauerkraut": {"min": 50, "max": 200},

    # High (20-100 mg/kg)
    "wine_red": {"min": 20, "max": 100},
    "anchovies": {"min": 20, "max": 100},

    # Moderate (5-20 mg/kg)
    "spinach": {"min": 5, "max": 20},
    "tomato": {"min": 5, "max": 20},

    # Low (<5 mg/kg)
    "fresh_meat": {"min": 0, "max": 5},
    "fresh_fish": {"min": 0, "max": 5},
}
```

---

## Part 6: ML Model for Sensitivity Detection

### 6.1 Feature Engineering

```python
class SensitivityFeatureExtractor:
    """Extract features for allergen-HRV correlation model."""

    def extract_features(
        self,
        meal: Meal,
        hrv_data: pd.DataFrame,
        user_sensitivities: List[UserSensitivity],
    ) -> Dict[str, float]:

        features = {}

        # 1. Allergen Exposure Features (per Big 9)
        for allergen in AllergenType:
            features[f'{allergen.value}_exposure_24h'] = self._get_allergen_load(
                meal, allergen, hours=24
            )
            features[f'{allergen.value}_exposure_7d'] = self._get_allergen_load(
                meal, allergen, hours=168
            )

        # 2. Compound Load Features
        features['histamine_load_24h'] = self._estimate_histamine(meal)
        features['tyramine_load_24h'] = self._estimate_tyramine(meal)
        features['fodmap_score'] = self._calculate_fodmap_score(meal)
        features['salicylate_load'] = self._estimate_salicylate(meal)
        features['oxalate_load'] = self._estimate_oxalate(meal)

        # 3. User Sensitivity Interactions
        for sensitivity in user_sensitivities:
            if sensitivity.active:
                exposure = features.get(
                    f'{sensitivity.specificAllergen}_exposure_24h', 0
                )
                features[f'known_sensitivity_x_{sensitivity.sensitivityType}'] = (
                    exposure * sensitivity.severity.value
                )

        # 4. Temporal Features
        features['hours_since_meal'] = self._hours_since(meal.consumedAt)
        features['time_of_day'] = meal.consumedAt.hour
        features['is_late_meal'] = int(meal.consumedAt.hour >= 20)

        # 5. HRV Baseline Features
        features['hrv_baseline_7d'] = hrv_data['rmssd'].rolling(7).mean().iloc[-1]
        features['hrv_baseline_30d'] = hrv_data['rmssd'].rolling(30).mean().iloc[-1]
        features['hrv_cv'] = hrv_data['rmssd'].std() / hrv_data['rmssd'].mean()

        # 6. Cumulative Load Features
        features['total_allergen_load_24h'] = sum(
            features[f'{a.value}_exposure_24h'] for a in AllergenType
        )
        features['sensitivity_score_7d'] = self._calculate_sensitivity_score(
            meal.userId, days=7
        )

        return features
```

### 6.2 Multi-Window HRV Analysis

```python
class HRVSensitivityAnalyzer:
    """Analyze HRV patterns for food sensitivity detection."""

    WINDOWS = [
        {"name": "immediate", "start": 0, "end": 15, "weight": 1.5},
        {"name": "short_term", "start": 30, "end": 90, "weight": 1.2},
        {"name": "medium_term", "start": 180, "end": 240, "weight": 1.0},
        {"name": "next_day", "start": 720, "end": 1440, "weight": 0.8},
    ]

    def analyze_meal_response(
        self,
        meal_time: datetime,
        hrv_data: pd.DataFrame,
        baseline_rmssd: float,
    ) -> SensitivityAnalysis:

        results = {}

        for window in self.WINDOWS:
            window_start = meal_time + timedelta(minutes=window["start"])
            window_end = meal_time + timedelta(minutes=window["end"])

            window_hrv = hrv_data[
                (hrv_data['timestamp'] >= window_start) &
                (hrv_data['timestamp'] <= window_end)
            ]

            if len(window_hrv) > 0:
                avg_rmssd = window_hrv['rmssd'].mean()
                change = (avg_rmssd - baseline_rmssd) / baseline_rmssd * 100

                results[window["name"]] = {
                    "avg_rmssd": avg_rmssd,
                    "change_pct": change,
                    "significant": abs(change) > 10,  # >10% change
                    "direction": "decrease" if change < 0 else "increase",
                    "weight": window["weight"],
                }

        # Calculate weighted severity score
        severity_score = sum(
            abs(r["change_pct"]) * r["weight"]
            for r in results.values()
            if r["significant"]
        )

        return SensitivityAnalysis(
            windows=results,
            severity_score=severity_score,
            likely_reaction=severity_score > 15,
            confidence=self._calculate_confidence(results),
        )
```

### 6.3 Pattern Recognition Model

```python
class SensitivityPatternModel:
    """ML model for detecting food sensitivity patterns."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
        )
        self.feature_scaler = StandardScaler()

    def train(
        self,
        exposures: List[SensitivityExposure],
        hrv_changes: List[float],
    ):
        """Train model on historical exposure-reaction data."""

        X = self.feature_extractor.extract_batch(exposures)
        y = [1 if change < -10 else 0 for change in hrv_changes]  # >10% drop = reaction

        X_scaled = self.feature_scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

        # Store feature importance for interpretability
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

    def predict_reaction_risk(
        self,
        meal: Meal,
        user_sensitivities: List[UserSensitivity],
    ) -> ReactionRiskPrediction:
        """Predict likelihood of adverse reaction before eating."""

        features = self.feature_extractor.extract_features(
            meal, user_sensitivities
        )
        features_scaled = self.feature_scaler.transform([features])

        probability = self.model.predict_proba(features_scaled)[0][1]

        # Identify top risk factors
        risk_factors = self._get_risk_factors(features, probability)

        return ReactionRiskPrediction(
            probability=probability,
            risk_level=self._classify_risk(probability),
            risk_factors=risk_factors,
            recommendation=self._generate_recommendation(
                probability, risk_factors, user_sensitivities
            ),
        )
```

---

## Part 7: Database Schema Extensions

### 7.1 Prisma Schema Additions

```prisma
// ============================================================================
// INGREDIENT & ALLERGEN MODELS
// ============================================================================

model Ingredient {
  id              String   @id @default(cuid())
  name            String
  nameVariants    Json     // ["milk", "cow's milk", "whole milk"]
  category        IngredientCategory

  // Allergen mappings
  allergens       IngredientAllergen[]

  // Compound levels
  fodmapLevel     FodmapLevel?
  fodmapTypes     Json?    // ["fructans", "lactose"]
  histamineLevel  CompoundLevel?
  tyramineLevel   CompoundLevel?
  salicylateLevel CompoundLevel?
  oxalateLevel    CompoundLevel?
  lectinLevel     CompoundLevel?
  isNightshade    Boolean  @default(false)

  sources         Json     // ["open_food_facts", "usda"]

  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  @@index([name])
  @@index([category])
}

model IngredientAllergen {
  id            String       @id @default(cuid())
  ingredientId  String
  ingredient    Ingredient   @relation(fields: [ingredientId], references: [id])
  allergenType  AllergenType
  confidence    Float        // 0.0 to 1.0
  derivationType DerivationType

  @@unique([ingredientId, allergenType])
  @@index([allergenType])
}

model UserSensitivity {
  id               String              @id @default(cuid())
  userId           String
  user             User                @relation(fields: [userId], references: [id])

  sensitivityType  SensitivityType
  severity         SensitivitySeverity
  specificAllergen AllergenType?
  specificCompound String?             // "histamine", "tyramine"

  confirmedByTest  Boolean  @default(false)
  notes            String?
  active           Boolean  @default(true)

  // HRV correlation data
  avgHrvDrop       Float?              // Average HRV drop after exposure
  correlationScore Float?              // Pearson correlation
  exposureCount    Int      @default(0)
  reactionCount    Int      @default(0)

  createdAt        DateTime @default(now())
  updatedAt        DateTime @updatedAt

  @@index([userId, sensitivityType])
}

model SensitivityExposure {
  id               String             @id @default(cuid())
  userId           String
  user             User               @relation(fields: [userId], references: [id])
  mealId           String?
  meal             Meal?              @relation(fields: [mealId], references: [id])

  allergenType     AllergenType?
  compoundType     String?
  estimatedAmount  Float?

  // Reaction tracking
  hadReaction      Boolean
  reactionSeverity ReactionSeverity?
  symptoms         Json?              // ["hives", "GI distress"]
  onsetMinutes     Int?
  durationMinutes  Int?

  // HRV impact
  hrvBaseline      Float?
  hrvPostExposure  Float?
  hrvDropPct       Float?

  exposedAt        DateTime
  createdAt        DateTime @default(now())

  @@index([userId, exposedAt])
  @@index([userId, allergenType])
}

// ============================================================================
// ENUMS
// ============================================================================

enum AllergenType {
  MILK
  EGGS
  FISH
  SHELLFISH_CRUSTACEAN
  TREE_NUTS
  PEANUTS
  WHEAT
  SOY
  SESAME
  GLUTEN_CEREALS
  SHELLFISH_MOLLUSCAN
  MUSTARD
  CELERY
  LUPIN
  SULFITES
}

enum SensitivityType {
  ALLERGY           // IgE-mediated
  INTOLERANCE       // Non-IgE (lactose, fructose)
  SENSITIVITY       // Non-specific adverse reaction
  FODMAP
  HISTAMINE
  TYRAMINE
  SALICYLATE
  OXALATE
  LECTIN
  NIGHTSHADE
  SULFITE
}

enum SensitivitySeverity {
  MILD              // Minor discomfort
  MODERATE          // Notable symptoms
  SEVERE            // Significant reaction
  LIFE_THREATENING  // Anaphylaxis risk
}

enum CompoundLevel {
  NEGLIGIBLE
  LOW
  MEDIUM
  HIGH
  VERY_HIGH
}

enum FodmapLevel {
  LOW
  MEDIUM
  HIGH
}

enum DerivationType {
  DIRECTLY_CONTAINS
  DERIVED_FROM
  MAY_CONTAIN
  PROCESSED_WITH
  FREE_FROM
  LIKELY_CONTAINS
}

enum ReactionSeverity {
  NONE
  MILD
  MODERATE
  SEVERE
  EMERGENCY
}
```

---

## Part 8: Implementation Roadmap

### Phase 1: Core Allergen Detection (MVP)

**Week 1-2: Data Infrastructure**
- Implement Ingredient, IngredientAllergen models
- Seed Big 9 allergens + hidden sources
- Build synonym database

**Week 3-4: Basic Detection**
- Integrate Open Food Facts API
- Build fuzzy matching ingredient parser
- Implement allergen flagging

**Week 5-6: User Interface**
- Sensitivity profile setup
- Meal allergen warnings
- Exposure tracking

### Phase 2: HRV Correlation Engine

**Month 2:**
- Multi-window HRV analysis
- Exposure-HRV correlation tracking
- Pattern recognition model

**Month 3:**
- FODMAP/histamine/tyramine tracking
- Compound quantification
- Cumulative load alerts

### Phase 3: Advanced ML

**Month 4:**
- Personalized sensitivity detection
- Reaction prediction model
- Cross-reactivity warnings

**Month 5:**
- Population-level insights
- Ingredient OCR scanning
- Restaurant meal handling

---

## Part 9: API Endpoints

### Check Meal Sensitivities
```typescript
POST /api/meals/:mealId/check-sensitivities

Response: {
  warnings: [
    {
      severity: "HIGH",
      type: "ALLERGY",
      allergen: "MILK",
      foundIn: ["butter", "milk chocolate"],
      userSensitivity: {
        severity: "MODERATE",
        avgHrvDrop: -12.5,
        reactionRate: 0.85
      },
      recommendation: "Avoid - 85% reaction rate, avg 12.5ms HRV drop"
    }
  ],
  compoundAlerts: [
    {
      type: "HISTAMINE",
      level: "HIGH",
      estimatedLoad: 45,
      threshold: 10,
      sources: ["aged cheese", "wine"]
    }
  ],
  safetyScore: 0.3
}
```

### Get Sensitivity Analysis
```typescript
GET /api/users/:userId/sensitivity-analysis?allergen=MILK&days=30

Response: {
  allergen: "MILK",
  exposureCount: 15,
  reactionCount: 12,
  reactionRate: 0.80,
  avgOnsetMinutes: 45,
  commonSymptoms: ["bloating", "GI distress"],
  hrvImpact: {
    avgDrop: -11.2,
    correlation: -0.72,
    pValue: 0.001,
    significant: true
  },
  windows: {
    immediate: { avgDrop: -3.2, significant: false },
    short_term: { avgDrop: -8.5, significant: true },
    medium_term: { avgDrop: -11.2, significant: true },
    next_day: { avgDrop: -6.8, significant: true }
  },
  recommendation: "Strong evidence of milk sensitivity. Consider elimination."
}
```

---

## Part 10: Key Metrics Summary

### Detection Performance (from Research)

| Metric | Value | Notes |
|--------|-------|-------|
| **Sensitivity** | 90.5% | Anaphylaxis detection |
| **Specificity** | 79.4% | True negative rate |
| **ROC-AUC** | 0.89 | Excellent discrimination |
| **Early Detection** | 17 min | Before clinical symptoms |
| **RMS HR Change (Reaction)** | 8.2% | vs 5.6% tolerant |

### HRV Thresholds for Alerts

| Deviation | Confidence | Action |
|-----------|------------|--------|
| ≥10 points + 4x reproducible | Very High | Eliminate food |
| 7-10 points + 3x reproducible | High | Eliminate and retest |
| 5-7 points + 2x reproducible | Moderate | Monitor closely |
| 3-5 points + single occurrence | Low | Track patterns |

### Compound Thresholds

| Compound | Safe | Caution | High Risk |
|----------|------|---------|-----------|
| Histamine | <5 mg | 5-100 mg | >100 mg |
| Tyramine | <3 mg | 3-100 mg | >100 mg |
| Oxalates | <10 mg/serving | 10-30 mg | >30 mg |
| Salicylates | <0.25 mg/serving | 0.25-0.5 mg | >0.5 mg |

---

## Appendix A: Cross-Reactivity Matrix

| If Allergic To | Also Watch For |
|----------------|----------------|
| Peanuts | Tree nuts, lupins, legumes |
| Shellfish (Crustacean) | Mollusks, dust mites, cockroaches |
| Milk | Goat/sheep milk, all ruminants |
| Birch Pollen | Apples, pears, cherries, almonds, hazelnuts |
| Ragweed | Melons, bananas, zucchini |
| Latex | Bananas, avocados, kiwi, chestnuts |

## Appendix B: Ingredient Synonym Database

```json
{
  "milk": ["dairy", "casein", "whey", "lactose", "cream", "butter", "ghee"],
  "egg": ["albumin", "globulin", "lysozyme", "mayonnaise", "meringue"],
  "wheat": ["flour", "semolina", "durum", "spelt", "kamut", "triticale"],
  "soy": ["soya", "edamame", "tofu", "tempeh", "miso", "lecithin"],
  "peanut": ["groundnut", "arachis oil", "monkey nuts"],
  "tree_nut": ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut"],
  "fish": ["anchovy", "cod", "salmon", "tuna", "sardine", "worcestershire"],
  "shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish", "scallop"],
  "sesame": ["tahini", "halvah", "hummus", "benne seeds"]
}
```

---

*Document Version: 1.0*
*Created: December 2025*
*For: Nutri ML Service - Food Sensitivity Detection*
