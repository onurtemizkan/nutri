# Hierarchical Food Recognition Architecture

## Architecture Decision Record (ADR)

**Status:** Accepted
**Date:** December 2025
**Decision Makers:** Nutri Development Team

---

## Context

### Problem Statement

USDA FoodData Central contains 500K+ food items. The question: should we train a classifier to recognize all of them directly?

### State of the Art Analysis

Academic research shows clear accuracy degradation as class count increases:

| Dataset  | Classes | Top-1 Accuracy |
| -------- | ------- | -------------- |
| Food-101 | 101     | 95%            |
| Food-172 | 172     | 94%            |
| Food-251 | 251     | 81%            |
| Food-256 | 256     | 83%            |

**Pattern:** Every 2.5x increase in classes results in ~13% accuracy drop.

At 500K classes, accuracy would be effectively random (~0.0002%).

### Visual Ambiguity Problem

Many USDA items are visually identical and impossible to distinguish from images alone:

- "Chicken breast, grilled, skin removed" vs "Chicken breast, grilled, with skin"
- "Brown rice, cooked" vs "White rice, cooked" (similar appearance when cooked)
- "2% Milk, Brand A" vs "2% Milk, Brand B" (identical packaging styles)
- "Ground beef, 80% lean" vs "Ground beef, 85% lean"

No computer vision model can distinguish these variants - only metadata (brand, preparation method) and user confirmation can resolve ambiguity.

---

## Decision

**We will NOT attempt direct 500K-class classification.**

Instead, we implement a **Hierarchical Search-First Architecture** where:

1. Image classifier outputs **300-500 food categories** (achievable with 85%+ accuracy)
2. Category predictions serve as **search queries** for USDA FoodData Central
3. User **confirms selection** from search results
4. System **learns from corrections** to improve suggestions over time

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FOOD RECOGNITION FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📷 Photo ──► Classifier (300-500 categories)                      │
│                    │                                                │
│                    ▼                                                │
│              "chicken_breast" (89% confidence)                      │
│                    │                                                │
│                    ▼                                                │
│  USDA Search: query="chicken breast"                                │
│                    │                                                │
│                    ▼                                                │
│  Results: [grilled / fried / roasted / raw / with skin / skinless] │
│                    │                                                │
│                    ▼                                                │
│  User Confirms: "Chicken breast, grilled, without skin"            │
│                    │                                                │
│                    ▼                                                │
│  AR Portion: 150g × nutrition/100g = Final Nutrition               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Parallel Path: Barcode Scanner

For packaged foods, bypass classification entirely:

```
📦 Barcode ──► USDA Branded Lookup ──► Exact Nutrition (no ambiguity)
```

---

## Category Taxonomy Design

### Principles

1. **Visually Distinguishable**: Categories must be differentiable from images
2. **USDA Aligned**: Map to USDA food groups for effective search
3. **Granular Enough**: Narrow search results to <50 items
4. **Not Too Granular**: Maintain 85%+ classification accuracy

### Proposed 300-500 Category Structure

#### Proteins (~80 categories)

```
Poultry:
  chicken_breast, chicken_thigh, chicken_wing, chicken_drumstick,
  chicken_whole, turkey_breast, turkey_ground, duck_breast

Red Meat:
  beef_steak, beef_ground, beef_roast, beef_ribs,
  pork_chop, pork_tenderloin, pork_ribs, pork_ground,
  lamb_chop, lamb_leg, lamb_ground

Seafood:
  salmon_fillet, tuna_steak, tuna_canned, cod_fillet,
  shrimp, lobster, crab, scallops, oysters, mussels,
  tilapia, halibut, mackerel, sardines

Processed:
  bacon, sausage, hot_dog, deli_meat, jerky
```

#### Grains (~50 categories)

```
Rice:
  white_rice, brown_rice, fried_rice, rice_pilaf, sushi_rice

Pasta:
  spaghetti, penne, fusilli, lasagna, mac_and_cheese

Bread:
  white_bread, wheat_bread, sourdough, bagel, croissant,
  baguette, pita, naan, tortilla_flour, tortilla_corn

Breakfast:
  oatmeal, cereal_flakes, cereal_granola, pancakes, waffles
```

#### Vegetables (~70 categories)

```
Leafy:
  spinach, lettuce_romaine, lettuce_iceberg, kale, arugula,
  cabbage_green, cabbage_red, chard, collard_greens

Root:
  potato_baked, potato_mashed, potato_fries, sweet_potato,
  carrot, beet, turnip, parsnip, radish

Cruciferous:
  broccoli, cauliflower, brussels_sprouts, bok_choy

Other:
  tomato, bell_pepper_red, bell_pepper_green, cucumber,
  zucchini, eggplant, asparagus, green_beans, corn, peas
```

#### Fruits (~50 categories)

```
Citrus:
  orange, lemon, lime, grapefruit, tangerine

Berries:
  strawberry, blueberry, raspberry, blackberry

Stone:
  peach, plum, cherry, apricot, mango

Other:
  apple, banana, grapes, watermelon, cantaloupe, pineapple,
  kiwi, pear, pomegranate
```

#### Dairy (~30 categories)

```
Milk:
  milk_whole, milk_2percent, milk_skim, milk_chocolate

Cheese:
  cheese_cheddar, cheese_mozzarella, cheese_swiss, cheese_parmesan,
  cheese_cream, cheese_feta, cheese_brie

Yogurt:
  yogurt_plain, yogurt_greek, yogurt_fruit

Other:
  butter, cream, sour_cream, ice_cream, cottage_cheese
```

#### Mixed Dishes (~80 categories)

```
Pizza:
  pizza_cheese, pizza_pepperoni, pizza_veggie, pizza_meat

Sandwiches:
  burger, sandwich_sub, sandwich_grilled, wrap, taco, burrito

Asian:
  sushi_roll, ramen, pho, pad_thai, curry, stir_fry, dim_sum

Italian:
  lasagna, risotto, ravioli

Mexican:
  enchilada, quesadilla, nachos, tamale

Salads:
  salad_caesar, salad_greek, salad_garden, salad_cobb

Soups:
  soup_tomato, soup_chicken, soup_vegetable, soup_cream
```

#### Snacks & Sweets (~40 categories)

```
Sweet:
  cookie, brownie, cake, donut, muffin, pie, ice_cream_cone

Savory:
  chips, pretzels, popcorn, crackers, nuts_mixed

Candy:
  chocolate_bar, candy_gummy, candy_hard
```

---

## Implementation Details

### Tier 1: Coarse Category Classifier

**Model Architecture:**

- Base: Vision Transformer (ViT-B/16) or EfficientNet-B4
- Pre-trained on ImageNet, fine-tuned on Food-2K + custom dataset
- Output: Top-5 categories with confidence scores
- Inference: <100ms on CPU, <20ms on GPU

**Training Data Sources:**

- Food-101 (101K images)
- Food-2K (2M images)
- Custom labeled dataset (10K images)

**Target Metrics:**

- Top-1 Accuracy: >85%
- Top-5 Accuracy: >95%
- Per-category minimum: >80%

### Tier 2: USDA Search Integration

**Search Strategy:**

```typescript
interface ClassificationHints {
  category: string; // "chicken_breast"
  confidence: number; // 0.89
  alternatives: string[]; // ["turkey_breast", "pork_chop"]
  cookingHints: string[]; // ["appears grilled", "charred edges"]
  portionEstimate?: number; // grams from AR
}

async function searchWithHints(hints: ClassificationHints): Promise<USDAFood[]> {
  // 1. Primary search on category
  const primaryResults = await usda.search(hints.category);

  // 2. Filter by data type (Foundation > SR Legacy > Branded)
  const filtered = filterByDataType(primaryResults, ['Foundation', 'SR Legacy']);

  // 3. Re-rank by cooking method if detected
  if (hints.cookingHints.length > 0) {
    return rerankByCooking(filtered, hints.cookingHints);
  }

  return filtered;
}
```

### Tier 3: User Confirmation UI

**Design Principles:**

- Target: <2 taps to confirm correct item
- Show top-5 suggestions prominently
- Group variants logically (cooking methods, portions)
- Allow quick search if none match

```
┌────────────────────────────────────────┐
│  Detected: Chicken Breast  (89%)       │
│  ────────────────────────────────────  │
│  Select preparation:                   │
│                                        │
│  ○ Grilled, skinless       120 cal    │
│  ○ Grilled, with skin      165 cal    │
│  ○ Fried, breaded          280 cal    │
│  ○ Roasted, skinless       145 cal    │
│  ○ Raw                     110 cal    │
│                                        │
│  [Not seeing your food? Search →]      │
└────────────────────────────────────────┘
```

---

## Success Metrics

| Metric                        | Target  | Measurement                     |
| ----------------------------- | ------- | ------------------------------- |
| Classifier Top-5 Accuracy     | >95%    | Held-out test set               |
| USDA Search Recall            | >95%    | Correct item in top 10 results  |
| User Confirmation Rate        | <2 taps | Analytics tracking              |
| End-to-End Nutrition Accuracy | ±10%    | Comparison to weighed reference |
| Classification Latency        | <100ms  | P95 on production traffic       |
| Search Latency                | <500ms  | P95 with caching                |

---

## Alternatives Considered

### Alternative 1: Direct 500K-Class Classifier

**Rejected because:**

- Impossible accuracy (would be ~0.0002%)
- Visual ambiguity cannot be resolved by ML alone
- No research supports this approach

### Alternative 2: Smaller Category Set (50 categories)

**Rejected because:**

- Search results too broad (>100 items per category)
- User would need to scroll/search extensively
- Defeats purpose of ML assistance

### Alternative 3: Hierarchical Multi-Model (coarse → fine → finer)

**Partially adopted:**

- Coarse classifier (300-500) is implemented
- Fine-grained classifiers for specific categories could be added later
- But primary approach is search-first

---

## Implementation Status

### Completed

- [x] USDA FoodData Central API integration (Task 13)
- [x] Coarse category classifier with CLIP (Task 13.5)
- [x] Search-assisted classification endpoint (Task 13.6)
- [x] User feedback collection system (Task 13.7)
- [x] Mobile food search UI (Task 13.8)

### In Progress

- [ ] Category taxonomy refinement to 300-500 classes
- [ ] Model fine-tuning on expanded categories
- [ ] Accuracy benchmarking on test set

### Future Enhancements

- [ ] Category-specific fine-grained classifiers
- [ ] Continuous learning from user corrections
- [ ] Multi-food detection in single image

---

## References

1. **Food-101**: Bossard et al., "Food-101 Mining Discriminative Components with Random Forests" (2014)
2. **ViT for Food**: Vision Transformer achieving 95% on Food-101
3. **MyFitnessPal Meal Scan**: Uses similar hierarchical approach
4. **USDA FoodData Central**: https://fdc.nal.usda.gov/api-guide/

---

## Appendix: Category-to-USDA Search Mapping

Example mappings for effective USDA queries:

| Category          | USDA Search Query   | Expected Results |
| ----------------- | ------------------- | ---------------- |
| `chicken_breast`  | "chicken breast"    | ~40 variants     |
| `pizza_pepperoni` | "pizza pepperoni"   | ~25 variants     |
| `salmon_fillet`   | "salmon fillet"     | ~30 variants     |
| `apple`           | "apple raw"         | ~15 variants     |
| `white_rice`      | "rice white cooked" | ~20 variants     |

The search query can be enhanced with cooking method detection:

- "appears grilled" → add "grilled" to query
- "appears fried" → add "fried" to query
- "appears raw" → add "raw" to query
