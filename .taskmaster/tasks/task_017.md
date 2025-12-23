# Task ID: 17

**Title:** Design Hierarchical Food Recognition Architecture

**Status:** pending

**Dependencies:** 13

**Priority:** high

**Description:** Document and implement the hierarchical food recognition architecture where the image classifier outputs food CATEGORIES (300-500 classes) that serve as search queries for USDA FoodData Central, NOT direct 500K-class classification which is infeasible.

**Details:**

## Architecture Decision Record

### Problem Statement
USDA FoodData Central contains 500K+ foods. Should we train a classifier to recognize all of them?

### Decision: NO - Use Hierarchical Architecture

### Rationale

**State of the Art Accuracy by Class Count:**
- Food-101 (101 classes): 95% accuracy
- Food-172 (172 classes): 94% accuracy  
- Food-251 (251 classes): 81% accuracy
- Food-256 (256 classes): 83% accuracy

Pattern: Every 2.5x increase = ~13% accuracy drop. At 500K classes, accuracy would be effectively random.

**Visual Ambiguity Problem:**
Many USDA items are visually identical:
- "Chicken breast, grilled, skin removed" vs "with skin"
- "Brown rice, cooked" vs "white rice, cooked"
- Different brands of identical products

No classifier can distinguish these - only metadata and user confirmation can.

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FOOD RECOGNITION FLOW                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📷 Photo ──► Classifier (300-500 categories)               │
│                    │                                        │
│                    ▼                                        │
│              "chicken_breast" (89% confidence)              │
│                    │                                        │
│                    ▼                                        │
│  USDA Search: query="chicken breast"                        │
│                    │                                        │
│                    ▼                                        │
│  Results: [grilled/fried/roasted/raw variants...]           │
│                    │                                        │
│                    ▼                                        │
│  User Confirms: "Chicken breast, grilled, no skin"          │
│                    │                                        │
│                    ▼                                        │
│  AR Portion: 150g × nutrition/100g = Final Nutrition        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Parallel Path: Barcode Scanner
```
📦 Barcode ──► USDA Branded Lookup ──► Exact Nutrition
```

### Implementation Tasks

1. **Classifier Upgrade (Task 2)**
   - Expand from 111 → 300-500 categories
   - Architecture: Vision Transformer (ViT-B/16) or EfficientNet-B4
   - Target: 85%+ top-5 accuracy
   - Output: category string for USDA search

2. **USDA Integration (Task 13)**
   - Implement search API
   - Category → search results mapping
   - User confirmation UI

3. **Barcode Scanner (Task 14)**
   - Direct USDA branded lookup
   - No classification needed

### Category Taxonomy Design

Expand current 111 classes following USDA structure:
- Proteins: chicken_breast, chicken_thigh, beef_steak, ground_beef, salmon_fillet...
- Grains: white_rice, brown_rice, pasta, bread_white, bread_wheat...
- Vegetables: broccoli, spinach, carrot, potato_baked, potato_mashed...
- Fruits: apple, banana, orange, strawberry, blueberry...
- Dairy: milk, yogurt_plain, yogurt_greek, cheese_cheddar...
- Mixed: pizza, burger, sandwich, salad, soup...

### Success Metrics
- Classifier top-5 accuracy: >85%
- USDA search recall: >95% (correct item in top 10 results)
- User confirmation rate: <2 taps to find correct item
- End-to-end nutrition accuracy: within 10% of actual

### References
- SOTA Food Classification: ViT achieving 95% on Food-101
- MyFitnessPal Meal Scan: Uses same hierarchical approach
- USDA FoodData Central API: https://fdc.nal.usda.gov/api-guide/

**Test Strategy:**

1. Document review with team
2. Validate category taxonomy covers 95%+ of common foods
3. Prototype search flow with mock classifier output
4. User testing of confirmation UI (target <2 taps)
5. Accuracy benchmarking on held-out food images
