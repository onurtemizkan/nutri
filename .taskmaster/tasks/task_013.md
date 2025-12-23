# Task ID: 13

**Title:** Integrate USDA FoodData Central Database with Scalable Food Classification System

**Status:** pending

**Dependencies:** None

**Priority:** high

**Description:** Integrate the USDA FoodData Central API to expand the food database from ~100 items to 500K+ foods, AND implement a scalable multi-tier food classification architecture. This addresses the critical question: yes, integrating USDA's 500K+ food database requires a fundamentally different classification approach than the current ~100-class model.

**Details:**

## Overview
The current food database in `ml-service/app/data/food_database.py` has approximately 100 items with a simple classifier (`FoodAnalysisService._classify_food`). Integrating USDA FoodData Central's 500K+ foods requires a **multi-tier classification architecture** because:

1. **No ML model can reliably classify 500K+ food classes** - even state-of-the-art research focuses on 500-2000 classes
2. **USDA uses 5 distinct data types** with different classification schemas (Foundation, SR Legacy, Survey/FNDDS, Branded, Experimental)
3. **User search + AI-assisted refinement** is more practical than pure image classification

## USDA FoodData Central API
- **API Endpoint**: https://api.nal.usda.gov/fdc/v1/
- **API Key**: Free, requires registration at https://fdc.nal.usda.gov/api-key-signup.html
- **Rate Limits**: 1,000 requests/hour per IP (can request increase)
- **Data Types**: 
  - Foundation Foods (unprocessed/lightly processed)
  - SR Legacy (comprehensive, final release 2018)
  - Survey/FNDDS (dietary studies 2021-2023)
  - Branded (commercial products)
  - Experimental (research data)

## Multi-Tier Classification Architecture

### Tier 1: Coarse-Grained Visual Classifier (ML Model)
**Purpose**: Classify images into 20-50 high-level food categories
**Implementation**: `ml-service/app/ml_models/food_classifier_v3.py`

```python
class FoodCategory(str, Enum):
    # 25-30 categories mapping to USDA food groups
    FRUITS_FRESH = "fruits_fresh"
    FRUITS_PROCESSED = "fruits_processed"
    VEGETABLES_LEAFY = "vegetables_leafy"
    VEGETABLES_ROOT = "vegetables_root"
    VEGETABLES_OTHER = "vegetables_other"
    MEAT_RED = "meat_red"
    MEAT_POULTRY = "meat_poultry"
    SEAFOOD_FISH = "seafood_fish"
    SEAFOOD_SHELLFISH = "seafood_shellfish"
    DAIRY_MILK = "dairy_milk"
    DAIRY_CHEESE = "dairy_cheese"
    DAIRY_YOGURT = "dairy_yogurt"
    GRAINS_BREAD = "grains_bread"
    GRAINS_PASTA = "grains_pasta"
    GRAINS_RICE = "grains_rice"
    GRAINS_CEREAL = "grains_cereal"
    LEGUMES = "legumes"
    NUTS_SEEDS = "nuts_seeds"
    BEVERAGES_HOT = "beverages_hot"
    BEVERAGES_COLD = "beverages_cold"
    SNACKS_SWEET = "snacks_sweet"
    SNACKS_SAVORY = "snacks_savory"
    MIXED_DISHES = "mixed_dishes"
    FAST_FOOD = "fast_food"
    CONDIMENTS_SAUCES = "condiments_sauces"
    # ... additional categories

@dataclass
class CoarseClassification:
    category: FoodCategory
    confidence: float
    subcategory_hints: List[str]  # "appears sliced", "grilled texture", etc.
    color_profile: Dict[str, float]  # dominant colors for refinement
    texture_features: Dict[str, float]  # smooth, grainy, fibrous, etc.
```

**Model Architecture**:
- Base: EfficientNet-B4 or ConvNeXt-Base (pretrained on ImageNet)
- Fine-tuned on Food-2K + custom dataset
- Output: Top-3 categories with confidence scores
- Inference time: <100ms on CPU, <20ms on GPU

### Tier 2: Category-Specific Fine-Grained Classifier (Optional ML)
**Purpose**: Refine within categories (e.g., "apple" vs "pear" within FRUITS_FRESH)
**Implementation**: Specialized sub-models loaded on-demand

```python
class FinegrainedClassifierRegistry:
    """Registry of category-specific classifiers"""
    
    CLASSIFIERS = {
        FoodCategory.FRUITS_FRESH: "models/fruits_classifier_v1.onnx",
        FoodCategory.MEAT_RED: "models/red_meat_classifier_v1.onnx",
        FoodCategory.MIXED_DISHES: None,  # Too complex, skip to search
        # ...
    }
    
    async def classify_finegrained(
        self, 
        image: Image.Image, 
        category: FoodCategory
    ) -> Optional[List[FinegrainedPrediction]]:
        """Returns None if no specialized classifier available"""
        model_path = self.CLASSIFIERS.get(category)
        if not model_path:
            return None
        # Load and run category-specific model
        ...
```

**Category Coverage**:
- Fruits: ~200 classes (achievable with 85%+ accuracy)
- Vegetables: ~150 classes
- Meat/Poultry: ~100 classes
- Seafood: ~150 classes
- Branded/Processed: Skip (use text search)
- Mixed dishes: Skip (use user input + search)

### Tier 3: USDA Search Integration (Primary Lookup)
**Purpose**: Map classifications to specific USDA FDC entries
**Implementation**: `server/src/services/foodDatabaseService.ts`

```typescript
interface USDASearchStrategy {
  // Combine visual classification with text search
  searchWithClassificationContext(
    query: string,
    classificationHints: ClassificationHints
  ): Promise<USDASearchResult[]>;
}

interface ClassificationHints {
  coarseCategory: string;
  finegrainedSuggestions?: string[];
  colorProfile?: Record<string, number>;
  cookingMethod?: string;
  brandDetected?: string;  // OCR from packaging
  portionEstimate?: number;  // grams from AR
}

const searchWithContext = async (
  userQuery: string,
  hints: ClassificationHints
): Promise<USDAFoodItem[]> => {
  // 1. Build enhanced search query
  const enhancedQuery = buildEnhancedQuery(userQuery, hints);
  
  // 2. Filter by USDA data types based on category
  const dataTypes = getRelevantDataTypes(hints.coarseCategory);
  // Foundation/SR Legacy for whole foods
  // Branded for packaged foods
  // Survey for mixed dishes
  
  // 3. Execute search with USDA API
  const results = await usdaApi.search({
    query: enhancedQuery,
    dataType: dataTypes,
    pageSize: 25,
    sortBy: 'dataType.keyword',  // Prefer Foundation over Branded
  });
  
  // 4. Re-rank results using classification hints
  return rerankResults(results, hints);
};
```

### Tier 4: User Confirmation + Learning Loop
**Purpose**: Correct misclassifications and improve over time
**Implementation**: Feedback-driven learning system

```typescript
// server/src/services/foodFeedbackService.ts
interface FoodFeedback {
  originalPrediction: string;
  userSelection: string;  // FDC ID selected
  imageHash: string;      // For deduplication
  classificationHints: ClassificationHints;
  timestamp: Date;
  userId: string;
}

// Aggregate feedback for model retraining triggers
const aggregateFeedback = async (): Promise<RetrainingSignal> => {
  // When >100 corrections for a category, signal retraining
  ...
};
```

## Backend API Routes (Extended)

### Food Search Endpoints
```
GET /api/foods/search?q={query}&limit={limit}&page={page}&dataType={type}
GET /api/foods/:fdcId
GET /api/foods/:fdcId/nutrients
GET /api/foods/popular
GET /api/foods/recent  (user's recent selections)
```

### Classification-Assisted Endpoints
```
POST /api/foods/classify-and-search
  Body: { image: base64, dimensions?: ARDimensions }
  Response: {
    classification: { category, confidence, suggestions },
    searchResults: USDAFoodItem[],
    portionEstimate?: number
  }

POST /api/foods/feedback
  Body: { classificationId, selectedFdcId, wasCorrect }
```

## Caching Strategy (Multi-Layer)

### Layer 1: Edge Cache (CDN)
- Popular food queries: 24-hour TTL
- Static food data: 7-day TTL

### Layer 2: Redis Cache
```typescript
// Search results: 1-hour TTL
await redis.setex(`search:${hash(query+dataTypes)}`, 3600, JSON.stringify(results));

// Individual food data: 24-hour TTL
await redis.setex(`food:${fdcId}`, 86400, JSON.stringify(foodData));

// User's recent foods: 30-day TTL
await redis.zadd(`user:${userId}:foods`, timestamp, fdcId);

// Classification results: 1-hour TTL (for same image)
await redis.setex(`classify:${imageHash}`, 3600, JSON.stringify(classification));
```

### Layer 3: Local SQLite Cache (Mobile)
```typescript
// Cache top 10K most searched foods locally
// Weekly sync for updates
// Enables offline search with degraded ranking
```

## Nutrient Mapping (Extended)

Map USDA nutrient IDs to our schema (expanded set):
```typescript
const NUTRIENT_MAPPING = {
  // Core macros
  1003: 'protein',      // g
  1004: 'fat',          // g
  1005: 'carbs',        // g (by difference)
  1008: 'calories',     // kcal
  
  // Fiber & sugars
  1079: 'fiber',        // g
  2000: 'sugars_total', // g
  1235: 'sugars_added', // g
  
  // Fats breakdown
  1258: 'saturated_fat',     // g
  1292: 'monounsaturated_fat', // g
  1293: 'polyunsaturated_fat', // g
  1257: 'trans_fat',         // g
  1253: 'cholesterol',       // mg
  
  // Minerals
  1093: 'sodium',       // mg
  1092: 'potassium',    // mg
  1087: 'calcium',      // mg
  1089: 'iron',         // mg
  1090: 'magnesium',    // mg
  
  // Vitamins
  1106: 'vitamin_a',    // mcg RAE
  1162: 'vitamin_c',    // mg
  1114: 'vitamin_d',    // mcg
  
  // Amino acids (for Lysine/Arginine tracking)
  1213: 'lysine',       // g
  1220: 'arginine',     // g
};
```

## ML Model Training Pipeline

### Phase 1: Coarse Classifier Training
```python
# ml-service/scripts/train_coarse_classifier.py

# Dataset: Food-2K + custom images
# Classes: 25-30 food categories
# Architecture: EfficientNet-B4
# Training: 50 epochs, cosine LR schedule
# Target: >90% top-1 accuracy, >98% top-3 accuracy

from torchvision.models import efficientnet_b4
from torch.utils.data import DataLoader

def train_coarse_classifier():
    model = efficientnet_b4(pretrained=True)
    model.classifier[-1] = nn.Linear(1792, NUM_CATEGORIES)
    
    # Training config
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    # Mixed precision training
    scaler = GradScaler()
    
    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, scaler)
        validate(model, val_loader)
        scheduler.step()
```

### Phase 2: Fine-Grained Classifier Training (Per Category)
```python
# Only for categories with clear visual distinctions
# Skip: mixed dishes, branded foods, beverages

TRAINABLE_CATEGORIES = [
    'fruits_fresh',    # ~200 classes
    'vegetables',      # ~150 classes
    'meat_raw',        # ~50 classes
    'seafood',         # ~150 classes
]

def train_finegrained_classifier(category: str):
    # Filter USDA FDC images for category
    # Use Foundation Foods data type for quality images
    # Augmentation: color jitter, rotation, scale
    # Model: ResNet-50 or ConvNeXt-Small
    ...
```

### Phase 3: Continuous Learning Pipeline
```python
# Trigger retraining when:
# 1. >100 corrections for a category
# 2. Monthly scheduled retraining
# 3. New USDA data release

@celery.task
def check_retraining_triggers():
    feedback_stats = aggregate_user_feedback()
    for category, stats in feedback_stats.items():
        if stats.correction_count > 100:
            queue_retraining_job(category, stats.feedback_data)
```

## Environment Variables (Extended)
```env
# USDA API
USDA_API_KEY=your-api-key-here
USDA_API_BASE_URL=https://api.nal.usda.gov/fdc/v1
USDA_RATE_LIMIT_PER_HOUR=1000

# ML Model Configuration
ML_MODEL_COARSE_PATH=models/food_coarse_v1.onnx
ML_MODEL_INFERENCE_DEVICE=cpu  # or cuda
ML_MODEL_CONFIDENCE_THRESHOLD=0.6

# Caching
REDIS_URL=redis://localhost:6379
CACHE_SEARCH_TTL_SECONDS=3600
CACHE_FOOD_TTL_SECONDS=86400

# Feature Flags
ENABLE_FINEGRAINED_CLASSIFICATION=true
ENABLE_BRANDED_FOOD_SEARCH=true
ENABLE_FEEDBACK_COLLECTION=true
```

## Feasibility Analysis

### What IS Feasible:
1. ✅ Integrating USDA API for 500K+ food search
2. ✅ Coarse classification into 25-50 categories (>90% accuracy achievable)
3. ✅ Fine-grained classification for select categories (fruits, vegetables, meat)
4. ✅ Hybrid search combining visual hints + text queries
5. ✅ Progressive enhancement with user feedback

### What is NOT Feasible:
1. ❌ Direct 500K-class image classifier (no model can do this reliably)
2. ❌ Accurate classification of branded/packaged foods (need barcode/OCR)
3. ❌ Mixed dishes classification without user input
4. ❌ Real-time model retraining (must be batch/scheduled)

### Recommended Approach:
**Hybrid Search-First Architecture**
- Use ML for coarse categorization + portion estimation
- Let user search/select from USDA results
- Learn from corrections to improve suggestions
- Barcode scanning for packaged foods (Phase 2)

## Success Metrics (Extended)
- Coarse classification: >90% top-1, >98% top-3 accuracy
- Search returns results in <500ms (with caching)
- 95%+ of common foods found in search
- User selects from top-5 suggestions >80% of time
- Nutrition data accuracy verified against USDA source
- Model retraining pipeline completes in <4 hours

**Test Strategy:**

## Testing Strategy (Comprehensive)

### Unit Tests

#### ML Classification Tests
```python
# ml-service/tests/test_coarse_classifier.py
class TestCoarseClassifier:
    def test_classification_output_shape(self):
        """Verify model outputs correct number of categories"""
        
    def test_confidence_scores_sum_to_one(self):
        """Softmax outputs should sum to ~1.0"""
        
    def test_inference_time_under_threshold(self):
        """Inference should complete in <100ms CPU"""
        
    def test_batch_inference(self):
        """Multiple images processed correctly"""
        
    def test_model_handles_various_image_sizes(self):
        """Resizing/preprocessing works for any input size"""
        
    def test_model_handles_grayscale_images(self):
        """Graceful handling of non-RGB input"""
```

#### USDA API Client Tests
```typescript
// server/src/__tests__/foodDatabaseService.test.ts
describe('FoodDatabaseService', () => {
  test('searchFoods returns properly typed results');
  test('searchFoods handles empty query gracefully');
  test('searchFoods respects dataType filter');
  test('searchFoods paginates correctly');
  test('getFoodById returns complete nutrition data');
  test('getFoodById handles non-existent FDC ID');
  test('nutrient mapping transforms USDA format to app schema');
  test('nutrient mapping handles missing nutrients gracefully');
  test('rate limit handling backs off appropriately');
});
```

#### Caching Tests
```typescript
describe('FoodCacheService', () => {
  test('cache hit returns data without API call');
  test('cache miss fetches from API and caches');
  test('cache expiration triggers fresh fetch');
  test('cache handles Redis connection failure gracefully');
  test('search result deduplication works correctly');
});
```

### Integration Tests

#### Classification Pipeline Tests
```python
# ml-service/tests/integration/test_classification_pipeline.py
class TestClassificationPipeline:
    async def test_end_to_end_classification(self):
        """Image → Coarse → Fine-grained → Search suggestions"""
        
    async def test_classification_with_ar_dimensions(self):
        """Classification combined with portion estimation"""
        
    async def test_fallback_when_classifier_unavailable(self):
        """System degrades gracefully without ML model"""
        
    async def test_classification_caching(self):
        """Same image hash returns cached result"""
```

#### USDA API Integration Tests
```typescript
// Run with actual API (rate-limited test account)
describe('USDA API Integration', () => {
  test('search for "apple" returns Foundation Foods results');
  test('search for "coca cola" returns Branded results');
  test('getFoodById retrieves full nutrient profile');
  test('API handles special characters in query');
  test('API timeout is handled gracefully');
  test('concurrent requests respect rate limits');
});
```

#### Hybrid Search Tests
```typescript
describe('Hybrid Search', () => {
  test('classification hints improve search ranking');
  test('color profile helps distinguish similar foods');
  test('cooking method hint filters appropriate results');
  test('portion estimate affects serving size suggestions');
});
```

### Mobile Tests

#### Search UI Tests
```typescript
// __tests__/screens/FoodSearchScreen.test.tsx
describe('FoodSearchScreen', () => {
  test('renders search input and results list');
  test('debounces search input (300ms)');
  test('displays loading state during search');
  test('displays error state on API failure');
  test('displays empty state with suggestions');
  test('tapping result opens food detail');
  test('recent searches displayed on focus');
  test('clear search button works');
});
```

#### Classification Integration Tests
```typescript
describe('FoodClassificationScreen', () => {
  test('camera capture triggers classification');
  test('classification results displayed with confidence');
  test('user can override classification');
  test('AR dimensions captured when available');
  test('portion estimate displayed');
  test('proceed to search with classification hints');
});
```

#### Offline Behavior Tests
```typescript
describe('Offline Mode', () => {
  test('cached foods searchable offline');
  test('recent selections available offline');
  test('graceful degradation message shown');
  test('classification works offline (model in app)');
  test('sync queue for pending feedback');
});
```

### Performance Tests

#### Classification Performance
```python
def test_coarse_classifier_latency():
    """Target: <100ms on CPU, <20ms on GPU"""
    model = load_coarse_classifier()
    images = load_test_images(100)
    
    start = time.time()
    for img in images:
        model.predict(img)
    avg_latency = (time.time() - start) / 100
    
    assert avg_latency < 0.1  # 100ms

def test_finegrained_classifier_latency():
    """Target: <150ms on CPU"""
    ...
```

#### Search Performance
```typescript
describe('Search Performance', () => {
  test('cached search returns in <50ms');
  test('uncached search returns in <500ms');
  test('concurrent searches (10) complete in <2s');
  test('large result sets (100 items) render in <100ms');
});
```

#### Memory Tests
```python
def test_model_memory_footprint():
    """Coarse model should use <500MB RAM"""
    import tracemalloc
    tracemalloc.start()
    
    model = load_coarse_classifier()
    current, peak = tracemalloc.get_traced_memory()
    
    assert peak < 500 * 1024 * 1024  # 500MB
```

### Accuracy Tests

#### Classification Accuracy
```python
# Run on held-out test set
def test_coarse_classifier_accuracy():
    """Target: >90% top-1, >98% top-3"""
    model = load_coarse_classifier()
    test_set = load_test_dataset()
    
    top1_correct = 0
    top3_correct = 0
    
    for image, label in test_set:
        predictions = model.predict_top_k(image, k=3)
        if predictions[0].label == label:
            top1_correct += 1
        if label in [p.label for p in predictions]:
            top3_correct += 1
    
    top1_acc = top1_correct / len(test_set)
    top3_acc = top3_correct / len(test_set)
    
    assert top1_acc > 0.90
    assert top3_acc > 0.98

def test_per_category_accuracy():
    """Ensure no category has <80% accuracy"""
    ...
```

#### Nutrition Data Accuracy
```typescript
describe('Nutrition Data Accuracy', () => {
  test('calories within 5% of USDA source');
  test('macros within 5% of USDA source');
  test('portion scaling maintains ratios');
  test('cooking method adjustments reasonable');
});
```

### Edge Case Tests

#### Classification Edge Cases
```python
def test_multiple_foods_in_image():
    """Should return primary classification or multi-food flag"""
    
def test_partially_eaten_food():
    """Should still classify correctly"""
    
def test_food_in_packaging():
    """Should suggest barcode scanning"""
    
def test_non_food_image():
    """Should return low confidence / 'unknown'"""
    
def test_blurry_image():
    """Should request better image or proceed with caution"""
```

#### Search Edge Cases
```typescript
describe('Search Edge Cases', () => {
  test('handles Unicode characters (日本語, émoji)');
  test('handles very long queries (>200 chars)');
  test('handles SQL injection attempts');
  test('handles empty/whitespace-only queries');
  test('handles queries with only special characters');
});
```

#### API Failure Edge Cases
```typescript
describe('API Resilience', () => {
  test('handles USDA API timeout');
  test('handles USDA API 500 errors');
  test('handles rate limit exceeded (429)');
  test('handles malformed USDA response');
  test('circuit breaker activates after repeated failures');
  test('fallback to cached data when API unavailable');
});
```

### Feedback Loop Tests

```typescript
describe('Feedback Collection', () => {
  test('feedback submitted successfully');
  test('feedback deduplication works');
  test('feedback aggregation triggers retraining signal');
  test('feedback privacy (no PII stored)');
});
```

### Load Tests

```typescript
describe('Load Testing', () => {
  test('100 concurrent searches complete in <5s');
  test('1000 searches/minute sustained without errors');
  test('cache hit ratio >80% after warmup');
  test('memory usage stable under load');
});
```

### Acceptance Criteria

#### Core Functionality
- [ ] Search returns relevant results for common foods
- [ ] Classification correctly categorizes >90% of test images
- [ ] Nutrition data matches USDA source within 5%
- [ ] Performance meets latency targets (search <500ms, classify <100ms)

#### User Experience
- [ ] User selects from top-5 suggestions >80% of time
- [ ] Search autocomplete feels responsive (<300ms)
- [ ] Classification confidence displayed helpfully
- [ ] Graceful degradation when services unavailable

#### System Reliability
- [ ] Circuit breaker prevents cascade failures
- [ ] Rate limiting prevents USDA API abuse
- [ ] Caching reduces API calls by >80%
- [ ] Offline mode provides useful functionality

#### Data Quality
- [ ] Nutrient mapping covers all essential nutrients
- [ ] Serving size conversions accurate
- [ ] Cooking method adjustments reasonable
- [ ] Feedback loop improves suggestions over time
