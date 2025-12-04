# Food Analysis Architecture - ARKit + Camera + ML

## Overview

Integrate AR-powered food scanning to automatically estimate portion sizes, classify ingredients, and calculate nutritional values using computer vision and machine learning.

## System Architecture

### High-Level Flow

```
User → Camera Screen → ARKit Measurement → Capture Photo → ML Analysis → Review Results → Save Meal
```

## Components

### 1. Mobile App (React Native + Expo)

#### New Screens & Components

**A. Food Scanner Screen** (`app/scan-food.tsx`)
- Camera interface with ARKit integration
- Real-time AR surface detection and measurement
- Photo capture and preview
- Results display with nutrition estimates

**B. AR Measurement Module** (Native Module)
- iOS: ARKit plane detection and distance measurement
- Calculate real-world dimensions of food items
- Visual feedback (3D overlay, measurement guides)

#### Required Dependencies

```json
{
  "expo-camera": "~14.1.0",
  "expo-gl": "~14.0.0",
  "expo-dev-client": "~4.0.0",
  "@react-native-community/netinfo": "^11.0.0",
  "expo-image-manipulator": "~12.0.0",
  "expo-file-system": "~17.0.0"
}
```

#### Configuration Changes

**app.json:**
```json
{
  "expo": {
    "ios": {
      "infoPlist": {
        "NSCameraUsageDescription": "We need camera access to scan and analyze your food.",
        "NSPhotoLibraryUsageDescription": "We need photo library access to save food images.",
        "NSMicrophoneUsageDescription": "Microphone access is required for video recording (optional)."
      },
      "bitcode": false
    },
    "plugins": [
      [
        "expo-camera",
        {
          "cameraPermission": "Allow Nutri to access your camera to scan food."
        }
      ]
    ]
  }
}
```

#### User Flow

1. **Entry Point**: User taps "Scan Food" button on Add Meal screen
2. **Camera + AR Mode**:
   - App requests camera permission
   - ARKit detects horizontal planes (table, counter)
   - User positions food on detected surface
   - AR overlay shows measurement guides
3. **Measurement**:
   - ARKit calculates distance from camera to food
   - Estimates food dimensions using reference points
   - Visual confirmation (3D bounding box overlay)
4. **Capture**:
   - User taps capture button
   - Photo captured with AR measurements
5. **Analysis**:
   - Loading indicator shown
   - Photo + measurements sent to ML service
   - ML service returns food classification + nutrition
6. **Review**:
   - Display photo with detected items
   - Show nutrition estimates (calories, protein, carbs, fat)
   - Confidence scores displayed
   - User can edit values
7. **Save**:
   - Pre-fill Add Meal form
   - User confirms and saves

### 2. ML Service (Python + FastAPI)

#### New API Endpoints

**File**: `ml-service/app/api/food_analysis.py`

```python
POST /api/food/analyze
  Input: {
    image: File (multipart/form-data),
    dimensions: { width: float, height: float, depth: float } (cm),
    confidence: float (optional)
  }
  Output: {
    food_items: [{
      name: string,
      confidence: float,
      portion_size: string,
      nutrition: {
        calories: float,
        protein: float,
        carbs: float,
        fat: float,
        fiber: float
      }
    }],
    measurement_quality: "high" | "medium" | "low"
  }

GET /api/food/models/info
  Output: { available_models: [], active_model: string }

GET /api/food/nutrition-db/search?q={food_name}
  Output: { results: [...nutrition data...] }
```

#### Food Analysis Service

**File**: `ml-service/app/services/food_analysis_service.py`

**Responsibilities**:
1. Image preprocessing (resize, normalize, augment)
2. Food classification using pre-trained CNN
3. Portion size estimation from AR measurements
4. Nutrition calculation based on food type + portion size
5. Confidence scoring and fallback handling

**ML Models**:

1. **Food Classifier** (Transfer Learning):
   - Base: ResNet-50 or EfficientNet-B0 (pre-trained on ImageNet)
   - Fine-tuned on Food-101 dataset (101 food categories)
   - Input: 224x224 RGB image
   - Output: Food class probabilities

2. **Portion Size Estimator**:
   - Uses AR measurements (width, height, depth)
   - Maps dimensions to standard serving sizes
   - Accounts for food density (fluffy vs dense)
   - Reference database for common portion sizes

3. **Nutrition Database**:
   - USDA FoodData Central API integration
   - Local cache of common foods
   - Portion size scaling logic
   - Macro + micronutrient calculations

#### Implementation Strategy

**Phase 1 - MVP (Start Here)**:
- Single food item detection
- Pre-trained food classifier (Food-101)
- Basic portion size estimation
- Static nutrition database (top 100 common foods)

**Phase 2 - Enhanced**:
- Multi-food detection (YOLO/Faster R-CNN)
- Improved portion estimation using depth data
- Real-time USDA API integration
- Ingredient breakdown for complex meals

**Phase 3 - Advanced**:
- User-specific portion size learning
- Meal context (breakfast → likely eggs/toast)
- Historical meal patterns
- Barcode scanning integration

### 3. Native ARKit Module (iOS)

#### Implementation Options

**Option A: Custom Native Module** (Recommended for full ARKit features)
- Create Swift module using Expo Modules API
- Full access to ARKit capabilities
- Best measurement accuracy

**Option B: expo-gl + react-native-arkit**
- Use existing libraries
- Faster setup but limited features
- Good for basic AR

**For MVP, use Option B; migrate to Option A if needed.**

#### ARKit Features Used

1. **Plane Detection** (`ARPlaneAnchor`):
   - Detect horizontal surfaces (tables)
   - Provide visual feedback
   - Ensure food is on stable surface

2. **Distance Measurement** (`ARHitTestResult`):
   - Calculate distance from camera to food
   - Use for size estimation
   - Validate measurement quality

3. **Object Tracking** (Optional):
   - Track food position during capture
   - Ensure photo quality
   - Reduce motion blur

## Data Models

### Mobile App Types

```typescript
// lib/types/food-analysis.ts

export interface ARMeasurement {
  width: number;  // cm
  height: number; // cm
  depth: number;  // cm
  distance: number; // cm from camera
  confidence: 'high' | 'medium' | 'low';
  planeDetected: boolean;
}

export interface FoodAnalysisRequest {
  imageUri: string;
  measurements?: ARMeasurement;
}

export interface FoodItem {
  name: string;
  confidence: number; // 0-1
  portionSize: string;
  nutrition: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber?: number;
  };
}

export interface FoodAnalysisResponse {
  foodItems: FoodItem[];
  measurementQuality: 'high' | 'medium' | 'low';
  processingTime: number; // ms
  suggestions?: string[];
}
```

### ML Service Models

```python
# ml-service/app/schemas/food_analysis.py

class DimensionsInput(BaseModel):
    width: float  # cm
    height: float  # cm
    depth: float  # cm

class FoodAnalysisRequest(BaseModel):
    dimensions: Optional[DimensionsInput]
    confidence: Optional[float]

class NutritionInfo(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: Optional[float]

class FoodItem(BaseModel):
    name: str
    confidence: float
    portion_size: str
    nutrition: NutritionInfo

class FoodAnalysisResponse(BaseModel):
    food_items: List[FoodItem]
    measurement_quality: str
    processing_time: float
```

## ML Model Details

### Food Classifier

**Architecture**:
```python
import torchvision.models as models
import torch.nn as nn

class FoodClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        # Use pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
```

**Training Dataset**: Food-101 (101 food categories, 101,000 images)
**Performance Target**: >80% top-1 accuracy, >95% top-5 accuracy

### Portion Size Estimation

**Algorithm**:
```python
def estimate_portion_size(dimensions: Dimensions, food_class: str) -> float:
    """
    Estimate portion size in grams/ml based on AR measurements.

    Args:
        dimensions: Width, height, depth in cm
        food_class: Classified food category

    Returns:
        Estimated weight/volume in standard units
    """
    # Calculate volume
    volume_cm3 = dimensions.width * dimensions.height * dimensions.depth

    # Get food density from database
    density = FOOD_DENSITY_DB.get(food_class, 0.8)  # g/cm³

    # Apply shape correction factor
    shape_factor = FOOD_SHAPE_FACTORS.get(food_class, 0.7)

    # Calculate weight
    weight_grams = volume_cm3 * density * shape_factor

    return weight_grams
```

**Nutrition Scaling**:
```python
def scale_nutrition(base_nutrition: dict, base_portion: float,
                   actual_portion: float) -> dict:
    """Scale nutrition values based on portion size."""
    scale_factor = actual_portion / base_portion

    return {
        nutrient: value * scale_factor
        for nutrient, value in base_nutrition.items()
    }
```

## Security & Privacy

### Image Handling

1. **Upload Security**:
   - File size limit: 10MB
   - Allowed formats: JPEG, PNG
   - Image validation before processing
   - Automatic compression if needed

2. **Storage**:
   - Images NOT stored by default
   - Optional: User can save to meal history
   - Stored images encrypted at rest
   - Auto-delete after 30 days

3. **Privacy**:
   - All processing done on secure server
   - No image sharing with third parties
   - GDPR compliant data handling
   - User can delete images anytime

## Performance Considerations

### Mobile App

1. **Camera Performance**:
   - 30 FPS minimum
   - ARKit plane detection: ~1-3 seconds
   - Photo capture: <500ms

2. **Network**:
   - Image compression before upload
   - Retry logic with exponential backoff
   - Offline mode: Save photo, analyze later
   - Timeout: 30 seconds for analysis

### ML Service

1. **Response Time Targets**:
   - P50: <2 seconds
   - P95: <5 seconds
   - P99: <10 seconds

2. **Optimization**:
   - Model caching in memory
   - GPU acceleration (if available)
   - Batch processing support
   - Redis caching for common foods

3. **Scaling**:
   - Horizontal scaling with load balancer
   - Separate worker processes for ML inference
   - Queue system (Celery) for heavy operations

## Testing Strategy

### Unit Tests

- ARKit module: Measurement accuracy
- Food classifier: Model inference
- Portion estimator: Size calculations
- Nutrition calculator: Scaling logic

### Integration Tests

- End-to-end flow: Camera → ML → Save
- API endpoint testing
- Error handling and fallbacks

## Future Enhancements

1. **Multi-Food Detection**: Detect multiple items in one photo
2. **Ingredient Breakdown**: Show individual ingredients in complex meals
3. **Meal Templates**: Save and reuse common meals
4. **Social Features**: Share meals, compare with friends
5. **Barcode Scanning**: Quick lookup for packaged foods
6. **Voice Input**: "Log a chicken salad" → Auto-scan
7. **Smart Reminders**: "Time for lunch!" based on patterns
8. **Integration**: Apple Health, Google Fit, Fitbit, etc.

## Success Metrics

### MVP Success Criteria

- [ ] 70%+ food classification accuracy
- [ ] 80%+ user satisfaction with nutrition estimates
- [ ] <5 second analysis time (P95)
- [ ] 50%+ of users use scan feature regularly

### KPIs to Track

- Scan-to-save conversion rate
- Classification accuracy by food type
- User edit frequency (lower = better estimates)
- Feature adoption rate
- Processing time distribution

## Development Timeline

### Week 1-2: Foundation
- Set up camera permissions
- Basic camera UI
- Image upload to ML service
- Static nutrition database

### Week 3-4: ML Integration
- Food classifier training
- Portion size estimation
- API endpoint implementation
- Basic nutrition calculation

### Week 5-6: ARKit Integration
- Plane detection
- Distance measurement
- Size estimation UI
- Integration with ML service

### Week 7-8: Polish & Testing
- UI/UX refinement
- Integration testing
- Performance optimization
- Bug fixes and edge cases

---

**Total Estimated Development Time**: 6-8 weeks
**MVP Target**: 3-4 weeks (without advanced ARKit features)
