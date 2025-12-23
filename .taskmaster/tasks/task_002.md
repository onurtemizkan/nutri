# Task ID: 2

**Title:** Implement Real Food Classification ML Model

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Replace the mock food classifier in ml-service/app/services/food_analysis_service.py with a real CNN model (EfficientNet or ResNet) trained on food image datasets.

**Details:**

1. Create a new module `ml-service/app/ml_models/food_classifier.py` with:
   - Load pre-trained EfficientNet-B0 or ResNet-50 from torchvision
   - Fine-tune on Food-101 dataset or custom food dataset
   - Implement proper image preprocessing pipeline matching ImageNet stats
   - Support for GPU inference if available (auto-detect CUDA)

2. Update `food_analysis_service.py`:
   - Replace `NUTRITION_DATABASE` with a proper food database (USDA FoodData Central API integration)
   - Update `_classify_food()` to use real model inference instead of random selection
   - Add model loading with caching to avoid reloading on each request
   - Implement top-5 predictions with confidence scores

3. Add model versioning:
   - Store model checkpoints in `ml-service/models/food_classifier/`
   - Add `model_version` field to responses
   - Implement A/B testing capability by loading multiple model versions

4. Extend nutrition database:
   - Expand from current 6 items to 100+ common foods
   - Structure: JSON file or SQLite database with USDA data
   - Include serving size variations (small, medium, large)

Pseudo-code for classifier:
```python
class FoodClassifier:
    def __init__(self, model_path: str = None):
        self.model = self._load_model(model_path)
        self.classes = self._load_class_labels()
    
    def _load_model(self, path):
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier[-1] = nn.Linear(1280, num_food_classes)
        if path:
            model.load_state_dict(torch.load(path))
        model.eval()
        return model
    
    async def classify(self, image: np.ndarray) -> List[Tuple[str, float]]:
        tensor = self._preprocess(image)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
        return self._get_top_k(probs, k=5)
```

**Test Strategy:**

1. Unit tests for model loading and inference
2. Test classification accuracy on held-out test set (target >80% top-5)
3. Integration test: POST /api/food/analyze with real food images
4. Performance test: Inference time <3s per image
5. Test model fallback when GPU not available
