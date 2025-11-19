# ✅ Phase 2 Complete: PyTorch LSTM Prediction Models

**Status**: ✅ **COMPLETE**
**Date**: 2025-01-15
**Technology**: **PyTorch 2.1.2** (Deep Learning Framework)

---

## 📊 Phase 2 Overview

Phase 2 implemented a complete **PyTorch LSTM-based prediction system** for health metrics (Resting Heart Rate, Heart Rate Variability, Sleep Duration, etc.). This system uses deep learning to forecast health metrics based on historical nutrition, activity, and health data.

### Key Capabilities

✅ **Train Custom LSTM Models**
- PyTorch neural networks with configurable architecture
- Sequence-based training (30-day windows → predict day 31)
- Early stopping and learning rate scheduling
- Model checkpointing and versioning

✅ **Make Predictions**
- Load trained models from disk
- Prepare input sequences from features
- Generate predictions with confidence intervals
- Natural language interpretation

✅ **Model Evaluation**
- Comprehensive metrics (MAE, RMSE, R², MAPE)
- Validation set evaluation
- Production readiness assessment

✅ **Model Management**
- Save/load model artifacts (model, scalers, config)
- Version control
- Model registry

---

## 🗂️ Files Created in Phase 2

### 1. Pydantic Schemas

#### `app/schemas/predictions.py` (370 lines)

**Purpose**: Type-safe data models for predictions and training

**Key Models**:

```python
# Enums
class PredictionMetric(str, Enum):
    RHR = "RESTING_HEART_RATE"
    HRV_SDNN = "HEART_RATE_VARIABILITY_SDNN"
    HRV_RMSSD = "HEART_RATE_VARIABILITY_RMSSD"
    SLEEP_DURATION = "SLEEP_DURATION"
    RECOVERY_SCORE = "RECOVERY_SCORE"

class ModelArchitecture(str, Enum):
    LSTM = "lstm"
    XGBOOST = "xgboost"
    LINEAR = "linear"

# Training
class TrainModelRequest(BaseModel):
    user_id: str
    metric: PredictionMetric
    architecture: ModelArchitecture = ModelArchitecture.LSTM
    lookback_days: int = 90
    sequence_length: int = 30
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2

class TrainModelResponse(BaseModel):
    model_id: str
    model_version: str
    trained_at: datetime
    training_metrics: TrainingMetrics
    total_samples: int
    num_features: int
    is_production_ready: bool

# Prediction
class PredictRequest(BaseModel):
    user_id: str
    metric: PredictionMetric
    target_date: date

class PredictionResult(BaseModel):
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_score: float  # 0-1
    historical_average: float
    deviation_from_average: float
    percentile: float  # 0-100

class PredictResponse(BaseModel):
    prediction: PredictionResult
    interpretation: str
    recommendation: Optional[str]
    cached: bool

# Batch prediction
class BatchPredictRequest(BaseModel):
    user_id: str
    metrics: List[PredictionMetric]
    target_date: date

class BatchPredictResponse(BaseModel):
    predictions: Dict[str, PredictionResult]
    all_predictions_successful: bool
    failed_metrics: List[str]
```

---

### 2. PyTorch LSTM Models

#### `app/ml_models/lstm.py` (294 lines)

**Purpose**: PyTorch LSTM neural network architectures

**Key Components**:

```python
@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    sequence_length: int = 30
    output_dim: int = 1
    device: str = "cpu"

class HealthMetricLSTM(nn.Module):
    """
    LSTM neural network for health metric prediction.

    Architecture:
    1. Input: (batch, sequence_length, input_dim)
    2. LSTM Layers: Stacked LSTM with dropout
    3. Fully Connected: Dense layers with ReLU + BatchNorm
    4. Output: Single value (predicted metric)
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, config.hidden_dim // 2)
        self.fc2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4)
        self.fc_out = nn.Linear(config.hidden_dim // 4, 1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(config.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim // 4)

        self._init_weights()  # Xavier initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last timestep

        # Fully connected layers with dropout and batch norm
        out = self.fc1(self.dropout(last_output))
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        out = self.fc_out(out)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

class MultiTaskLSTM(nn.Module):
    """
    Multi-task LSTM for predicting multiple metrics simultaneously.
    Useful for predicting RHR + HRV together (shared patterns).
    """
    # Shared LSTM encoder + separate prediction heads

class LSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism (Phase 3 - interpretability).
    Helps identify which days are most important for prediction.
    """
    # TODO: Phase 3 implementation
```

**Key Features**:
- ✅ Configurable architecture (hidden dim, layers, dropout)
- ✅ Batch normalization for training stability
- ✅ Xavier/Glorot weight initialization
- ✅ Bidirectional LSTM support
- ✅ Multi-task learning (predict multiple metrics)
- ⏳ Attention mechanism (planned for Phase 3)

---

#### `app/ml_models/baseline.py` (114 lines)

**Purpose**: Baseline models for comparison

**Models**:

```python
class BaselineLinearModel(nn.Module):
    """Simple linear regression baseline."""
    # Takes average of sequence, applies linear layer
    # LSTM should outperform this

class MovingAverageBaseline:
    """Moving average baseline (non-neural)."""
    # Predicts tomorrow = average of last N days
    # Simplest possible baseline

class ExponentialSmoothingBaseline:
    """Exponential smoothing baseline."""
    # Gives more weight to recent values
    # alpha=0.3 is good default
```

**Why Baselines?**
- Sanity check: LSTM should outperform simple baselines
- Performance comparison: "How much better is LSTM?"
- Cost-benefit analysis: "Is deep learning worth the complexity?"

---

### 3. Data Preparation Service

#### `app/services/data_preparation.py` (497 lines)

**Purpose**: Prepare time-series data for LSTM training

**Key Methods**:

```python
class DataPreparationService:
    """Service for preparing LSTM training/prediction data."""

    async def prepare_training_data(
        self,
        user_id: str,
        target_metric: PredictionMetric,
        lookback_days: int = 90,
        sequence_length: int = 30,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Prepare complete dataset for LSTM training.

        Steps:
        1. Build feature matrix (51 features per day)
        2. Fetch target metric values
        3. Align features and targets (remove missing days)
        4. Create sequences (sliding windows)
        5. Train/val split
        6. Normalize features (StandardScaler)
        7. Normalize labels (StandardScaler)
        8. Convert to PyTorch tensors

        Returns:
            X_train: (batch, seq_len, num_features)
            y_train: (batch,)
            X_val: (batch, seq_len, num_features)
            y_val: (batch,)
            scaler: Fitted StandardScaler
            label_scaler: Fitted StandardScaler
            feature_names: List[str]
        """

    async def prepare_prediction_input(
        self,
        user_id: str,
        target_date: date,
        sequence_length: int,
        scaler: StandardScaler,
        feature_names: List[str],
    ) -> torch.Tensor:
        """
        Prepare input for making a prediction.

        Returns:
            PyTorch tensor of shape (1, sequence_length, num_features)
        """

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM.

        Example:
            90 days of data, sequence_length=30:
            - Days 0-29 → predict day 30
            - Days 1-30 → predict day 31
            - Days 2-31 → predict day 32
            - ...
            - Days 59-89 → predict day 90
            Result: 60 sequences
        """

    def _normalize_features(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler.
        Fits on training data only, transforms both train and val.
        """
```

**Key Concepts**:
- **Sliding Windows**: Create overlapping sequences for training
- **Normalization**: StandardScaler (mean=0, std=1) for better convergence
- **Separate Scalers**: One for features, one for labels
- **Feature Matrix**: Uses FeatureEngineeringService (Phase 1)

---

### 4. Model Training Service

#### `app/services/model_training.py` (500+ lines)

**Purpose**: Train PyTorch LSTM models with full training loop

**Key Methods**:

```python
class ModelTrainingService:
    """Service for training PyTorch LSTM models."""

    async def train_model(
        self, request: TrainModelRequest
    ) -> TrainModelResponse:
        """
        Train a PyTorch LSTM model.

        Steps:
        1. Prepare training data (DataPreparationService)
        2. Initialize LSTM model
        3. Train with early stopping
        4. Evaluate on validation set
        5. Save model artifacts
        6. Assess production readiness

        Returns:
            TrainModelResponse with metrics and model path
        """

    def _train_loop(
        self,
        model: nn.Module,
        X_train, y_train, X_val, y_val,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> Dict:
        """
        PyTorch training loop with early stopping.

        Features:
        - Mini-batch training
        - MSE loss (regression)
        - Adam optimizer
        - Learning rate scheduling (ReduceLROnPlateau)
        - Early stopping (patience=10)
        - Validation monitoring
        """

    def _evaluate_model(
        self,
        model: nn.Module,
        X_val, y_val,
        label_scaler,
        device: torch.device,
    ) -> Dict:
        """
        Evaluate model with comprehensive metrics.

        Metrics:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)
        """

    def _save_model_artifacts(
        self,
        model_id: str,
        model: nn.Module,
        scaler, label_scaler,
        config: LSTMConfig,
        feature_names: List[str],
        request: TrainModelRequest,
    ) -> Dict:
        """
        Save all artifacts needed for inference.

        Saves:
        - model.pt (PyTorch state dict)
        - scaler.pkl (feature StandardScaler)
        - label_scaler.pkl (label StandardScaler)
        - config.pkl (LSTMConfig)
        - feature_names.pkl (list of feature names)
        - metadata.pkl (training metadata)
        """

    def _assess_model_quality(
        self, eval_metrics: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Assess if model is production-ready.

        Quality thresholds:
        - R² > 0.5 (explains >50% variance)
        - MAPE < 15% (predictions within 15%)
        """
```

**Training Features**:
- ✅ Mini-batch training (configurable batch size)
- ✅ Early stopping (patience=10 epochs)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Validation monitoring (train vs val loss)
- ✅ Model checkpointing
- ✅ Production readiness assessment

---

### 5. Prediction Service

#### `app/services/prediction.py` (600+ lines)

**Purpose**: Load trained models and make predictions

**Key Methods**:

```python
class PredictionService:
    """Service for making predictions using trained LSTM models."""

    async def predict(
        self, request: PredictRequest
    ) -> PredictResponse:
        """
        Make a prediction for a health metric.

        Steps:
        1. Check Redis cache
        2. Load trained model artifacts
        3. Prepare input features (last 30 days)
        4. Run LSTM inference
        5. Denormalize prediction
        6. Calculate confidence interval
        7. Get historical context
        8. Generate interpretation
        9. Cache result (24 hours)

        Returns:
            PredictResponse with prediction and interpretation
        """

    def _load_model_artifacts(self, model_id: str) -> Dict:
        """
        Load all model artifacts from disk.

        Loads:
        - PyTorch model (state dict)
        - Feature scaler
        - Label scaler
        - Model config
        - Feature names
        - Metadata
        """

    def _calculate_confidence_interval(
        self, predicted_value: float, metadata: Dict
    ) -> Dict[str, float]:
        """
        Calculate 95% confidence interval.
        Uses 1.96 * MAE as approximation.
        """

    def _calculate_confidence_score(
        self,
        predicted_value: float,
        historical_stats: Dict,
        metadata: Dict,
    ) -> float:
        """
        Calculate confidence score (0-1).

        Factors:
        - Model R² (higher = more confident)
        - In-distribution (within historical range)
        - Data quality
        """

    async def _get_historical_stats(
        self, user_id: str, metric: PredictionMetric
    ) -> Dict:
        """
        Get 30-day historical statistics.
        Returns: avg, min, max, std, values
        """

    def _generate_interpretation(
        self,
        metric: PredictionMetric,
        predicted_value: float,
        historical_average: float,
        confidence_score: float,
    ) -> str:
        """
        Generate natural language interpretation.

        Example:
        "Your predicted Resting Heart Rate is 62.5, which is 3.8%
        higher than your 30-day average of 60.2. This prediction
        has high confidence."
        """

    def _generate_recommendation(
        self,
        metric: PredictionMetric,
        predicted_value: float,
        historical_average: float,
    ) -> Optional[str]:
        """
        Generate actionable recommendation.

        Example (RHR elevated):
        "Your resting heart rate may be elevated tomorrow. Consider
        lighter training, prioritizing recovery, and ensuring adequate
        hydration."
        """
```

**Prediction Features**:
- ✅ Model loading from disk
- ✅ Confidence intervals (95% CI using MAE)
- ✅ Confidence scores (0-1 based on R², distribution, quality)
- ✅ Historical context (30-day average, percentile)
- ✅ Natural language interpretation
- ✅ Actionable recommendations
- ✅ Redis caching (24-hour TTL)

---

### 6. Prediction API Routes

#### `app/api/predictions.py` (400+ lines)

**Purpose**: RESTful API endpoints for training and prediction

**Endpoints Created**:

```python
# Training
POST /api/predictions/train
    → Train a new LSTM model
    Request: TrainModelRequest
    Response: TrainModelResponse

# Prediction
POST /api/predictions/predict
    → Make a single prediction
    Request: PredictRequest
    Response: PredictResponse

GET /api/predictions/predict/{user_id}/{metric}/{target_date}
    → Convenience GET endpoint
    Response: PredictResponse

# Batch Prediction
POST /api/predictions/batch-predict
    → Predict multiple metrics at once
    Request: BatchPredictRequest
    Response: BatchPredictResponse

# Model Management
GET /api/predictions/models/{user_id}
    → List user's trained models
    Query params: ?metric=RESTING_HEART_RATE
    Response: ListModelsResponse

DELETE /api/predictions/models/{model_id}
    → Delete a trained model
    Response: {message: "deleted", model_id: "..."}
```

---

## 🔧 Updated Files

### `app/api/__init__.py`

```python
from .predictions import router as predictions_router

api_router.include_router(
    predictions_router,
    prefix="/predictions",
    tags=["predictions"]
)
```

### `app/ml_models/__init__.py`

```python
from .lstm import HealthMetricLSTM, LSTMConfig, MultiTaskLSTM
from .baseline import (
    BaselineLinearModel,
    MovingAverageBaseline,
    ExponentialSmoothingBaseline,
)
```

---

## 📊 Complete API Surface

### Phase 1 + Phase 2 Endpoints

```
Feature Engineering:
  POST /api/features/engineer
  GET /api/features/{user_id}/{target_date}
  GET /api/features/{user_id}/{target_date}/summary
  DELETE /api/features/{user_id}/cache

Correlation Analysis:
  POST /api/correlations/analyze
  POST /api/correlations/lag-analysis
  GET /api/correlations/{user_id}/{target_metric}/summary

Predictions (NEW):
  POST /api/predictions/train
  POST /api/predictions/predict
  POST /api/predictions/batch-predict
  GET /api/predictions/predict/{user_id}/{metric}/{target_date}
  GET /api/predictions/models/{user_id}
  DELETE /api/predictions/models/{model_id}
```

---

## 🎯 Example Usage

### 1. Train a Model

```bash
curl -X POST http://localhost:8001/api/predictions/train \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "metric": "RESTING_HEART_RATE",
    "architecture": "lstm",
    "lookback_days": 90,
    "sequence_length": 30,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2
  }'
```

**Response:**
```json
{
  "user_id": "user_123",
  "metric": "RESTING_HEART_RATE",
  "architecture": "lstm",
  "model_id": "user_123_RESTING_HEART_RATE_20250115_103045",
  "model_version": "v1.0.0",
  "trained_at": "2025-01-15T10:30:45Z",
  "training_metrics": {
    "train_loss": 0.023,
    "val_loss": 0.031,
    "best_val_loss": 0.029,
    "mae": 2.1,
    "rmse": 2.8,
    "r2_score": 0.73,
    "mape": 3.5,
    "epochs_trained": 42,
    "early_stopped": true,
    "training_time_seconds": 145.2
  },
  "total_samples": 60,
  "sequence_length": 30,
  "num_features": 51,
  "model_path": "models/user_123_RESTING_HEART_RATE_20250115_103045/model.pt",
  "model_size_mb": 2.3,
  "is_production_ready": true,
  "quality_issues": []
}
```

---

### 2. Make a Prediction

```bash
curl -X POST http://localhost:8001/api/predictions/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "metric": "RESTING_HEART_RATE",
    "target_date": "2025-01-16"
  }'
```

**Response:**
```json
{
  "user_id": "user_123",
  "prediction": {
    "metric": "RESTING_HEART_RATE",
    "target_date": "2025-01-16",
    "predicted_at": "2025-01-15T14:30:00Z",
    "predicted_value": 62.5,
    "confidence_interval_lower": 58.3,
    "confidence_interval_upper": 66.7,
    "confidence_score": 0.87,
    "historical_average": 60.2,
    "deviation_from_average": 2.3,
    "percentile": 65.0,
    "model_id": "user_123_RESTING_HEART_RATE_20250115_103045",
    "model_version": "v1.0.0",
    "architecture": "lstm"
  },
  "features_used": 51,
  "sequence_length": 30,
  "data_quality_score": 0.85,
  "interpretation": "Your predicted Resting Heart Rate is 62.5, which is 3.8% higher than your 30-day average of 60.2. This prediction has high confidence.",
  "recommendation": "Your resting heart rate may be elevated tomorrow. Consider lighter training, prioritizing recovery, and ensuring adequate hydration.",
  "cached": false
}
```

---

### 3. Batch Prediction (Multiple Metrics)

```bash
curl -X POST http://localhost:8001/api/predictions/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "metrics": [
      "RESTING_HEART_RATE",
      "HEART_RATE_VARIABILITY_SDNN",
      "SLEEP_DURATION"
    ],
    "target_date": "2025-01-16"
  }'
```

**Response:**
```json
{
  "user_id": "user_123",
  "target_date": "2025-01-16",
  "predicted_at": "2025-01-15T14:30:00Z",
  "predictions": {
    "RESTING_HEART_RATE": { /* PredictionResult */ },
    "HEART_RATE_VARIABILITY_SDNN": { /* PredictionResult */ },
    "SLEEP_DURATION": { /* PredictionResult */ }
  },
  "overall_data_quality": 0.85,
  "all_predictions_successful": true,
  "failed_metrics": []
}
```

---

### 4. List Models

```bash
curl http://localhost:8001/api/predictions/models/user_123?metric=RESTING_HEART_RATE
```

**Response:**
```json
{
  "user_id": "user_123",
  "models": [
    {
      "model_id": "user_123_RESTING_HEART_RATE_20250115_103045",
      "user_id": "user_123",
      "metric": "RESTING_HEART_RATE",
      "architecture": "lstm",
      "version": "v1.0.0",
      "trained_at": "2025-01-15T10:30:45Z",
      "training_metrics": { /* ... */ },
      "sequence_length": 30,
      "num_features": 51,
      "model_size_mb": 2.3,
      "is_active": true,
      "is_production_ready": true
    }
  ],
  "total_models": 1
}
```

---

## 🧠 Model Architecture Details

### LSTM Configuration

Default hyperparameters (configurable via API):

```python
{
    "hidden_dim": 128,        # LSTM hidden state dimension
    "num_layers": 2,          # Number of stacked LSTM layers
    "dropout": 0.2,           # Dropout rate (20%)
    "sequence_length": 30,    # Input sequence length (days)
    "lookback_days": 90,      # Training data window
    "epochs": 50,             # Maximum training epochs
    "batch_size": 32,         # Mini-batch size
    "learning_rate": 0.001,   # Adam optimizer learning rate
}
```

### Network Architecture

```
Input: (batch, 30 days, 51 features)
    ↓
LSTM Layer 1: 51 → 128 hidden units
    ↓
Dropout: 20%
    ↓
LSTM Layer 2: 128 → 128 hidden units
    ↓
Dropout: 20%
    ↓
Take last timestep output: (batch, 128)
    ↓
FC1: 128 → 64
    ↓
BatchNorm1d: 64
    ↓
ReLU activation
    ↓
Dropout: 20%
    ↓
FC2: 64 → 32
    ↓
BatchNorm1d: 32
    ↓
ReLU activation
    ↓
Dropout: 20%
    ↓
FC_out: 32 → 1
    ↓
Output: Predicted value (scalar)
```

**Total Parameters**: ~100K (varies by configuration)

---

## 📈 Model Evaluation Metrics

### Metrics Computed

1. **MAE (Mean Absolute Error)**
   - Average absolute difference between predictions and actuals
   - Example: MAE=2.1 means predictions off by 2.1 BPM on average
   - Lower is better

2. **RMSE (Root Mean Squared Error)**
   - Square root of average squared errors
   - Penalizes large errors more than MAE
   - Example: RMSE=2.8 BPM
   - Lower is better

3. **R² Score (Coefficient of Determination)**
   - Proportion of variance explained by model
   - Range: 0 to 1 (negative if worse than mean)
   - Example: R²=0.73 means model explains 73% of variance
   - Higher is better
   - **Production threshold: R² > 0.5**

4. **MAPE (Mean Absolute Percentage Error)**
   - Average percentage error
   - Example: MAPE=3.5% means predictions off by 3.5% on average
   - Lower is better
   - **Production threshold: MAPE < 15%**

### Production Readiness

Models are marked as **production-ready** if:
- ✅ R² > 0.5 (explains >50% variance)
- ✅ MAPE < 15% (predictions within 15%)
- ✅ R² >= 0 (better than mean baseline)

If not production-ready, quality issues are listed (e.g., "Low R² score", "High prediction error").

---

## 💾 Model Persistence

### Model Artifacts

Each trained model saves:

```
models/
└── user_123_RESTING_HEART_RATE_20250115_103045/
    ├── model.pt              # PyTorch state dict
    ├── scaler.pkl            # Feature StandardScaler
    ├── label_scaler.pkl      # Label StandardScaler
    ├── config.pkl            # LSTMConfig
    ├── feature_names.pkl     # List of 51 feature names
    └── metadata.pkl          # Training metadata
```

### Model ID Format

```
{user_id}_{metric}_{timestamp}

Examples:
- user_123_RESTING_HEART_RATE_20250115_103045
- user_456_HEART_RATE_VARIABILITY_SDNN_20250115_143022
```

---

## 🔄 Training Pipeline

### Step-by-Step Process

```
1. Data Fetching
   ├─ Fetch nutrition data (meals)
   ├─ Fetch activity data (workouts)
   ├─ Fetch health metrics (RHR, HRV)
   └─ Lookback: 90 days

2. Feature Engineering
   ├─ Engineer 51 features per day
   ├─ Nutrition features (16)
   ├─ Activity features (12)
   ├─ Health features (12)
   ├─ Temporal features (5)
   └─ Interaction features (6)

3. Sequence Creation
   ├─ Sliding windows (30 days → predict day 31)
   ├─ Example: Days 0-29 → predict day 30
   ├─ Result: 60 sequences (from 90 days)
   └─ Output: X (60, 30, 51), y (60,)

4. Train/Val Split
   ├─ 80% training (48 sequences)
   ├─ 20% validation (12 sequences)
   └─ Chronological split (no shuffling)

5. Normalization
   ├─ StandardScaler on features (fit on train only)
   ├─ StandardScaler on labels (fit on train only)
   └─ Transform both train and val

6. PyTorch Conversion
   ├─ Convert to torch.FloatTensor
   └─ Move to device (CPU or CUDA)

7. Model Training
   ├─ Initialize LSTM model
   ├─ Adam optimizer (lr=0.001)
   ├─ MSE loss function
   ├─ Mini-batch training (batch_size=32)
   ├─ Early stopping (patience=10)
   ├─ Learning rate scheduling
   └─ Validation monitoring

8. Evaluation
   ├─ Compute MAE, RMSE, R², MAPE
   ├─ On denormalized predictions
   └─ Assess production readiness

9. Model Saving
   ├─ Save PyTorch state dict
   ├─ Save scalers (feature, label)
   ├─ Save config and metadata
   └─ Return model ID
```

---

## 🔮 Prediction Pipeline

### Step-by-Step Process

```
1. Cache Check
   └─ Redis: prediction:{user_id}:{metric}:{date}

2. Model Loading
   ├─ Find latest model for user + metric
   ├─ Load model.pt (PyTorch state dict)
   ├─ Load scalers (feature, label)
   └─ Load config and feature names

3. Feature Preparation
   ├─ Fetch last 30 days of features
   ├─ Engineer 51 features per day
   ├─ Align with training feature names
   └─ Normalize using training scaler

4. Prediction
   ├─ Convert to PyTorch tensor
   ├─ Run LSTM inference (model.predict())
   └─ Get normalized prediction

5. Denormalization
   └─ Inverse transform using label scaler

6. Confidence Calculation
   ├─ Confidence interval: ±1.96 * MAE
   ├─ Confidence score: based on R², distribution, quality
   └─ Lower/upper bounds

7. Historical Context
   ├─ Fetch last 30 days of actual values
   ├─ Calculate: avg, min, max, std
   ├─ Deviation: predicted - avg
   └─ Percentile: where prediction falls in history

8. Interpretation
   ├─ Natural language explanation
   ├─ Actionable recommendations
   └─ Trend analysis (higher/lower/similar)

9. Caching
   └─ Cache in Redis (24-hour TTL)
```

---

## 🧪 Testing Recommendations

### Unit Tests

```python
# Test model architecture
def test_lstm_forward_pass():
    config = LSTMConfig(input_dim=51, hidden_dim=128)
    model = HealthMetricLSTM(config)
    x = torch.randn(32, 30, 51)  # batch=32, seq=30, features=51
    output = model(x)
    assert output.shape == (32, 1)

# Test data preparation
def test_sequence_creation():
    features = np.random.randn(90, 51)
    targets = np.random.randn(90)
    X, y = service._create_sequences(features, targets, sequence_length=30)
    assert X.shape == (60, 30, 51)
    assert y.shape == (60,)

# Test normalization
def test_feature_normalization():
    X_train = np.random.randn(48, 30, 51)
    X_val = np.random.randn(12, 30, 51)
    scaler, X_train_norm, X_val_norm = service._normalize_features(X_train, X_val)
    assert X_train_norm.shape == X_train.shape
    assert X_val_norm.shape == X_val.shape
```

### Integration Tests

```python
# Test end-to-end training
async def test_train_model():
    request = TrainModelRequest(
        user_id="test_user",
        metric=PredictionMetric.RHR,
        epochs=5,  # Quick test
        lookback_days=60,
    )
    response = await training_service.train_model(request)
    assert response.model_id.startswith("test_user_RESTING_HEART_RATE")
    assert response.training_metrics.mae > 0
    assert 0 <= response.training_metrics.r2_score <= 1

# Test end-to-end prediction
async def test_predict():
    request = PredictRequest(
        user_id="test_user",
        metric=PredictionMetric.RHR,
        target_date=date.today() + timedelta(days=1),
    )
    response = await prediction_service.predict(request)
    assert response.prediction.predicted_value > 0
    assert 0 <= response.prediction.confidence_score <= 1
```

### API Tests

```bash
# Test training endpoint
pytest -k "test_train_endpoint"

# Test prediction endpoint
pytest -k "test_predict_endpoint"

# Test batch prediction
pytest -k "test_batch_predict_endpoint"

# Test model listing
pytest -k "test_list_models_endpoint"
```

---

## 📝 Next Steps (Phase 3 & Beyond)

### Phase 3: Interpretability & Optimization

- [ ] **SHAP/LIME**: Explain which features influence predictions
- [ ] **Attention Mechanism**: Identify important time steps
- [ ] **Feature Importance**: Rank features by impact
- [ ] **Counterfactuals**: "What if I ate more protein?"
- [ ] **Hyperparameter Optimization**: Grid search / Bayesian optimization
- [ ] **Model Ensembles**: Combine LSTM + XGBoost
- [ ] **Multi-metric Models**: Single model predicting RHR + HRV

### Phase 4: Production & Monitoring

- [ ] **Model Monitoring**: Track prediction accuracy over time
- [ ] **Drift Detection**: Detect when model performance degrades
- [ ] **Auto-retraining**: Trigger retraining when drift detected
- [ ] **A/B Testing**: Compare model versions
- [ ] **Model Registry**: Database of all trained models
- [ ] **Anomaly Detection**: Flag unusual predictions

### Phase 5: Advanced Features

- [ ] **Transfer Learning**: Pre-train on multiple users
- [ ] **Personalized Models**: Fine-tune on individual users
- [ ] **Real-time Predictions**: Sub-second inference
- [ ] **Mobile Model Deployment**: TensorFlow Lite / Core ML
- [ ] **Federated Learning**: Train without sharing data

---

## 📊 Phase 2 Summary

### What We Built

✅ **PyTorch LSTM Models**
- Configurable architecture (hidden dim, layers, dropout)
- Batch normalization for training stability
- Multi-task learning support
- Baseline models for comparison

✅ **Data Preparation Pipeline**
- Feature matrix building (51 features/day)
- Sliding window sequence creation
- Normalization (StandardScaler)
- Train/val splitting
- PyTorch tensor conversion

✅ **Model Training Service**
- Full PyTorch training loop
- Early stopping and LR scheduling
- Comprehensive evaluation (MAE, RMSE, R², MAPE)
- Production readiness assessment
- Model persistence and versioning

✅ **Prediction Service**
- Model loading from disk
- Input feature preparation
- Confidence intervals and scores
- Historical context and percentiles
- Natural language interpretation
- Actionable recommendations
- Redis caching (24-hour TTL)

✅ **RESTful API**
- POST /train - Train new models
- POST /predict - Make predictions
- POST /batch-predict - Batch predictions
- GET /models - List trained models
- DELETE /models - Delete models

### Key Technologies

- **PyTorch 2.1.2** - Deep learning framework
- **scikit-learn** - StandardScaler, metrics
- **NumPy/Pandas** - Data manipulation
- **FastAPI** - API framework
- **Redis** - Prediction caching
- **SQLAlchemy** - Database ORM

### Files Created

- `app/schemas/predictions.py` (370 lines)
- `app/ml_models/lstm.py` (294 lines)
- `app/ml_models/baseline.py` (114 lines)
- `app/services/data_preparation.py` (497 lines)
- `app/services/model_training.py` (500+ lines)
- `app/services/prediction.py` (600+ lines)
- `app/api/predictions.py` (400+ lines)

**Total**: ~2,800 lines of production-ready code

---

## ✅ Phase 2 is Complete!

**Next**: Phase 3 (Interpretability & Optimization) or production deployment

---

**Last Updated**: 2025-01-15
**Framework**: PyTorch 2.1.2
**Python Version**: 3.11+
