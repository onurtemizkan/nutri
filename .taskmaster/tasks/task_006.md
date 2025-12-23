# Task ID: 6

**Title:** Train and Deploy LSTM Models for Health Predictions

**Status:** pending

**Dependencies:** 3 ✓, 5 ✓

**Priority:** high

**Description:** Train LSTM models for RHR and HRV prediction using the existing model architecture in ml-service/app/ml_models/lstm.py and make them production-ready.

**Details:**

1. Create training pipeline in `ml-service/app/services/model_training.py`:
   - Already has TrainModelRequest/Response schemas
   - Implement data loading from database
   - Create training/validation split (80/20)
   - Add early stopping with patience=10
   - Save model checkpoints and metadata

2. Training data preparation:
   - Use FeatureEngineeringService to generate features
   - Create sliding window sequences (30-day windows)
   - Normalize features using StandardScaler (save scaler with model)
   - Handle missing data: forward-fill then drop incomplete sequences

3. Training configuration:
   - RHR model: hidden_dim=128, num_layers=2, dropout=0.2
   - HRV model: hidden_dim=128, num_layers=2, dropout=0.2
   - Batch size: 32, learning rate: 0.001
   - Use Adam optimizer, MSE loss
   - Train for max 100 epochs with early stopping

4. Model evaluation metrics:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)
   - R² score (>0.5 for production)
   - MAPE (Mean Absolute Percentage Error, <15% for production)

5. Update PredictionService in `ml-service/app/services/prediction.py`:
   - Load trained model from disk
   - Load corresponding scaler
   - Prepare input sequence from recent features
   - Run inference and denormalize output
   - Calculate confidence intervals

6. Model storage structure:
```
ml-service/models/
  {user_id}_{metric}_{timestamp}/
    model.pt              # PyTorch model weights
    scaler.pkl           # Feature scaler
    metadata.pkl         # Training config and metrics
```

7. Add minimum data requirements:
   - At least 30 days of health data
   - At least 21 days of nutrition data
   - Check requirements before training

**Test Strategy:**

1. Unit tests for data preparation pipeline
2. Test training with synthetic data (verify loss decreases)
3. Test model save/load roundtrip
4. Test prediction accuracy on held-out test set
5. Integration test: full train -> predict flow
6. Test minimum data requirement validation
7. Test early stopping triggers correctly

## Subtasks

### 6.1. Implement Minimum Data Requirements Validation

**Status:** pending  
**Dependencies:** None  

Add validation logic to check minimum data requirements before training LSTM models for RHR and HRV prediction.

**Details:**

Add a method `validate_training_data_requirements()` to ModelTrainingService in `ml-service/app/services/model_training.py` that checks: (1) At least 30 days of health data with RHR/HRV measurements, (2) At least 21 days of nutrition data (meals), (3) Sufficient non-null feature coverage. Raise a descriptive ValueError with specific missing requirements if validation fails. Call this method at the start of `train_model()` before data preparation. Add corresponding schema for data requirements status in `schemas/predictions.py`.

### 6.2. Create Training API Endpoints

**Status:** pending  
**Dependencies:** 6.1  

Add FastAPI endpoints for triggering LSTM model training and checking training status.

**Details:**

Create `ml-service/app/api/training.py` with endpoints: (1) POST `/api/v1/models/train` - accepts TrainModelRequest, validates minimum data requirements, starts training asynchronously, returns job_id. (2) GET `/api/v1/models/{model_id}` - returns model info and training metrics. (3) GET `/api/v1/models/user/{user_id}` - lists all models for a user. Register router in `main.py`. Use background tasks for long-running training to prevent HTTP timeout. Store training status in Redis with TTL.

### 6.3. Implement Model Versioning and Management

**Status:** pending  
**Dependencies:** 6.1, 6.2  

Add model versioning system to track multiple model versions per user/metric and enable rollback capabilities.

**Details:**

Enhance ModelTrainingService: (1) Add `_generate_model_version()` using semantic versioning (v1.0.0, v1.0.1, etc.) based on existing models. (2) Add `_set_active_model()` to mark latest trained model as active. (3) Add `get_model_history()` to list all model versions with metrics. (4) Add `rollback_to_version()` to revert to previous model. (5) Add `cleanup_old_models()` to remove models older than N versions (keep last 5). Store version metadata in `metadata.pkl`. Update PredictionService to use active model by default.

### 6.4. Add Prediction API Endpoints

**Status:** pending  
**Dependencies:** 6.3  

Create FastAPI endpoints for making health metric predictions using trained LSTM models.

**Details:**

Create `ml-service/app/api/predictions_api.py` (rename from existing predictions.py if needed) with endpoints: (1) POST `/api/v1/predict` - accepts PredictRequest, returns PredictResponse with prediction, confidence interval, interpretation. (2) POST `/api/v1/predict/batch` - accepts BatchPredictRequest for multiple metrics at once. (3) POST `/api/v1/predict/what-if` - accepts WhatIfRequest for scenario testing. Integrate PredictionService. Add proper error handling for missing models, insufficient data. Cache predictions in Redis with 24h TTL. Register router in main.py.

### 6.5. Create Comprehensive LSTM Training and Prediction Tests

**Status:** pending  
**Dependencies:** 6.1, 6.2, 6.3, 6.4  

Implement end-to-end tests for the complete LSTM training and prediction pipeline using synthetic data.

**Details:**

Create `ml-service/tests/test_lstm_training.py` with tests: (1) `test_training_data_preparation` - verify sliding windows, normalization, train/val split using synthetic data generator. (2) `test_lstm_model_forward_pass` - verify model output shapes with random input. (3) `test_training_loss_decreases` - verify loss decreases over 10 epochs with synthetic data. (4) `test_model_save_load_roundtrip` - verify saved model produces identical predictions after reload. (5) `test_evaluation_metrics` - verify MAE, RMSE, R², MAPE calculations are correct. (6) `test_early_stopping` - verify training stops when validation loss plateaus. (7) `test_full_train_predict_flow` - integration test from training to prediction. Use synthetic data generator from `app/data/synthetic_generator.py`. Mark slow tests with @pytest.mark.slow.
