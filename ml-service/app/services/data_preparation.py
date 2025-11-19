"""
Data Preparation Service for LSTM Training

Prepares time-series data for PyTorch LSTM models:
1. Feature extraction (using FeatureEngineeringService)
2. Sequence creation (sliding windows)
3. Normalization (StandardScaler)
4. Train/val/test split
5. PyTorch tensor conversion
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.feature_engineering import FeatureEngineeringService
from app.schemas.features import FeatureCategory
from app.schemas.predictions import PredictionMetric


class DataPreparationService:
    """
    Service for preparing time-series data for LSTM training and prediction.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_service = FeatureEngineeringService(db)

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    async def prepare_training_data(
        self,
        user_id: str,
        target_metric: PredictionMetric,
        lookback_days: int = 90,
        sequence_length: int = 30,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Prepare complete dataset for model training.

        This is the main function used by the training service.

        Args:
            user_id: User ID
            target_metric: Health metric to predict (RHR, HRV, etc.)
            lookback_days: Days of historical data (default: 90)
            sequence_length: Length of input sequences (default: 30)
            validation_split: Fraction for validation (default: 0.2)

        Returns:
            Dictionary containing:
                - X_train: Training features (batch, seq_len, num_features)
                - y_train: Training labels (batch,)
                - X_val: Validation features
                - y_val: Validation labels
                - scaler: Fitted StandardScaler for features
                - label_scaler: Fitted StandardScaler for labels
                - feature_names: List of feature names
                - num_features: Number of features
                - num_samples: Total samples
        """
        # Step 1: Build feature matrix (one row per day)
        print(f"📊 Fetching features for {lookback_days} days...")
        feature_matrix, dates = await self._build_feature_matrix(
            user_id, lookback_days
        )

        if feature_matrix.empty:
            raise ValueError(f"No feature data available for user {user_id}")

        # Step 2: Fetch target metric values
        print(f"🎯 Fetching target metric: {target_metric.value}")
        target_values = await self._fetch_target_metric(
            user_id, target_metric, dates
        )

        if target_values.empty:
            raise ValueError(
                f"No target metric data ({target_metric.value}) for user {user_id}"
            )

        # Step 3: Align features and targets (remove days with missing target)
        aligned_features, aligned_targets, aligned_dates = self._align_data(
            feature_matrix, target_values, dates
        )

        if len(aligned_features) < sequence_length + 10:
            raise ValueError(
                f"Insufficient data for training: {len(aligned_features)} days "
                f"(need at least {sequence_length + 10})"
            )

        print(f"✅ Aligned {len(aligned_features)} days of data")

        # Step 4: Create sequences (sliding windows)
        print(f"🪟 Creating sequences (length={sequence_length})...")
        X_sequences, y_labels = self._create_sequences(
            aligned_features.values,
            aligned_targets.values,
            sequence_length=sequence_length,
        )

        print(f"✅ Created {len(X_sequences)} sequences")

        # Step 5: Train/validation split
        split_idx = int(len(X_sequences) * (1 - validation_split))
        X_train_raw = X_sequences[:split_idx]
        y_train_raw = y_labels[:split_idx]
        X_val_raw = X_sequences[split_idx:]
        y_val_raw = y_labels[split_idx:]

        # Step 6: Normalize features
        print("📏 Normalizing features...")
        scaler, X_train_norm, X_val_norm = self._normalize_features(
            X_train_raw, X_val_raw
        )

        # Step 7: Normalize labels
        label_scaler, y_train_norm, y_val_norm = self._normalize_labels(
            y_train_raw, y_val_raw
        )

        # Step 8: Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train_norm)
        y_train = torch.FloatTensor(y_train_norm).unsqueeze(1)  # (batch, 1)
        X_val = torch.FloatTensor(X_val_norm)
        y_val = torch.FloatTensor(y_val_norm).unsqueeze(1)

        print(f"✅ Training data prepared:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Validation samples: {len(X_val)}")
        print(f"   - Features per day: {X_train.shape[2]}")
        print(f"   - Sequence length: {sequence_length}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "scaler": scaler,
            "label_scaler": label_scaler,
            "feature_names": list(aligned_features.columns),
            "num_features": X_train.shape[2],
            "num_samples": len(X_sequences),
            "sequence_length": sequence_length,
        }

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

        Args:
            user_id: User ID
            target_date: Date to predict for
            sequence_length: Length of input sequence (from model config)
            scaler: Fitted StandardScaler (from training)
            feature_names: Expected feature names (from training)

        Returns:
            PyTorch tensor of shape (1, sequence_length, num_features)
        """
        # Fetch features for the last sequence_length days before target_date
        start_date = target_date - timedelta(days=sequence_length)

        feature_matrix, dates = await self._build_feature_matrix(
            user_id, lookback_days=sequence_length + 5
        )

        if feature_matrix.empty:
            raise ValueError(f"No feature data available for user {user_id}")

        # Filter to sequence range
        # Convert dates to pandas datetime for comparison with datetime64 index
        start_dt = pd.to_datetime(start_date)
        target_dt = pd.to_datetime(target_date)
        mask = (feature_matrix.index >= start_dt) & (feature_matrix.index < target_dt)
        sequence_features = feature_matrix[mask]

        if len(sequence_features) < sequence_length:
            raise ValueError(
                f"Insufficient data for prediction: {len(sequence_features)} days "
                f"(need {sequence_length})"
            )

        # Take last sequence_length days
        sequence_features = sequence_features.tail(sequence_length)

        # Ensure same feature order as training
        sequence_features = sequence_features[feature_names]

        # Fill missing values with 0 (should be rare after feature engineering)
        sequence_features = sequence_features.fillna(0)

        # Normalize using training scaler
        sequence_norm = scaler.transform(sequence_features.values)

        # Convert to tensor: (1, sequence_length, num_features)
        X = torch.FloatTensor(sequence_norm).unsqueeze(0)

        return X

    # ========================================================================
    # Feature Matrix Building
    # ========================================================================

    async def _build_feature_matrix(
        self, user_id: str, lookback_days: int
    ) -> Tuple[pd.DataFrame, List[date]]:
        """
        Build feature matrix with one row per day.

        Returns:
            (feature_df, dates) where feature_df has dates as index
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        features_by_date = {}
        dates = []

        current_date = start_date
        while current_date <= end_date:
            try:
                # Engineer features for this date
                response = await self.feature_service.engineer_features(
                    user_id=user_id,
                    target_date=current_date,
                    categories=[FeatureCategory.ALL],
                    lookback_days=30,  # Features use 30-day lookback internally
                    force_recompute=False,
                )

                # Flatten features
                flat_features = {}

                if response.nutrition:
                    flat_features.update({
                        f"nutrition_{k}": v
                        for k, v in response.nutrition.model_dump().items()
                        if v is not None
                    })

                if response.activity:
                    flat_features.update({
                        f"activity_{k}": v
                        for k, v in response.activity.model_dump().items()
                        if v is not None
                    })

                if response.health:
                    flat_features.update({
                        f"health_{k}": v
                        for k, v in response.health.model_dump().items()
                        if v is not None
                    })

                if response.temporal:
                    flat_features.update({
                        f"temporal_{k}": v
                        for k, v in response.temporal.model_dump().items()
                        if v is not None and not isinstance(v, bool)
                    })

                    # Convert boolean to int
                    if response.temporal.is_weekend is not None:
                        flat_features["temporal_is_weekend"] = (
                            1 if response.temporal.is_weekend else 0
                        )

                if response.interaction:
                    flat_features.update({
                        f"interaction_{k}": v
                        for k, v in response.interaction.model_dump().items()
                        if v is not None
                    })

                if flat_features:
                    features_by_date[current_date] = flat_features
                    dates.append(current_date)

            except Exception as e:
                print(f"⚠️ Error engineering features for {current_date}: {e}")

            current_date += timedelta(days=1)

        if not features_by_date:
            return pd.DataFrame(), []

        # Convert to DataFrame with dates as index
        feature_df = pd.DataFrame.from_dict(features_by_date, orient="index")
        feature_df.index = pd.to_datetime(feature_df.index)

        return feature_df, dates

    async def _fetch_target_metric(
        self, user_id: str, target_metric: PredictionMetric, dates: List[date]
    ) -> pd.Series:
        """Fetch target health metric values."""
        from app.models.health_metric import HealthMetric
        from sqlalchemy import select, and_

        start_date = min(dates)
        end_date = max(dates)

        result = await self.db.execute(
            select(HealthMetric).where(
                and_(
                    HealthMetric.user_id == user_id,
                    HealthMetric.metric_type == target_metric.value,
                    HealthMetric.recorded_at >= pd.Timestamp(start_date),
                    HealthMetric.recorded_at <= pd.Timestamp(end_date),
                )
            )
        )
        metrics = result.scalars().all()

        if not metrics:
            return pd.Series(dtype=float)

        # Group by date and average
        data = {}
        for metric in metrics:
            metric_date = metric.recorded_at.date()
            if metric_date not in data:
                data[metric_date] = []
            data[metric_date].append(metric.value)

        averaged = {d: np.mean(values) for d, values in data.items()}
        series = pd.Series(averaged)
        series.index = pd.to_datetime(series.index)

        return series

    def _align_data(
        self,
        feature_matrix: pd.DataFrame,
        target_values: pd.Series,
        dates: List[date],
    ) -> Tuple[pd.DataFrame, pd.Series, List[date]]:
        """Align features and targets, remove days with missing target."""
        # Find dates with both features and target
        valid_dates = feature_matrix.index.intersection(target_values.index)

        aligned_features = feature_matrix.loc[valid_dates]
        aligned_targets = target_values.loc[valid_dates]

        # Sort by date
        aligned_features = aligned_features.sort_index()
        aligned_targets = aligned_targets.sort_index()

        aligned_dates = [d.date() for d in aligned_features.index]

        return aligned_features, aligned_targets, aligned_dates

    # ========================================================================
    # Sequence Creation
    # ========================================================================

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM.

        Args:
            features: Array of shape (num_days, num_features)
            targets: Array of shape (num_days,)
            sequence_length: Length of input sequences

        Returns:
            X: Array of shape (num_sequences, sequence_length, num_features)
            y: Array of shape (num_sequences,)

        Example:
            If we have 90 days of data and sequence_length=30:
            - Days 0-29 → predict day 30
            - Days 1-30 → predict day 31
            - ...
            - Days 59-89 → predict day 90
            Result: 60 sequences
        """
        X = []
        y = []

        for i in range(len(features) - sequence_length):
            # Input: days i to i+sequence_length
            X.append(features[i : i + sequence_length])

            # Output: day i+sequence_length
            y.append(targets[i + sequence_length])

        return np.array(X), np.array(y)

    # ========================================================================
    # Normalization
    # ========================================================================

    def _normalize_features(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
        """
        Normalize features using StandardScaler.

        Fits scaler on training data only, then transforms both train and val.
        Handles NaN values and zero-variance features.

        Args:
            X_train: Training features (num_samples, seq_len, num_features)
            X_val: Validation features

        Returns:
            (scaler, X_train_norm, X_val_norm)
        """
        # Reshape to 2D for fitting: (num_samples * seq_len, num_features)
        num_samples_train, seq_len, num_features = X_train.shape
        X_train_2d = X_train.reshape(-1, num_features)

        # Replace NaN with 0 before normalization
        X_train_2d = np.nan_to_num(X_train_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_norm_2d = scaler.fit_transform(X_train_2d)

        # Replace NaN values that may result from zero-variance features
        X_train_norm_2d = np.nan_to_num(X_train_norm_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape back to 3D
        X_train_norm = X_train_norm_2d.reshape(num_samples_train, seq_len, num_features)

        # Transform validation data
        if len(X_val) > 0:
            num_samples_val = X_val.shape[0]
            X_val_2d = X_val.reshape(-1, num_features)

            # Replace NaN with 0
            X_val_2d = np.nan_to_num(X_val_2d, nan=0.0, posinf=0.0, neginf=0.0)

            X_val_norm_2d = scaler.transform(X_val_2d)

            # Replace NaN values
            X_val_norm_2d = np.nan_to_num(X_val_norm_2d, nan=0.0, posinf=0.0, neginf=0.0)

            X_val_norm = X_val_norm_2d.reshape(num_samples_val, seq_len, num_features)
        else:
            X_val_norm = X_val

        return scaler, X_train_norm, X_val_norm

    def _normalize_labels(
        self, y_train: np.ndarray, y_val: np.ndarray
    ) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
        """
        Normalize labels using StandardScaler.
        Handles NaN values that may exist in labels.

        Args:
            y_train: Training labels (num_samples,)
            y_val: Validation labels

        Returns:
            (label_scaler, y_train_norm, y_val_norm)
        """
        label_scaler = StandardScaler()

        # Replace NaN with 0 before normalization
        y_train_clean = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit on training labels
        y_train_norm = label_scaler.fit_transform(y_train_clean.reshape(-1, 1)).flatten()

        # Replace any NaN that may result from normalization
        y_train_norm = np.nan_to_num(y_train_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Transform validation labels
        if len(y_val) > 0:
            y_val_clean = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_val_norm = label_scaler.transform(y_val_clean.reshape(-1, 1)).flatten()
            y_val_norm = np.nan_to_num(y_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            y_val_norm = y_val

        return label_scaler, y_train_norm, y_val_norm

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def denormalize_prediction(
        self, normalized_value: float, label_scaler: StandardScaler
    ) -> float:
        """
        Convert normalized prediction back to original scale.

        Args:
            normalized_value: Predicted value in normalized space
            label_scaler: Fitted StandardScaler for labels

        Returns:
            Denormalized prediction
        """
        return float(
            label_scaler.inverse_transform([[normalized_value]])[0][0]
        )
