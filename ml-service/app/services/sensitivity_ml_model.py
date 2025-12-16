"""
Sensitivity Pattern ML Model

Machine learning model for food sensitivity detection and prediction.

Features:
- Reaction probability prediction for new exposures
- Severity classification based on historical patterns
- Multi-trigger interaction detection
- Personalized threshold learning
- Online learning for continuous improvement

Models:
- XGBoost for reaction prediction (tabular data)
- LSTM for temporal HRV pattern analysis (optional)
- Isolation Forest for anomaly detection

Based on research showing:
- 90.5% sensitivity, 79.4% specificity for HRV-based allergy detection
- 17-minute early warning capability
"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import IsolationForest

from app.services.hrv_sensitivity_analyzer import (
    TimeWindow,
)
from app.models.sensitivity import (
    AllergenType,
    ReactionSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Feature names for the model
FEATURE_COLUMNS = [
    # HRV features per window
    "hrv_drop_immediate",
    "hrv_drop_short_term",
    "hrv_drop_medium_term",
    "hrv_drop_extended",
    "hrv_drop_next_day",
    "max_hrv_drop",
    "mean_hrv_drop",
    "std_hrv_drop",
    # Baseline features
    "baseline_rmssd",
    "baseline_std",
    # Trigger features
    "trigger_type_encoded",
    "compound_level",  # For compounds like histamine
    # Temporal features
    "hour_of_day",
    "day_of_week",
    # Historical features
    "prior_reaction_rate",
    "days_since_last_exposure",
    "exposure_count_last_30d",
    # Compound interaction features
    "has_dao_inhibitor",
    "has_histamine_liberator",
    "total_histamine_mg",
    "total_tyramine_mg",
]

# Model hyperparameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# Severity model hyperparameters
XGB_SEVERITY_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 80,
    "num_class": 4,  # none, mild, moderate, severe
    "random_state": 42,
}

# Minimum training samples
MIN_TRAINING_SAMPLES = 30


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrainingDataPoint:
    """Single training data point for the ML model."""

    # Features
    hrv_drops: Dict[TimeWindow, float]  # HRV drop per window
    baseline_rmssd: float
    baseline_std: float
    trigger_type: str
    compound_level: Optional[float] = None
    hour_of_day: int = 12
    day_of_week: int = 0
    prior_reaction_rate: float = 0.0
    days_since_last_exposure: int = 30
    exposure_count_last_30d: int = 0
    has_dao_inhibitor: bool = False
    has_histamine_liberator: bool = False
    total_histamine_mg: float = 0.0
    total_tyramine_mg: float = 0.0

    # Labels
    had_reaction: bool = False
    reaction_severity: ReactionSeverity = ReactionSeverity.NONE


@dataclass
class PredictionResult:
    """Result of sensitivity prediction."""

    reaction_probability: float
    predicted_severity: ReactionSeverity
    severity_probabilities: Dict[str, float]
    confidence: float
    feature_importances: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray
    feature_importance: Dict[str, float]
    training_samples: int
    evaluation_date: datetime


# =============================================================================
# MAIN ML SERVICE
# =============================================================================


class SensitivityMLModel:
    """
    Machine learning model for food sensitivity prediction.

    Uses XGBoost for reaction prediction and severity classification,
    with support for online learning and personalization.
    """

    def __init__(self, model_dir: str = "./models/sensitivity"):
        """
        Initialize the ML model.

        Args:
            model_dir: Directory for model persistence
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.reaction_model: Optional[xgb.XGBClassifier] = None
        self.severity_model: Optional[xgb.XGBClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None

        # Preprocessors
        self.scaler = StandardScaler()
        self.trigger_encoder = LabelEncoder()

        # State
        self.is_trained = False
        self.training_metrics: Optional[ModelMetrics] = None

        # Initialize trigger encoder with known allergens
        all_triggers = [a.value for a in AllergenType] + [
            "histamine",
            "tyramine",
            "fodmap",
            "salicylate",
            "oxalate",
            "lectin",
        ]
        self.trigger_encoder.fit(all_triggers)

        logger.info(
            f"Initialized SensitivityMLModel "
            f"(XGBoost: {'enabled' if XGBOOST_AVAILABLE else 'disabled'})"
        )

    def prepare_features(
        self, data_points: List[TrainingDataPoint]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels from training data.

        Returns:
            Tuple of (features, reaction_labels, severity_labels)
        """
        features = []
        reaction_labels = []
        severity_labels = []

        severity_map = {
            ReactionSeverity.NONE: 0,
            ReactionSeverity.MILD: 1,
            ReactionSeverity.MODERATE: 2,
            ReactionSeverity.SEVERE: 3,
            ReactionSeverity.EMERGENCY: 3,  # Group with severe
        }

        for dp in data_points:
            # Build feature vector
            hrv_drops = [
                dp.hrv_drops.get(TimeWindow.IMMEDIATE, 0.0),
                dp.hrv_drops.get(TimeWindow.SHORT_TERM, 0.0),
                dp.hrv_drops.get(TimeWindow.MEDIUM_TERM, 0.0),
                dp.hrv_drops.get(TimeWindow.EXTENDED, 0.0),
                dp.hrv_drops.get(TimeWindow.NEXT_DAY, 0.0),
            ]

            feature_vector = [
                # HRV features
                *hrv_drops,
                max(abs(d) for d in hrv_drops) if hrv_drops else 0.0,
                np.mean([abs(d) for d in hrv_drops]) if hrv_drops else 0.0,
                np.std([abs(d) for d in hrv_drops]) if hrv_drops else 0.0,
                # Baseline features
                dp.baseline_rmssd,
                dp.baseline_std,
                # Trigger features
                self._encode_trigger(dp.trigger_type),
                dp.compound_level or 0.0,
                # Temporal features
                dp.hour_of_day,
                dp.day_of_week,
                # Historical features
                dp.prior_reaction_rate,
                dp.days_since_last_exposure,
                dp.exposure_count_last_30d,
                # Compound interaction features
                1.0 if dp.has_dao_inhibitor else 0.0,
                1.0 if dp.has_histamine_liberator else 0.0,
                dp.total_histamine_mg,
                dp.total_tyramine_mg,
            ]

            features.append(feature_vector)
            reaction_labels.append(1 if dp.had_reaction else 0)
            severity_labels.append(severity_map.get(dp.reaction_severity, 0))

        return (
            np.array(features, dtype=np.float32),
            np.array(reaction_labels, dtype=np.int32),
            np.array(severity_labels, dtype=np.int32),
        )

    def _encode_trigger(self, trigger_type: str) -> float:
        """Encode trigger type to numeric value."""
        try:
            return float(self.trigger_encoder.transform([trigger_type])[0])
        except ValueError:
            return -1.0  # Unknown trigger

    def train(
        self,
        data_points: List[TrainingDataPoint],
        test_size: float = 0.2,
    ) -> ModelMetrics:
        """
        Train the sensitivity prediction models.

        Args:
            data_points: List of training data points
            test_size: Fraction of data for testing

        Returns:
            Training metrics
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "XGBoost not available. Install with: pip install xgboost"
            )

        if len(data_points) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Insufficient training data: {len(data_points)} samples, "
                f"need at least {MIN_TRAINING_SAMPLES}"
            )

        logger.info(f"Training sensitivity model on {len(data_points)} samples")

        # Prepare features
        X, y_reaction, y_severity = self.prepare_features(data_points)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y_reaction,
            test_size=test_size,
            random_state=42,
            stratify=y_reaction,
        )

        # Train reaction prediction model
        self.reaction_model = xgb.XGBClassifier(**XGB_PARAMS)
        self.reaction_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.reaction_model.predict(X_test)
        y_prob = self.reaction_model.predict_proba(X_test)[:, 1]

        # Train severity model (on samples with reactions only)
        reaction_indices = y_reaction == 1
        if np.sum(reaction_indices) >= 10:
            X_severity = X_scaled[reaction_indices]
            y_sev = y_severity[reaction_indices]

            self.severity_model = xgb.XGBClassifier(**XGB_SEVERITY_PARAMS)
            self.severity_model.fit(X_severity, y_sev)
        else:
            logger.warning("Insufficient reaction samples for severity model")

        # Train anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
        )
        self.anomaly_detector.fit(X_scaled)

        # Calculate metrics
        feature_importance = dict(
            zip(
                FEATURE_COLUMNS[: len(self.reaction_model.feature_importances_)],
                self.reaction_model.feature_importances_.tolist(),
            )
        )

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=(
                roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
            ),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            feature_importance=feature_importance,
            training_samples=len(data_points),
            evaluation_date=datetime.utcnow(),
        )

        self.is_trained = True
        self.training_metrics = metrics

        logger.info(
            f"Model trained: accuracy={metrics.accuracy:.3f}, "
            f"recall={metrics.recall:.3f}, AUC={metrics.auc_roc:.3f}"
        )

        return metrics

    def predict(
        self,
        data_point: TrainingDataPoint,
    ) -> PredictionResult:
        """
        Predict reaction probability and severity for a new exposure.

        Args:
            data_point: Data point to predict

        Returns:
            PredictionResult with probabilities and recommendations
        """
        if not self.is_trained:
            return self._fallback_prediction(data_point)

        # Prepare features
        X, _, _ = self.prepare_features([data_point])
        X_scaled = self.scaler.transform(X)

        # Predict reaction probability
        reaction_prob = self.reaction_model.predict_proba(X_scaled)[0, 1]

        # Predict severity
        severity_probs = {
            "none": 1.0 - reaction_prob,
            "mild": 0.0,
            "moderate": 0.0,
            "severe": 0.0,
        }

        predicted_severity = ReactionSeverity.NONE

        if self.severity_model is not None and reaction_prob > 0.3:
            sev_probs = self.severity_model.predict_proba(X_scaled)[0]
            severity_probs = {
                "none": sev_probs[0] * (1 - reaction_prob),
                "mild": sev_probs[1] * reaction_prob,
                "moderate": sev_probs[2] * reaction_prob,
                "severe": sev_probs[3] * reaction_prob,
            }

            # Get predicted class
            pred_class = np.argmax(sev_probs)
            severity_map = {
                0: ReactionSeverity.NONE,
                1: ReactionSeverity.MILD,
                2: ReactionSeverity.MODERATE,
                3: ReactionSeverity.SEVERE,
            }
            predicted_severity = severity_map.get(pred_class, ReactionSeverity.NONE)

        # Check for anomalies
        is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1

        # Calculate confidence
        confidence = self._calculate_confidence(reaction_prob, is_anomaly)

        # Get feature importances for this prediction
        feature_importances = self._get_local_feature_importance(X_scaled[0])

        # Identify risk factors
        risk_factors = self._identify_risk_factors(data_point, feature_importances)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            reaction_prob, predicted_severity, risk_factors
        )

        return PredictionResult(
            reaction_probability=round(reaction_prob, 3),
            predicted_severity=predicted_severity,
            severity_probabilities={k: round(v, 3) for k, v in severity_probs.items()},
            confidence=round(confidence, 3),
            feature_importances=feature_importances,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def _fallback_prediction(self, data_point: TrainingDataPoint) -> PredictionResult:
        """
        Fallback prediction when model is not trained.
        Uses rule-based approach.
        """
        # Calculate basic risk from HRV drops
        max_drop = (
            max(abs(d) for d in data_point.hrv_drops.values())
            if data_point.hrv_drops
            else 0
        )
        _ = (
            np.mean([abs(d) for d in data_point.hrv_drops.values()])
            if data_point.hrv_drops
            else 0
        )

        # Base probability from HRV drops
        if max_drop >= 25:
            reaction_prob = 0.85
            severity = ReactionSeverity.SEVERE
        elif max_drop >= 15:
            reaction_prob = 0.65
            severity = ReactionSeverity.MODERATE
        elif max_drop >= 10:
            reaction_prob = 0.45
            severity = ReactionSeverity.MILD
        else:
            reaction_prob = 0.2
            severity = ReactionSeverity.NONE

        # Adjust for historical reaction rate
        if data_point.prior_reaction_rate > 0.5:
            reaction_prob = min(0.95, reaction_prob + 0.2)

        # Adjust for compound interactions
        if data_point.has_dao_inhibitor and data_point.total_histamine_mg > 10:
            reaction_prob = min(0.95, reaction_prob + 0.15)

        risk_factors = []
        if max_drop >= 15:
            risk_factors.append(f"Significant HRV drop ({max_drop:.0f}%)")
        if data_point.prior_reaction_rate > 0.5:
            risk_factors.append("High historical reaction rate")
        if data_point.has_dao_inhibitor:
            risk_factors.append("DAO inhibitor present")

        return PredictionResult(
            reaction_probability=round(reaction_prob, 3),
            predicted_severity=severity,
            severity_probabilities={
                "none": round(1 - reaction_prob, 3),
                "mild": (
                    round(reaction_prob * 0.4, 3) if severity.value == "mild" else 0.0
                ),
                "moderate": (
                    round(reaction_prob * 0.4, 3)
                    if severity.value == "moderate"
                    else 0.0
                ),
                "severe": (
                    round(reaction_prob * 0.2, 3) if severity.value == "severe" else 0.0
                ),
            },
            confidence=0.5,  # Low confidence for fallback
            feature_importances={},
            risk_factors=risk_factors,
            recommendations=["Model not yet trained. Using rule-based prediction."],
        )

    def _calculate_confidence(self, reaction_prob: float, is_anomaly: bool) -> float:
        """Calculate prediction confidence."""
        # Base confidence from probability extremes
        # More extreme probabilities = higher confidence
        prob_extremity = 2 * abs(reaction_prob - 0.5)

        # Reduce confidence for anomalies
        anomaly_factor = 0.7 if is_anomaly else 1.0

        # Factor in model metrics
        model_factor = self.training_metrics.auc_roc if self.training_metrics else 0.5

        confidence = prob_extremity * 0.4 + model_factor * 0.6
        confidence *= anomaly_factor

        return min(0.99, max(0.1, confidence))

    def _get_local_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for a specific prediction."""
        if not self.is_trained or self.reaction_model is None:
            return {}

        # Use global feature importance weighted by feature values
        global_importance = self.reaction_model.feature_importances_

        # Normalize feature values
        feature_contrib = features * global_importance[: len(features)]

        result = {}
        for i, name in enumerate(FEATURE_COLUMNS[: len(feature_contrib)]):
            if abs(feature_contrib[i]) > 0.01:
                result[name] = round(float(feature_contrib[i]), 4)

        # Sort by absolute importance
        result = dict(
            sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

        return result

    def _identify_risk_factors(
        self, data_point: TrainingDataPoint, feature_importances: Dict[str, float]
    ) -> List[str]:
        """Identify human-readable risk factors."""
        risk_factors = []

        # HRV-based risks
        max_drop = (
            max(abs(d) for d in data_point.hrv_drops.values())
            if data_point.hrv_drops
            else 0
        )
        if max_drop >= 20:
            risk_factors.append(f"Large HRV drop observed ({max_drop:.0f}%)")
        elif max_drop >= 10:
            risk_factors.append(f"Moderate HRV drop observed ({max_drop:.0f}%)")

        # Historical risk
        if data_point.prior_reaction_rate >= 0.7:
            risk_factors.append(
                f"High historical reaction rate ({data_point.prior_reaction_rate*100:.0f}%)"
            )
        elif data_point.prior_reaction_rate >= 0.4:
            risk_factors.append("Moderate historical reaction rate")

        # Compound risks
        if data_point.has_dao_inhibitor:
            risk_factors.append("DAO inhibitor in meal (slows histamine metabolism)")

        if data_point.has_histamine_liberator:
            risk_factors.append("Histamine liberator present")

        if data_point.total_histamine_mg > 50:
            risk_factors.append(
                f"High histamine load ({data_point.total_histamine_mg:.0f}mg)"
            )

        if data_point.total_tyramine_mg > 25:
            risk_factors.append(
                f"Elevated tyramine ({data_point.total_tyramine_mg:.0f}mg)"
            )

        # Low baseline HRV
        if data_point.baseline_rmssd < 25:
            risk_factors.append("Below-average baseline HRV (increased vulnerability)")

        # Recent exposures
        if data_point.exposure_count_last_30d >= 5:
            risk_factors.append("Multiple recent exposures to this trigger")

        return risk_factors

    def _generate_recommendations(
        self, reaction_prob: float, severity: ReactionSeverity, risk_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if reaction_prob >= 0.7:
            recommendations.append(
                "HIGH RISK: Consider avoiding this food or reducing portion significantly"
            )
        elif reaction_prob >= 0.5:
            recommendations.append(
                "MODERATE RISK: Monitor closely for symptoms after eating"
            )
        elif reaction_prob >= 0.3:
            recommendations.append("SLIGHT RISK: Be aware of potential mild reaction")

        if severity == ReactionSeverity.SEVERE:
            recommendations.append(
                "If consuming, ensure antihistamine/medication available"
            )

        if "DAO inhibitor" in str(risk_factors):
            recommendations.append(
                "Consider taking DAO supplement 15 min before eating"
            )

        if "histamine" in str(risk_factors).lower():
            recommendations.append("Space high-histamine foods throughout the day")

        if "baseline HRV" in str(risk_factors):
            recommendations.append("Stress reduction before meal may improve tolerance")

        return recommendations

    def save_model(self, filename: str = "sensitivity_model.pkl"):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        model_path = self.model_dir / filename
        model_data = {
            "reaction_model": self.reaction_model,
            "severity_model": self.severity_model,
            "anomaly_detector": self.anomaly_detector,
            "scaler": self.scaler,
            "trigger_encoder": self.trigger_encoder,
            "training_metrics": self.training_metrics,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str = "sensitivity_model.pkl"):
        """Load trained model from disk."""
        model_path = self.model_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.reaction_model = model_data["reaction_model"]
        self.severity_model = model_data["severity_model"]
        self.anomaly_detector = model_data["anomaly_detector"]
        self.scaler = model_data["scaler"]
        self.trigger_encoder = model_data["trigger_encoder"]
        self.training_metrics = model_data["training_metrics"]
        self.is_trained = True

        logger.info(f"Model loaded from {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics."""
        info = {
            "is_trained": self.is_trained,
            "xgboost_available": XGBOOST_AVAILABLE,
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
        }

        if self.training_metrics:
            info["metrics"] = {
                "accuracy": self.training_metrics.accuracy,
                "precision": self.training_metrics.precision,
                "recall": self.training_metrics.recall,
                "f1": self.training_metrics.f1,
                "auc_roc": self.training_metrics.auc_roc,
                "training_samples": self.training_metrics.training_samples,
                "evaluation_date": self.training_metrics.evaluation_date.isoformat(),
            }
            info["top_features"] = dict(
                sorted(
                    self.training_metrics.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            )

        return info


# Singleton instance
sensitivity_ml_model = SensitivityMLModel()
