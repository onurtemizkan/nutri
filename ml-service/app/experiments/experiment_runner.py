"""
Comprehensive Experimentation Framework for Health Prediction Models

Features:
- Optuna hyperparameter optimization
- Cross-validation on synthetic data
- Model comparison and evaluation
- Experiment tracking and logging
- Visualization of results

Based on best practices from:
- Optuna documentation
- Machine Learning Mastery tutorials
- Research paper methodologies
"""
# mypy: ignore-errors

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

# Import our models
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.advanced_lstm import (  # noqa: E402
    ModelFactory,
)
from data.synthetic_generator import (  # noqa: E402
    SyntheticDataGenerator,
    UserPersona,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    model_type: str  # 'lstm_attention', 'bilstm_residual', 'tcn'
    target_metric: str  # 'RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_RMSSD'
    sequence_length: int = 30
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from an experiment run."""

    config: ExperimentConfig
    train_loss: float
    val_loss: float
    test_metrics: Dict[str, float]
    training_time: float
    best_epoch: int
    model_path: Optional[str] = None
    attention_weights: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None


@dataclass
class HyperparameterSearchResult:
    """Results from hyperparameter search."""

    best_params: Dict[str, Any]
    best_value: float
    all_trials: List[Dict[str, Any]]
    study_name: str
    n_trials: int
    optimization_time: float


# =============================================================================
# Data Preparation
# =============================================================================


class ExperimentDataLoader:
    """
    Prepares data for experiments from synthetic generator.
    """

    def __init__(
        self,
        num_days: int = 180,
        seed: int = 42,
    ):
        self.num_days = num_days
        self.seed = seed
        self.generator = SyntheticDataGenerator(seed=seed)
        self._data_cache: Dict[str, Dict] = {}

    def get_user_data(
        self,
        persona: UserPersona,
        target_metric: str = "RESTING_HEART_RATE",
        sequence_length: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get prepared data for a single user persona.

        Returns:
            X: Features array (num_samples, seq_len, num_features)
            y: Targets array (num_samples,)
            feature_names: List of feature names
        """
        cache_key = f"{persona.value}_{self.num_days}"

        if cache_key not in self._data_cache:
            user_id = f"synthetic_{persona.value}"
            data = self.generator.generate_user_data(
                persona=persona,
                user_id=user_id,
                num_days=self.num_days,
            )
            self._data_cache[cache_key] = data

        data = self._data_cache[cache_key]

        # Extract features and targets
        X, y, feature_names = self._prepare_sequences(
            meals_df=data["meals"],
            activities_df=data["activities"],
            health_df=data["health_metrics"],
            target_metric=target_metric,
            sequence_length=sequence_length,
        )

        return X, y, feature_names

    def get_all_users_data(
        self,
        target_metric: str = "RESTING_HEART_RATE",
        sequence_length: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Get combined data from all user personas.

        Returns:
            X: Combined features
            y: Combined targets
            feature_names: Feature names
            user_ids: List mapping each sample to user
        """
        all_X = []
        all_y = []
        all_user_ids = []
        feature_names = None

        for persona in UserPersona:
            X, y, fnames = self.get_user_data(
                persona=persona,
                target_metric=target_metric,
                sequence_length=sequence_length,
            )

            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_user_ids.extend([persona.value] * len(X))

                if feature_names is None:
                    feature_names = fnames

        if not all_X:
            raise ValueError("No data generated for any persona")

        return (
            np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            feature_names,
            all_user_ids,
        )

    def _prepare_sequences(
        self,
        meals_df: pd.DataFrame,
        activities_df: pd.DataFrame,
        health_df: pd.DataFrame,
        target_metric: str,
        sequence_length: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Prepare feature sequences and targets from raw data.
        """
        if meals_df.empty or health_df.empty:
            return None, None, []

        # Get date range
        all_dates = sorted(
            set(
                health_df[health_df["metric_type"] == target_metric][
                    "recorded_at"
                ].dt.date
            )
        )

        if len(all_dates) < sequence_length + 10:
            return None, None, []

        # Build daily feature vectors
        daily_features = {}
        feature_names = []

        for current_date in all_dates:
            features = {}

            # Nutrition features
            day_meals = meals_df[meals_df["consumed_at"].dt.date == current_date]

            features["calories_daily"] = day_meals["calories"].sum()
            features["protein_daily"] = day_meals["protein"].sum()
            features["carbs_daily"] = day_meals["carbs"].sum()
            features["fat_daily"] = day_meals["fat"].sum()
            features["fiber_daily"] = (
                day_meals["fiber"].sum() if "fiber" in day_meals else 0
            )
            features["meal_count"] = len(day_meals)

            # Late eating
            late_meals = day_meals[day_meals["consumed_at"].dt.hour >= 20]
            features["late_night_calories"] = late_meals["calories"].sum()

            # Activity features
            day_activities = activities_df[
                activities_df["started_at"].dt.date == current_date
            ]

            features["active_minutes"] = day_activities["duration"].sum()
            features["calories_burned"] = day_activities["calories_burned"].sum()
            features["workout_count"] = len(
                day_activities[day_activities["activity_type"] != "WALKING"]
            )

            high_intensity = day_activities[day_activities["intensity"] == "HIGH"]
            features["high_intensity_minutes"] = high_intensity["duration"].sum()

            # Health features (lagged - use previous values as features)
            for metric_type in [
                "RESTING_HEART_RATE",
                "HEART_RATE_VARIABILITY_RMSSD",
                "SLEEP_DURATION",
                "SLEEP_SCORE",
                "RECOVERY_SCORE",
            ]:
                metric_data = health_df[
                    (health_df["metric_type"] == metric_type)
                    & (health_df["recorded_at"].dt.date == current_date)
                ]
                if not metric_data.empty:
                    features[f"{metric_type.lower()}_prev"] = metric_data["value"].iloc[
                        0
                    ]
                else:
                    features[f"{metric_type.lower()}_prev"] = np.nan

            # Temporal features
            dt = datetime.combine(current_date, datetime.min.time())
            features["day_of_week"] = dt.weekday()
            features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
            features["month"] = dt.month

            daily_features[current_date] = features

            if not feature_names:
                feature_names = list(features.keys())

        # Convert to DataFrame
        features_df = pd.DataFrame.from_dict(daily_features, orient="index")
        features_df = features_df.fillna(method="ffill").fillna(0)

        # Get target values
        target_df = health_df[health_df["metric_type"] == target_metric].copy()
        target_df["date"] = target_df["recorded_at"].dt.date
        target_df = target_df.groupby("date")["value"].mean()

        # Align dates
        common_dates = sorted(set(features_df.index) & set(target_df.index))

        if len(common_dates) < sequence_length + 10:
            return None, None, []

        # Create sequences
        X_sequences = []
        y_targets = []

        for i in range(len(common_dates) - sequence_length):
            seq_dates = common_dates[i : i + sequence_length]
            target_date = common_dates[i + sequence_length]

            # Get feature sequence
            seq_features = features_df.loc[seq_dates].values

            # Get target
            target_value = target_df.loc[target_date]

            X_sequences.append(seq_features)
            y_targets.append(target_value)

        return np.array(X_sequences), np.array(y_targets), feature_names


# =============================================================================
# Training Utilities
# =============================================================================


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())

    predictions_array = np.concatenate(all_predictions, axis=0).flatten()
    actuals_array = np.concatenate(all_actuals, axis=0).flatten()

    return total_loss / num_batches, predictions_array, actuals_array


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoiding division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("inf")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# =============================================================================
# Experiment Runner
# =============================================================================


class ExperimentRunner:
    """
    Runs experiments for model training and evaluation.
    """

    def __init__(
        self,
        output_dir: str = "experiments",
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.data_loader = ExperimentDataLoader()
        self.results: List[ExperimentResult] = []

    def run_experiment(
        self,
        config: ExperimentConfig,
        hyperparams: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration
            hyperparams: Optional hyperparameters to override defaults
            verbose: Print progress

        Returns:
            ExperimentResult with metrics and trained model
        """
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment: {config.name}")
            print(f"Model: {config.model_type}")
            print(f"Target: {config.target_metric}")
            print(f"{'='*60}\n")

        start_time = time.time()

        # Load data
        X, y, feature_names, user_ids = self.data_loader.get_all_users_data(
            target_metric=config.target_metric,
            sequence_length=config.sequence_length,
        )

        if verbose:
            print(f"Data loaded: {X.shape[0]} samples, {X.shape[2]} features")

        # Normalize features
        num_samples, seq_len, num_features = X.shape
        X_flat = X.reshape(-1, num_features)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_flat).reshape(
            num_samples, seq_len, num_features
        )

        # Normalize targets
        y_scaler = StandardScaler()
        y_norm = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Split data (time-series aware)
        n_train = int(len(X_norm) * config.train_split)
        n_val = int(len(X_norm) * config.val_split)

        X_train = X_norm[:n_train]
        y_train = y_norm[:n_train]
        X_val = X_norm[n_train : n_train + n_val]
        y_val = y_norm[n_train : n_train + n_val]
        X_test = X_norm[n_train + n_val :]
        y_test = y_norm[n_train + n_val :]

        if verbose:
            print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

        test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Create model
        hp = hyperparams or {}
        model = ModelFactory.create(
            model_type=config.model_type,
            input_dim=num_features,
            hidden_dim=hp.get("hidden_dim", 128),
            num_layers=hp.get("num_layers", 2),
            dropout=hp.get("dropout", 0.2),
        )
        model = model.to(self.device)

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=hp.get("learning_rate", config.learning_rate),
            weight_decay=hp.get("weight_decay", 1e-5),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # Training loop
        train_losses = []
        val_losses = []
        best_epoch = 0

        for epoch in range(config.epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, self.device
            )
            val_loss, _, _ = evaluate(model, val_loader, criterion, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if early_stopping(val_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if val_loss <= early_stopping.best_loss:
                best_epoch = epoch + 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        # Restore best model
        early_stopping.restore_best_weights(model)

        # Evaluate on test set
        test_loss, predictions_norm, actuals_norm = evaluate(
            model, test_loader, criterion, self.device
        )

        # Denormalize for real metrics
        predictions = y_scaler.inverse_transform(
            predictions_norm.reshape(-1, 1)
        ).flatten()
        actuals = y_scaler.inverse_transform(actuals_norm.reshape(-1, 1)).flatten()

        test_metrics = calculate_metrics(actuals, predictions)

        if verbose:
            print("\nTest Results:")
            print(f"  MAE: {test_metrics['mae']:.4f}")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
            print(f"  R²: {test_metrics['r2']:.4f}")
            print(f"  MAPE: {test_metrics['mape']:.2f}%")

        training_time = time.time() - start_time

        # Save model
        model_path = self.output_dir / f"{config.name}_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "y_scaler": y_scaler,
                "config": config,
                "hyperparams": hyperparams,
                "feature_names": feature_names,
            },
            model_path,
        )

        # Get attention weights if available
        attention_weights = None
        if hasattr(model, "get_attention_weights"):
            model.eval()
            with torch.no_grad():
                sample_input = X_test_t[:10].to(self.device)
                attention_weights = model.get_attention_weights(sample_input)
                attention_weights = attention_weights.cpu().numpy()

        result = ExperimentResult(
            config=config,
            train_loss=train_losses[-1],
            val_loss=val_losses[-1],
            test_metrics=test_metrics,
            training_time=training_time,
            best_epoch=best_epoch,
            model_path=str(model_path),
            attention_weights=attention_weights,
            predictions=predictions,
            actuals=actuals,
        )

        self.results.append(result)
        return result

    def run_model_comparison(
        self,
        target_metric: str = "RESTING_HEART_RATE",
        sequence_length: int = 30,
        seed: int = 42,
    ) -> Dict[str, ExperimentResult]:
        """
        Compare all three model architectures.

        Returns:
            Dictionary mapping model type to results
        """
        results = {}

        for model_type in ["lstm_attention", "bilstm_residual", "tcn"]:
            config = ExperimentConfig(
                name=f"{model_type}_{target_metric}",
                model_type=model_type,
                target_metric=target_metric,
                sequence_length=sequence_length,
                seed=seed,
            )

            result = self.run_experiment(config)
            results[model_type] = result

        # Print comparison
        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)
        print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Time(s)':<10}")
        print("-" * 70)

        for model_type, result in results.items():
            print(
                f"{model_type:<20} "
                f"{result.test_metrics['mae']:<10.4f} "
                f"{result.test_metrics['rmse']:<10.4f} "
                f"{result.test_metrics['r2']:<10.4f} "
                f"{result.training_time:<10.1f}"
            )

        return results

    def save_results(self, filename: str = "experiment_results.json") -> None:
        """Save all results to JSON."""
        results_data = []

        for result in self.results:
            # Convert numpy floats to Python floats for JSON serialization
            test_metrics = {k: float(v) for k, v in result.test_metrics.items()}
            results_data.append(
                {
                    "name": result.config.name,
                    "model_type": result.config.model_type,
                    "target_metric": result.config.target_metric,
                    "train_loss": float(result.train_loss),
                    "val_loss": float(result.val_loss),
                    "test_metrics": test_metrics,
                    "training_time": float(result.training_time),
                    "best_epoch": result.best_epoch,
                    "model_path": result.model_path,
                }
            )

        with open(self.output_dir / filename, "w") as f:
            json.dump(results_data, f, indent=2)


# =============================================================================
# Optuna Hyperparameter Optimization
# =============================================================================


class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """

    def __init__(
        self,
        experiment_runner: ExperimentRunner,
        model_type: str,
        target_metric: str = "RESTING_HEART_RATE",
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.runner = experiment_runner
        self.model_type = model_type
        self.target_metric = target_metric

    def create_objective(
        self,
        n_epochs: int = 50,
    ) -> Callable[[optuna.Trial], float]:
        """Create Optuna objective function."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hyperparams = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
            }

            config = ExperimentConfig(
                name=f"optuna_trial_{trial.number}",
                model_type=self.model_type,
                target_metric=self.target_metric,
                epochs=n_epochs,
                early_stopping_patience=7,
            )

            try:
                result = self.runner.run_experiment(
                    config=config,
                    hyperparams=hyperparams,
                    verbose=False,
                )
                return result.test_metrics["mae"]
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return float("in")

        return objective

    def optimize(
        self,
        n_trials: int = 50,
        n_epochs: int = 50,
        study_name: Optional[str] = None,
    ) -> HyperparameterSearchResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of Optuna trials
            n_epochs: Epochs per trial
            study_name: Name for the study

        Returns:
            HyperparameterSearchResult with best parameters
        """
        if study_name is None:
            study_name = f"{self.model_type}_{self.target_metric}_optimization"

        print(f"\n{'='*60}")
        print("Starting Optuna Optimization")
        print(f"Model: {self.model_type}")
        print(f"Target: {self.target_metric}")
        print(f"Trials: {n_trials}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

        # Run optimization
        objective = self.create_objective(n_epochs=n_epochs)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        optimization_time = time.time() - start_time

        # Collect results
        all_trials = []
        for trial in study.trials:
            all_trials.append(
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": str(trial.state),
                }
            )

        result = HyperparameterSearchResult(
            best_params=study.best_params,
            best_value=study.best_value,
            all_trials=all_trials,
            study_name=study_name,
            n_trials=n_trials,
            optimization_time=optimization_time,
        )

        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"Best MAE: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print(f"Time: {optimization_time:.1f}s")
        print(f"{'='*60}\n")

        return result


# =============================================================================
# Main Entry Point
# =============================================================================


def run_full_experiment_suite(
    output_dir: str = "ml-service/experiments",
    n_optuna_trials: int = 30,
) -> None:
    """
    Run the complete experiment suite:
    1. Compare all three models
    2. Optimize the best performer
    3. Final evaluation
    """
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT SUITE")
    print("=" * 70 + "\n")

    runner = ExperimentRunner(output_dir=output_dir)

    # Phase 1: Model Comparison for RHR
    print("\n[Phase 1] Model Comparison - RHR Prediction")
    rhr_results = runner.run_model_comparison(
        target_metric="RESTING_HEART_RATE",
        sequence_length=30,
    )

    # Phase 2: Model Comparison for HRV
    print("\n[Phase 2] Model Comparison - HRV Prediction")
    hrv_results = runner.run_model_comparison(
        target_metric="HEART_RATE_VARIABILITY_RMSSD",
        sequence_length=30,
    )

    # Find best model
    _ = {**rhr_results, **hrv_results}
    best_model = min(
        rhr_results.keys(), key=lambda k: rhr_results[k].test_metrics["mae"]
    )

    print(f"\n[Phase 3] Optimizing Best Model: {best_model}")

    if OPTUNA_AVAILABLE:
        optimizer = OptunaOptimizer(
            experiment_runner=runner,
            model_type=best_model,
            target_metric="RESTING_HEART_RATE",
        )

        optuna_result = optimizer.optimize(
            n_trials=n_optuna_trials,
            n_epochs=50,
        )

        # Run final experiment with best parameters
        print("\n[Phase 4] Final Training with Best Parameters")
        final_config = ExperimentConfig(
            name="final_optimized_model",
            model_type=best_model,
            target_metric="RESTING_HEART_RATE",
            epochs=100,
        )

        _ = runner.run_experiment(
            config=final_config,
            hyperparams=optuna_result.best_params,
        )

    # Save all results
    runner.save_results()

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Quick test
    print("Testing Experiment Framework...")

    runner = ExperimentRunner(output_dir="test_experiments")

    # Test single experiment
    config = ExperimentConfig(
        name="test_lstm_attention",
        model_type="lstm_attention",
        target_metric="RESTING_HEART_RATE",
        epochs=10,  # Quick test
    )

    result = runner.run_experiment(config)
    print(f"\nTest complete! MAE: {result.test_metrics['mae']:.4f}")
