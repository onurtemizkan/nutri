"""
Ensemble Methods for Health Metric Prediction

Implements various ensemble strategies to combine multiple models
for improved prediction accuracy and robustness.

Ensemble methods implemented:
1. Simple Averaging
2. Weighted Averaging (learned weights)
3. Stacking (meta-learner)
4. Boosted Ensemble

Based on research:
- "ES-RNN: Hybrid Exponential Smoothing + RNN" (M4 Competition Winner)
- Ensemble learning best practices
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""

    models: List[nn.Module]
    model_names: List[str]
    ensemble_type: str = "weighted"  # 'simple', 'weighted', 'stacking'
    device: str = "cpu"


class SimpleAverageEnsemble(nn.Module):
    """
    Simple averaging ensemble - combines predictions by averaging.

    This is the simplest ensemble method but often surprisingly effective.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, 1)
        return stacked.mean(dim=0)  # (batch, 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble with learnable weights.

    Learns optimal combination weights for each model based on
    validation performance. More sophisticated than simple averaging.
    """

    def __init__(
        self,
        models: List[nn.Module],
        initial_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        # Learnable weights (will be softmaxed to ensure they sum to 1)
        if initial_weights is not None:
            init_logits = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            init_logits = torch.ones(self.n_models)

        self.weight_logits = nn.Parameter(init_logits)

    @property
    def weights(self) -> torch.Tensor:
        """Get normalized weights (sum to 1)."""
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted combination of model predictions."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, 1)

        # Apply weights
        weights = self.weights.view(-1, 1, 1)  # (n_models, 1, 1)
        weighted = stacked * weights

        return weighted.sum(dim=0)  # (batch, 1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_weights_dict(self) -> Dict[int, float]:
        """Get weights as a dictionary."""
        weights = self.weights.detach().cpu().numpy()
        return {i: float(w) for i, w in enumerate(weights)}

    def train_weights(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> List[float]:
        """
        Train only the ensemble weights (freeze individual models).

        Args:
            X_val: Validation features
            y_val: Validation targets
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Training loss history
        """
        # Freeze individual models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Only optimize weights
        optimizer = optim.Adam([self.weight_logits], lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(X_val)
            loss = criterion(predictions, y_val)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Unfreeze models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = True

        return losses


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner.

    Uses predictions from base models as features for a meta-learner
    that makes the final prediction.

    Architecture:
    Base Models -> Predictions -> Meta-Learner -> Final Prediction
    """

    def __init__(
        self,
        models: List[nn.Module],
        meta_hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(self.n_models, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim // 2, 1),
        )

        self._init_meta_weights()

    def _init_meta_weights(self):
        for module in self.meta_learner:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def get_base_predictions(
        self,
        x: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Get predictions from all base models."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad() if detach else torch.enable_grad():
                pred = model(x)
                predictions.append(pred)

        return torch.cat(predictions, dim=1)  # (batch, n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacking ensemble."""
        # Get base model predictions
        base_preds = self.get_base_predictions(x, detach=True)

        # Pass through meta-learner
        return self.meta_learner(base_preds)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def train_meta_learner(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> Tuple[List[float], List[float]]:
        """
        Train only the meta-learner (base models frozen).

        Returns:
            (train_losses, val_losses)
        """
        # Freeze base models
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        optimizer = optim.Adam(self.meta_learner.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Pre-compute base predictions for efficiency
        with torch.no_grad():
            train_base_preds = self.get_base_predictions(X_train, detach=True)
            val_base_preds = self.get_base_predictions(X_val, detach=True)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.meta_learner.train()

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(len(train_base_preds))
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_preds = train_base_preds[batch_idx]
                batch_targets = y_train[batch_idx]

                optimizer.zero_grad()
                output = self.meta_learner(batch_preds)
                loss = criterion(output, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / n_batches)

            # Validation
            self.meta_learner.eval()
            with torch.no_grad():
                val_output = self.meta_learner(val_base_preds)
                val_loss = criterion(val_output, y_val).item()
                val_losses.append(val_loss)

        return train_losses, val_losses


class DynamicEnsemble(nn.Module):
    """
    Dynamic ensemble that selects models based on input characteristics.

    Uses a gating network to dynamically weight models based on the
    input sequence, allowing different experts for different patterns.
    """

    def __init__(
        self,
        models: List[nn.Module],
        input_dim: int,
        seq_length: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        # Gating network: learns which model to use based on input
        self.gate = nn.Sequential(
            nn.Linear(input_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.n_models),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with dynamic model selection.

        Args:
            x: (batch, seq_len, input_dim)
            return_weights: If True, also return gating weights

        Returns:
            predictions, optionally with gating weights
        """
        batch_size = x.size(0)

        # Flatten input for gating network
        x_flat = x.view(batch_size, -1)

        # Get gating weights
        gate_weights = self.gate(x_flat)  # (batch, n_models)

        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=1)  # (batch, n_models, 1)

        # Weight predictions by gates
        weighted = stacked * gate_weights.unsqueeze(-1)  # (batch, n_models, 1)
        output = weighted.sum(dim=1)  # (batch, 1)

        if return_weights:
            return output, gate_weights
        return output

    def predict(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ):
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_weights)


# =============================================================================
# Ensemble Factory and Utilities
# =============================================================================


class EnsembleFactory:
    """Factory for creating ensemble models."""

    ENSEMBLES = {
        "simple": SimpleAverageEnsemble,
        "weighted": WeightedEnsemble,
        "stacking": StackingEnsemble,
        "dynamic": DynamicEnsemble,
    }

    @classmethod
    def create(
        cls,
        ensemble_type: str,
        models: List[nn.Module],
        **kwargs,
    ) -> nn.Module:
        """Create an ensemble model."""
        if ensemble_type not in cls.ENSEMBLES:
            raise ValueError(
                f"Unknown ensemble type: {ensemble_type}. "
                f"Choose from: {list(cls.ENSEMBLES.keys())}"
            )

        return cls.ENSEMBLES[ensemble_type](models, **kwargs)

    @classmethod
    def list_types(cls) -> List[str]:
        return list(cls.ENSEMBLES.keys())


def evaluate_ensemble(
    ensemble: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate ensemble performance."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    ensemble.eval()
    with torch.no_grad():
        predictions = ensemble.predict(X_test)

    y_pred = predictions.cpu().numpy().flatten()
    y_true = y_test.cpu().numpy().flatten()

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def create_best_ensemble(
    models: List[nn.Module],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Create and evaluate all ensemble types, return the best one.

    Returns:
        (best_ensemble, metrics)
    """
    best_ensemble = None
    best_mae = float("inf")
    best_metrics = {}

    for ensemble_type in ["simple", "weighted", "stacking"]:
        print(f"Testing {ensemble_type} ensemble...")

        if ensemble_type == "stacking":
            ensemble = EnsembleFactory.create(ensemble_type, models)
            ensemble.train_meta_learner(X_val, y_val, X_val, y_val, epochs=50, lr=0.001)
        elif ensemble_type == "weighted":
            ensemble = EnsembleFactory.create(ensemble_type, models)
            ensemble.train_weights(X_val, y_val, epochs=50)
        else:
            ensemble = EnsembleFactory.create(ensemble_type, models)

        metrics = evaluate_ensemble(ensemble, X_test, y_test)

        print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_ensemble = ensemble
            best_metrics = metrics

    print(f"\nBest ensemble: {type(best_ensemble).__name__}")
    return best_ensemble, best_metrics


if __name__ == "__main__":
    # Quick test
    from advanced_lstm import ModelFactory

    print("Testing Ensemble Methods...")

    # Create test models
    input_dim = 20
    models = [
        ModelFactory.create("lstm_attention", input_dim=input_dim, hidden_dim=64),
        ModelFactory.create("bilstm_residual", input_dim=input_dim, hidden_dim=64),
        ModelFactory.create("tcn", input_dim=input_dim, hidden_channels=64),
    ]

    # Test data
    batch_size = 32
    seq_len = 30
    X = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, 1)

    # Test each ensemble type
    for ens_type in EnsembleFactory.list_types():
        print(f"\nTesting {ens_type} ensemble:")

        if ens_type == "dynamic":
            ensemble = EnsembleFactory.create(
                ens_type, models, input_dim=input_dim, seq_length=seq_len
            )
        else:
            ensemble = EnsembleFactory.create(ens_type, models)

        output = ensemble(X)
        print(f"  Output shape: {output.shape}")
