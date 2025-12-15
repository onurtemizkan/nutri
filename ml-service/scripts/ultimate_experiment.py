#!/usr/bin/env python3
"""
ULTIMATE EXPERIMENT RUNNER

This script performs exhaustive experimentation to find the best possible
model architecture for health prediction. Features:

1. Ultra-realistic synthetic data generation (365 days, 10 users)
2. Comprehensive baseline evaluation
3. Custom hybrid architecture development
4. Extensive hyperparameter optimization
5. Advanced ensemble methods
6. Brutal final evaluation

Run with: python scripts/ultimate_experiment.py
"""

import gc
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.synthetic_generator import (
    SyntheticDataGenerator,
    UserPersona,
    PERSONA_CONFIGS,
    PersonaConfig,
)
from app.ml_models.advanced_lstm import (
    AdvancedLSTMConfig,
    EnhancedLSTMWithAttention,
    BiLSTMWithResiduals,
    TemporalConvNet,
    ModelFactory,
    TCNConfig,
    TemporalAttention,
    MultiHeadTemporalAttention,
)

# Try importing optuna
try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")


# =============================================================================
# ENHANCED SYNTHETIC DATA GENERATOR
# =============================================================================

class EnhancedSyntheticGenerator(SyntheticDataGenerator):
    """
    Enhanced generator with more realistic patterns:
    - Seasonal variations (winter/summer)
    - Illness episodes
    - Training adaptation over time
    - Stress cycles
    - More complex nutrition-health relationships
    """

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.illness_probability = 0.02  # 2% chance per day
        self.stress_cycle_days = 7  # Weekly stress patterns

    def _generate_daily_health_metrics(
        self,
        config: PersonaConfig,
        user_id: str,
        current_date: date,
        state: Dict,
        daily_meals: List[Dict],
        daily_activities: List[Dict],
    ) -> List[Dict]:
        """Enhanced health metrics with more realistic variations."""
        metrics = []
        is_weekend = current_date.weekday() >= 5
        day_of_year = current_date.timetuple().tm_yday

        # Seasonal variation (RHR slightly higher in winter)
        seasonal_factor = np.sin(2 * np.pi * day_of_year / 365) * 2

        # Illness check
        is_sick = np.random.random() < self.illness_probability
        if is_sick:
            state["sick_days"] = state.get("sick_days", 0) + 3

        recovering_from_illness = state.get("sick_days", 0) > 0
        if recovering_from_illness:
            state["sick_days"] -= 1

        # Calculate nutrition quality
        total_calories = sum(m["calories"] for m in daily_meals)
        total_protein = sum(m["protein"] for m in daily_meals)
        total_carbs = sum(m["carbs"] for m in daily_meals)

        protein_adequacy = min(1.0, total_protein / (config.protein_target * config.weight_kg))
        calorie_deviation = abs(total_calories - config.calories_target) / config.calories_target

        # Late eating impact
        late_meals = [m for m in daily_meals if m["consumed_at"].hour >= 20]
        late_eating_impact = len(late_meals) * 0.2

        # Activity load
        high_intensity_mins = sum(
            a["duration"] for a in daily_activities if a.get("intensity") == "HIGH"
        )
        total_activity = sum(a["duration"] for a in daily_activities)

        # Weekly stress cycle
        stress_cycle = np.sin(2 * np.pi * current_date.weekday() / 7) * 0.2

        # 7-day rolling averages
        avg_nutrition = np.mean(state.get("nutrition_quality_7d", [0.5])[-7:])
        avg_activity = np.mean(state.get("activity_load_7d", [0])[-7:])

        # ========== RHR Calculation (Enhanced) ==========
        rhr_base = config.rhr_baseline

        rhr_modifiers = [
            # Seasonal
            seasonal_factor,
            # Sleep debt (bigger impact)
            state.get("sleep_debt", 0) * 3,
            # Yesterday's high activity
            (state.get("activity_load_7d", [0])[-1] if state.get("activity_load_7d") else 0) * 0.03,
            # Poor nutrition
            (1 - avg_nutrition) * 4,
            # Cumulative fatigue
            state.get("cumulative_fatigue", 0) * 3,
            # Base stress
            config.stress_level * 4,
            # Stress cycle
            stress_cycle * 3,
            # Weekend relaxation
            -2 if is_weekend else 0,
            # Illness
            8 if recovering_from_illness else 0,
            # Training adaptation (fitness improves over time)
            -min(5, avg_activity * 0.01),
        ]

        # Apply with correlation strength
        total_modifier = sum(rhr_modifiers) * config.nutrition_health_correlation
        rhr_value = rhr_base + total_modifier + np.random.normal(0, config.rhr_std)
        rhr_value = max(40, min(110, rhr_value))

        # ========== HRV Calculation (Enhanced) ==========
        hrv_base = config.hrv_baseline

        hrv_modifiers = [
            # Protein benefit (strong correlation)
            protein_adequacy * 10,
            # Sleep debt (major impact)
            -state.get("sleep_debt", 0) * 6,
            # Late eating
            -late_eating_impact * 8,
            # Fitness benefit
            min(8, avg_activity * 0.015),
            # Overtraining penalty
            -(state.get("cumulative_fatigue", 0) - 0.3) * 20 if state.get("cumulative_fatigue", 0) > 0.3 else 0,
            # Stress (big impact on HRV)
            -config.stress_level * 15,
            # Stress cycle
            -stress_cycle * 5,
            # Weekend recovery
            4 if is_weekend else 0,
            # Illness
            -15 if recovering_from_illness else 0,
            # Seasonal (HRV slightly lower in winter)
            -seasonal_factor * 0.5,
            # Calorie deviation penalty
            -calorie_deviation * 5,
            # Carb timing (carbs before bed hurt HRV)
            -late_eating_impact * 3 if any(m["carbs"] > 30 for m in late_meals) else 0,
        ]

        total_hrv_modifier = sum(hrv_modifiers) * config.nutrition_health_correlation
        hrv_value = hrv_base + total_hrv_modifier + np.random.normal(0, config.hrv_std)
        hrv_value = max(10, min(150, hrv_value))

        # ========== Sleep Calculation (Enhanced) ==========
        sleep_base = config.sleep_baseline

        sleep_modifiers = [
            -late_eating_impact * 0.4,
            0.2 if 30 < high_intensity_mins < 60 else (-0.3 if high_intensity_mins > 90 else 0),
            np.random.uniform(0.5, 1.5) if is_weekend and config.sleep_regularity < 0.7 else 0,
            -config.stress_level * 0.8,
            -0.5 if recovering_from_illness else 0,
            # Caffeine proxy (late meals often include caffeine)
            -0.3 if any(m["consumed_at"].hour >= 18 for m in daily_meals) else 0,
        ]

        sleep_duration = sleep_base + sum(sleep_modifiers) + np.random.normal(0, config.sleep_std)
        sleep_duration = max(3, min(12, sleep_duration))

        # Sleep quality
        sleep_quality = 70 + (sleep_duration - 6) * 5 - late_eating_impact * 8 - config.stress_level * 12
        sleep_quality += 10 if protein_adequacy > 0.8 else 0  # Good protein helps sleep
        sleep_quality = max(20, min(100, sleep_quality + np.random.normal(0, 6)))

        # ========== Recovery Score (Composite) ==========
        rhr_component = max(0, (90 - rhr_value) / 50 * 35)
        hrv_component = min(40, hrv_value / config.hrv_baseline * 30)
        sleep_component = sleep_quality * 0.25

        recovery_score = rhr_component + hrv_component + sleep_component
        recovery_score = max(0, min(100, recovery_score + np.random.normal(0, 4)))

        # Record timestamp
        recorded_at = datetime.combine(current_date, datetime.min.time()).replace(hour=7, minute=0)

        # Add all metrics
        metrics.append({
            "user_id": user_id,
            "metric_type": "RESTING_HEART_RATE",
            "value": round(rhr_value, 1),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        metrics.append({
            "user_id": user_id,
            "metric_type": "HEART_RATE_VARIABILITY_RMSSD",
            "value": round(hrv_value, 1),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        metrics.append({
            "user_id": user_id,
            "metric_type": "HEART_RATE_VARIABILITY_SDNN",
            "value": round(hrv_value * 1.15 + np.random.normal(0, 3), 1),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        metrics.append({
            "user_id": user_id,
            "metric_type": "SLEEP_DURATION",
            "value": round(sleep_duration, 2),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        metrics.append({
            "user_id": user_id,
            "metric_type": "SLEEP_SCORE",
            "value": round(sleep_quality, 1),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        metrics.append({
            "user_id": user_id,
            "metric_type": "RECOVERY_SCORE",
            "value": round(recovery_score, 1),
            "source": "SYNTHETIC_ENHANCED",
            "recorded_at": recorded_at,
        })

        return metrics


# =============================================================================
# CUSTOM HYBRID ARCHITECTURES
# =============================================================================

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from Temporal Fusion Transformer.
    Provides flexible nonlinear processing with gating.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Gating layer
        self.gate = nn.Linear(hidden_dim, output_dim)

        # Skip connection
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        hidden = self.elu(self.fc1(x))
        hidden = self.dropout(hidden)

        # Output with gating
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        gated_output = gate * output

        # Residual connection
        skip = self.skip(x)
        return self.layer_norm(skip + gated_output)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network from Temporal Fusion Transformer.
    Learns which input features are most important.
    """

    def __init__(self, input_dim: int, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.num_features = num_features

        # GRN for each feature
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim // num_features, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

        # Softmax weights for variable selection
        self.flattened_grn = GatedResidualNetwork(input_dim, hidden_dim, num_features, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape

        # Split into features
        feature_dim = x.shape[-1] // self.num_features
        features = x.view(batch, seq_len, self.num_features, feature_dim)

        # Process each feature
        processed = []
        for i, grn in enumerate(self.grns):
            feat = features[:, :, i, :]  # (batch, seq_len, feature_dim)
            processed.append(grn(feat))

        processed = torch.stack(processed, dim=2)  # (batch, seq_len, num_features, hidden)

        # Variable selection weights
        weights = F.softmax(self.flattened_grn(x.view(batch * seq_len, -1)).view(batch, seq_len, -1), dim=-1)

        # Weighted combination
        output = (processed * weights.unsqueeze(-1)).sum(dim=2)

        return output, weights


class UltimateHealthPredictor(nn.Module):
    """
    ULTIMATE STATE-OF-THE-ART ARCHITECTURE

    Combines the best of all worlds:
    1. TCN for efficient long-range dependency capture
    2. LSTM for sequential pattern learning
    3. Multi-head attention for interpretability
    4. Gated Residual Networks for flexible nonlinearity
    5. Variable selection for feature importance
    6. Skip connections for gradient flow

    This is our custom hybrid architecture designed specifically
    for health metric prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        sequence_length: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection with GRN
        self.input_grn = GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)

        # TCN branch for long-range patterns
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                             padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        # LSTM branch for sequential patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Multi-head attention for interpretability
        self.attention = MultiHeadTemporalAttention(hidden_dim, num_heads, dropout)

        # Gating mechanism to combine TCN and LSTM
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Output GRN
        self.output_grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim // 2, dropout)

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the ultimate architecture.

        Args:
            x: (batch, seq_len, input_dim)
            return_attention: Return attention weights for interpretability

        Returns:
            predictions: (batch, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x_proj = self.input_grn(x)  # (batch, seq_len, hidden_dim)

        # TCN branch
        tcn_out = x_proj.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        for tcn_layer in self.tcn_layers:
            residual = tcn_out
            tcn_out = tcn_layer(tcn_out)
            tcn_out = tcn_out + residual  # Residual connection
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq_len, hidden_dim)

        # LSTM branch
        lstm_out, _ = self.lstm(x_proj)  # (batch, seq_len, hidden_dim)
        lstm_out = self.layer_norm(lstm_out)

        # Attention over LSTM output
        attn_out, attn_weights = self.attention(lstm_out)  # (batch, hidden_dim)

        # Gated combination of branches
        tcn_final = tcn_out[:, -1, :]  # Take last timestep
        combined = torch.cat([tcn_final, attn_out], dim=-1)

        gate = self.branch_gate(combined)
        gated_tcn = gate * tcn_final
        gated_attn = (1 - gate) * attn_out

        # Combine gated outputs
        merged = torch.cat([gated_tcn, gated_attn], dim=-1)

        # Output processing
        out = self.output_grn(merged)
        out = self.dropout(out)
        out = F.gelu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        if return_attention:
            return out, attn_weights
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, weights = self.forward(x, return_attention=True)
        return weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightHybrid(nn.Module):
    """
    Lightweight hybrid for faster training and inference.
    Uses efficient techniques while maintaining performance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Efficient input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Depthwise separable TCN (more efficient)
        self.tcn = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn.append(nn.Sequential(
                # Depthwise
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation,
                         dilation=dilation, groups=hidden_dim),
                # Pointwise
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Lightweight attention (single head)
        self.attention = TemporalAttention(hidden_dim, hidden_dim // 2)

        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed
        x = self.input_embed(x)

        # TCN
        x_tcn = x.transpose(1, 2)
        for layer in self.tcn:
            residual = x_tcn
            x_tcn = layer(x_tcn) + residual
        x_tcn = x_tcn.transpose(1, 2)

        # Attention pooling
        context, _ = self.attention(x_tcn)

        return self.output(context)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class CosineAnnealingWarmRestarts:
    """Learning rate scheduler with warm restarts."""

    def __init__(self, optimizer, T_0, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * \
                (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2


class FocalMSELoss(nn.Module):
    """Focal loss for regression - focuses on hard examples."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        # Weight by error magnitude
        weights = torch.abs(pred - target) ** self.gamma
        return (weights * mse).mean()


class HuberLoss(nn.Module):
    """Smooth L1 loss - robust to outliers."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        return torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        ).mean()


def train_model_advanced(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: torch.device = None,
    use_mixed_precision: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Advanced training with:
    - Cosine annealing with warm restarts
    - Gradient clipping
    - Mixed precision (optional)
    - Multiple loss functions
    - Comprehensive logging
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Loss functions
    mse_loss = nn.MSELoss()
    huber_loss = HuberLoss(delta=1.0)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            # Combined loss
            loss = 0.7 * mse_loss(predictions, y_batch) + 0.3 * huber_loss(predictions, y_batch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_d = X_val.to(device)
            y_val_d = y_val.to(device)
            val_pred = model(X_val_d)
            val_loss = mse_loss(val_pred, y_val_d).item()
            val_mae = torch.abs(val_pred - y_val_d).mean().item()

        train_loss = np.mean(train_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["learning_rate"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"MAE={val_mae:.6f}, LR={current_lr:.2e}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": epoch + 1,
        "history": history,
    }


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    scaler: StandardScaler,
    device: torch.device = None,
) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_d = X.to(device)
        predictions = model(X_d).cpu().numpy()

    # Denormalize
    y_true = scaler.inverse_transform(y.numpy()).flatten()
    y_pred = scaler.inverse_transform(predictions).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0

    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))

    # Percentile errors
    errors = np.abs(y_true - y_pred)
    p90_error = np.percentile(errors, 90)
    p95_error = np.percentile(errors, 95)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "max_error": float(max_error),
        "median_ae": float(median_ae),
        "p90_error": float(p90_error),
        "p95_error": float(p95_error),
    }


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def print_header(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")


def print_section(text: str):
    print("\n" + "-" * 60)
    print(f" {text}")
    print("-" * 60)


def prepare_data(
    generator: EnhancedSyntheticGenerator,
    num_days: int = 365,
    sequence_length: int = 30,
    target_metric: str = "RESTING_HEART_RATE",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data from all personas."""
    all_X = []
    all_y = []

    for persona in UserPersona:
        user_id = f"ultimate_{persona.value}"
        print(f"  Generating data for {persona.value}...")

        data = generator.generate_user_data(
            persona=persona,
            user_id=user_id,
            num_days=num_days,
        )

        health = data["health_metrics"]
        meals = data["meals"]
        activities = data["activities"]

        # Build daily feature vectors
        target_df = health[health["metric_type"] == target_metric].copy()
        target_df["date"] = target_df["recorded_at"].dt.date
        daily_targets = target_df.groupby("date")["value"].mean()

        # Build features
        features = {}
        for current_date in sorted(daily_targets.index):
            day_features = {}

            # Nutrition
            day_meals = meals[meals["consumed_at"].dt.date == current_date]
            day_features["calories"] = day_meals["calories"].sum()
            day_features["protein"] = day_meals["protein"].sum()
            day_features["carbs"] = day_meals["carbs"].sum()
            day_features["fat"] = day_meals["fat"].sum()
            day_features["meal_count"] = len(day_meals)

            late_meals = day_meals[day_meals["consumed_at"].dt.hour >= 20]
            day_features["late_calories"] = late_meals["calories"].sum()
            day_features["late_carbs"] = late_meals["carbs"].sum()

            # Activity
            day_activities = activities[activities["started_at"].dt.date == current_date]
            day_features["active_minutes"] = day_activities["duration"].sum()
            day_features["calories_burned"] = day_activities["calories_burned"].sum()

            high_intensity = day_activities[day_activities["intensity"] == "HIGH"]
            day_features["high_intensity_mins"] = high_intensity["duration"].sum()

            # Health metrics (lagged)
            for mt in ["RESTING_HEART_RATE", "HEART_RATE_VARIABILITY_RMSSD",
                      "SLEEP_DURATION", "SLEEP_SCORE", "RECOVERY_SCORE"]:
                metric_data = health[
                    (health["metric_type"] == mt) &
                    (health["recorded_at"].dt.date == current_date)
                ]
                day_features[f"{mt.lower()}_lag"] = metric_data["value"].iloc[0] if len(metric_data) > 0 else np.nan

            # Temporal
            dt = datetime.combine(current_date, datetime.min.time())
            day_features["day_of_week"] = dt.weekday()
            day_features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
            day_features["month"] = dt.month
            day_features["day_of_year"] = dt.timetuple().tm_yday

            features[current_date] = day_features

        # Convert to DataFrame
        features_df = pd.DataFrame.from_dict(features, orient="index")
        features_df = features_df.ffill().fillna(0)

        # Create sequences
        common_dates = sorted(set(features_df.index) & set(daily_targets.index))

        if len(common_dates) < sequence_length + 10:
            continue

        for i in range(len(common_dates) - sequence_length):
            seq_dates = common_dates[i:i + sequence_length]
            target_date = common_dates[i + sequence_length]

            X_seq = features_df.loc[seq_dates].values
            y_val = daily_targets.loc[target_date]

            all_X.append(X_seq)
            all_y.append(y_val)

    feature_names = list(features_df.columns)
    return np.array(all_X), np.array(all_y), feature_names


def run_ultimate_experiments():
    """Run the ultimate experiment suite."""
    print_header("ULTIMATE HEALTH PREDICTION EXPERIMENT")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================================================
    # PHASE 1: Generate Ultra-Realistic Data
    # ==========================================================================
    print_header("PHASE 1: GENERATING ULTRA-REALISTIC DATA")

    generator = EnhancedSyntheticGenerator(seed=42)

    print("Generating 365 days of data for 5 personas...")
    X_rhr, y_rhr, feature_names = prepare_data(
        generator,
        num_days=365,
        sequence_length=30,
        target_metric="RESTING_HEART_RATE",
    )

    print(f"\nRHR Dataset: {X_rhr.shape[0]} samples, {X_rhr.shape[2]} features")

    # Also prepare HRV data
    X_hrv, y_hrv, _ = prepare_data(
        generator,
        num_days=365,
        sequence_length=30,
        target_metric="HEART_RATE_VARIABILITY_RMSSD",
    )

    print(f"HRV Dataset: {X_hrv.shape[0]} samples")

    # Normalize
    num_samples, seq_len, num_features = X_rhr.shape

    scaler_X = RobustScaler()  # More robust to outliers
    X_rhr_flat = X_rhr.reshape(-1, num_features)
    X_rhr_norm = scaler_X.fit_transform(X_rhr_flat).reshape(num_samples, seq_len, num_features)

    scaler_y_rhr = StandardScaler()
    y_rhr_norm = scaler_y_rhr.fit_transform(y_rhr.reshape(-1, 1)).flatten()

    # Split (time-series aware)
    n_train = int(len(X_rhr_norm) * 0.7)
    n_val = int(len(X_rhr_norm) * 0.15)

    X_train = torch.FloatTensor(X_rhr_norm[:n_train])
    y_train = torch.FloatTensor(y_rhr_norm[:n_train]).unsqueeze(1)
    X_val = torch.FloatTensor(X_rhr_norm[n_train:n_train + n_val])
    y_val = torch.FloatTensor(y_rhr_norm[n_train:n_train + n_val]).unsqueeze(1)
    X_test = torch.FloatTensor(X_rhr_norm[n_train + n_val:])
    y_test = torch.FloatTensor(y_rhr_norm[n_train + n_val:]).unsqueeze(1)

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ==========================================================================
    # PHASE 2: Baseline Experiments
    # ==========================================================================
    print_header("PHASE 2: BASELINE MODEL EVALUATION")

    results = {}

    # Test existing architectures
    baseline_models = {
        "tcn": lambda: ModelFactory.create("tcn", input_dim=num_features, hidden_dim=64, num_layers=5),
        "lstm_attention": lambda: ModelFactory.create("lstm_attention", input_dim=num_features, hidden_dim=128, num_layers=2),
        "bilstm_residual": lambda: ModelFactory.create("bilstm_residual", input_dim=num_features, hidden_dim=128, num_layers=2),
    }

    for name, model_fn in baseline_models.items():
        print_section(f"Training {name}")

        model = model_fn()
        print(f"Parameters: {model.count_parameters():,}")

        train_result = train_model_advanced(
            model, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, learning_rate=0.001,
            patience=15, device=device, verbose=True
        )

        metrics = evaluate_model(model, X_test, y_test, scaler_y_rhr, device)
        results[name] = {
            "metrics": metrics,
            "training": train_result,
            "parameters": model.count_parameters(),
        }

        print(f"\n  Test MAE: {metrics['mae']:.4f} bpm")
        print(f"  Test RMSE: {metrics['rmse']:.4f} bpm")
        print(f"  Test R²: {metrics['r2']:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================================
    # PHASE 3: Custom Hybrid Architecture
    # ==========================================================================
    print_header("PHASE 3: ULTIMATE HYBRID ARCHITECTURE")

    print_section("Training UltimateHealthPredictor")

    ultimate_model = UltimateHealthPredictor(
        input_dim=num_features,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.2,
        sequence_length=30,
    )
    print(f"Ultimate Model Parameters: {ultimate_model.count_parameters():,}")

    train_result = train_model_advanced(
        ultimate_model, X_train, y_train, X_val, y_val,
        epochs=150, batch_size=32, learning_rate=0.0008,
        patience=20, device=device, verbose=True
    )

    metrics = evaluate_model(ultimate_model, X_test, y_test, scaler_y_rhr, device)
    results["ultimate_hybrid"] = {
        "metrics": metrics,
        "training": train_result,
        "parameters": ultimate_model.count_parameters(),
    }

    print(f"\n  Ultimate Model Test MAE: {metrics['mae']:.4f} bpm")
    print(f"  Ultimate Model Test RMSE: {metrics['rmse']:.4f} bpm")
    print(f"  Ultimate Model Test R²: {metrics['r2']:.4f}")

    # Lightweight hybrid
    print_section("Training LightweightHybrid")

    lightweight_model = LightweightHybrid(
        input_dim=num_features,
        hidden_dim=64,
        num_layers=3,
        dropout=0.15,
    )
    print(f"Lightweight Parameters: {lightweight_model.count_parameters():,}")

    train_result = train_model_advanced(
        lightweight_model, X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32, learning_rate=0.001,
        patience=15, device=device, verbose=True
    )

    metrics = evaluate_model(lightweight_model, X_test, y_test, scaler_y_rhr, device)
    results["lightweight_hybrid"] = {
        "metrics": metrics,
        "training": train_result,
        "parameters": lightweight_model.count_parameters(),
    }

    print(f"\n  Lightweight Test MAE: {metrics['mae']:.4f} bpm")

    # ==========================================================================
    # PHASE 4: Hyperparameter Optimization
    # ==========================================================================
    if OPTUNA_AVAILABLE:
        print_header("PHASE 4: OPTUNA HYPERPARAMETER OPTIMIZATION")

        def objective(trial):
            # Sample hyperparameters
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 160])
            num_layers = trial.suggest_int("num_layers", 2, 4)
            num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            model = UltimateHealthPredictor(
                input_dim=num_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )

            try:
                result = train_model_advanced(
                    model, X_train, y_train, X_val, y_val,
                    epochs=50, batch_size=batch_size, learning_rate=lr,
                    patience=10, device=device, verbose=False
                )
                return result["best_val_loss"]
            except Exception as e:
                return float("inf")

        print("Running 50 optimization trials...")
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10),
        )
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        print(f"\nBest trial:")
        print(f"  Value (Val Loss): {study.best_value:.6f}")
        print(f"  Params: {study.best_params}")

        # Train with best params
        print_section("Training with Optimized Hyperparameters")

        optimized_model = UltimateHealthPredictor(
            input_dim=num_features,
            hidden_dim=study.best_params["hidden_dim"],
            num_layers=study.best_params["num_layers"],
            num_heads=study.best_params["num_heads"],
            dropout=study.best_params["dropout"],
        )

        train_result = train_model_advanced(
            optimized_model, X_train, y_train, X_val, y_val,
            epochs=200, batch_size=study.best_params["batch_size"],
            learning_rate=study.best_params["lr"],
            patience=25, device=device, verbose=True
        )

        metrics = evaluate_model(optimized_model, X_test, y_test, scaler_y_rhr, device)
        results["optimized_ultimate"] = {
            "metrics": metrics,
            "training": train_result,
            "parameters": optimized_model.count_parameters(),
            "best_params": study.best_params,
        }

        print(f"\n  Optimized Model Test MAE: {metrics['mae']:.4f} bpm")
        print(f"  Optimized Model Test R²: {metrics['r2']:.4f}")

    # ==========================================================================
    # PHASE 5: Final Comparison
    # ==========================================================================
    print_header("PHASE 5: FINAL BRUTAL COMPARISON")

    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10} {'P90 Err':<10} {'Params':<12}")
    print("=" * 100)

    # Sort by MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"]["mae"])

    for name, data in sorted_results:
        m = data["metrics"]
        print(
            f"{name:<25} "
            f"{m['mae']:<10.4f} "
            f"{m['rmse']:<10.4f} "
            f"{m['r2']:<10.4f} "
            f"{m['mape']:<10.2f} "
            f"{m['p90_error']:<10.4f} "
            f"{data['parameters']:<12,}"
        )

    print("=" * 100)

    # Winner
    best_model = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best_model[0]}")
    print(f"   MAE: {best_model[1]['metrics']['mae']:.4f} bpm")
    print(f"   RMSE: {best_model[1]['metrics']['rmse']:.4f} bpm")
    print(f"   R²: {best_model[1]['metrics']['r2']:.4f}")

    # Improvement over baselines
    baseline_mae = results.get("tcn", results.get("lstm_attention", {})).get("metrics", {}).get("mae", 0)
    if baseline_mae > 0:
        improvement = (baseline_mae - best_model[1]['metrics']['mae']) / baseline_mae * 100
        print(f"   Improvement over baseline: {improvement:.1f}%")

    # Save results
    print_header("SAVING RESULTS")

    output_dir = Path("experiments/ultimate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_to_save = {}
    for name, data in results.items():
        metrics_to_save[name] = {
            "metrics": data["metrics"],
            "parameters": data["parameters"],
        }
        if "best_params" in data:
            metrics_to_save[name]["best_params"] = data["best_params"]

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"Results saved to {output_dir}")

    print_header("EXPERIMENT COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = run_ultimate_experiments()
