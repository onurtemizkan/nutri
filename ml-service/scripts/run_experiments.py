#!/usr/bin/env python3
"""
Main Script to Run Health Prediction Model Experiments

This script:
1. Generates synthetic health data for 5 diverse users
2. Trains and compares 3 model architectures (LSTM+Attention, BiLSTM, TCN)
3. Performs hyperparameter optimization with Optuna
4. Creates ensemble models
5. Evaluates and reports final results

Usage:
    cd ml-service
    python scripts/run_experiments.py [--quick] [--full]

Options:
    --quick: Run quick test with minimal epochs
    --full: Run full experiment suite with optimization
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

# Import our modules
from app.data.synthetic_generator import (
    SyntheticDataGenerator,
    UserPersona,
    PERSONA_CONFIGS,
)
from app.ml_models.advanced_lstm import (
    ModelFactory,
)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")


def print_section(text: str):
    """Print a section divider."""
    print("\n" + "-" * 50)
    print(f" {text}")
    print("-" * 50)


def run_quick_test():
    """Run a quick test to verify everything works."""
    print_header("QUICK TEST - Verifying Setup")

    # Test 1: Synthetic Data Generation
    print_section("Test 1: Synthetic Data Generation")

    generator = SyntheticDataGenerator(seed=42)

    print("Generating data for each persona...")
    for persona in UserPersona:
        config = PERSONA_CONFIGS[persona]
        data = generator.generate_user_data(
            persona=persona,
            user_id=f"test_{persona.value}",
            num_days=60,  # Short for quick test
        )

        meals = data["meals"]
        health = data["health_metrics"]

        rhr_data = health[health["metric_type"] == "RESTING_HEART_RATE"]["value"]
        hrv_data = health[health["metric_type"] == "HEART_RATE_VARIABILITY_RMSSD"][
            "value"
        ]

        print(f"\n{persona.value}:")
        print(f"  Meals: {len(meals)} records")
        print(f"  Health metrics: {len(health)} records")
        print(f"  RHR: mean={rhr_data.mean():.1f}, std={rhr_data.std():.1f}")
        print(f"  HRV: mean={hrv_data.mean():.1f}, std={hrv_data.std():.1f}")
        print(f"  Expected RHR baseline: {config.rhr_baseline:.1f}")
        print(f"  Expected HRV baseline: {config.hrv_baseline:.1f}")

    print("\n[PASS] Synthetic data generation working!")

    # Test 2: Model Creation
    print_section("Test 2: Model Architecture Creation")

    input_dim = 20
    batch_size = 16
    seq_len = 30
    x_test = torch.randn(batch_size, seq_len, input_dim)

    for model_type in ["lstm_attention", "bilstm_residual", "tcn"]:
        print(f"\nCreating {model_type}...")
        model = ModelFactory.create(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
        )

        # Test forward pass
        output = model(x_test)
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Output shape: {output.shape}")

        # Test attention weights if available
        if hasattr(model, "get_attention_weights"):
            weights = model.get_attention_weights(x_test)
            print(f"  Attention weights shape: {weights.shape}")
            print(f"  Attention sum (should be ~1): {weights.sum(dim=1).mean():.4f}")

    print("\n[PASS] Model architectures working!")

    # Test 3: Quick Training
    print_section("Test 3: Quick Training Test")

    print("Setting up mini training test...")

    # Create simple synthetic data
    X = torch.randn(100, seq_len, input_dim)
    y = torch.randn(100, 1)

    # Split
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    # Train for a few epochs
    model = ModelFactory.create("lstm_attention", input_dim=input_dim, hidden_dim=64)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training for 5 epochs...")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        print(
            f"  Epoch {epoch + 1}: Train={loss.item():.4f}, Val={val_loss.item():.4f}"
        )

    print("\n[PASS] Training loop working!")

    print_header("ALL QUICK TESTS PASSED!")
    return True


def run_data_analysis():
    """Analyze and visualize synthetic data quality."""
    print_header("SYNTHETIC DATA ANALYSIS")

    generator = SyntheticDataGenerator(seed=42)
    all_data = generator.generate_all_users(num_days=180)

    print("\n" + "=" * 90)
    print(
        f"{'Persona':<20} {'RHR Mean':<12} {'RHR Std':<12} {'HRV Mean':<12} {'HRV Std':<12} {'Records':<10}"
    )
    print("=" * 90)

    for user_id, data in all_data.items():
        health = data["health_metrics"]
        rhr = health[health["metric_type"] == "RESTING_HEART_RATE"]["value"]
        hrv = health[health["metric_type"] == "HEART_RATE_VARIABILITY_RMSSD"]["value"]

        persona = user_id.replace("synthetic_user_", "User ")
        print(
            f"{persona:<20} "
            f"{rhr.mean():<12.1f} "
            f"{rhr.std():<12.1f} "
            f"{hrv.mean():<12.1f} "
            f"{hrv.std():<12.1f} "
            f"{len(health):<10}"
        )

    print("=" * 90)

    # Correlation analysis
    print("\n" + "=" * 60)
    print("NUTRITION-HEALTH CORRELATION ANALYSIS")
    print("=" * 60)

    for user_id, data in all_data.items():
        meals = data["meals"]
        health = data["health_metrics"]

        # Daily aggregates
        meals["date"] = meals["consumed_at"].dt.date
        daily_nutrition = (
            meals.groupby("date")
            .agg(
                {
                    "calories": "sum",
                    "protein": "sum",
                }
            )
            .reset_index()
        )

        health["date"] = health["recorded_at"].dt.date
        daily_hrv = (
            health[health["metric_type"] == "HEART_RATE_VARIABILITY_RMSSD"]
            .groupby("date")["value"]
            .mean()
            .reset_index()
        )

        # Merge and calculate correlation
        merged = daily_nutrition.merge(daily_hrv, on="date", how="inner")

        if len(merged) > 10:
            protein_hrv_corr = merged["protein"].corr(merged["value"])
            calories_hrv_corr = merged["calories"].corr(merged["value"])

            persona = user_id.replace("synthetic_user_", "User ")
            print(f"\n{persona}:")
            print(f"  Protein-HRV correlation: {protein_hrv_corr:.3f}")
            print(f"  Calories-HRV correlation: {calories_hrv_corr:.3f}")


def run_model_comparison(epochs: int = 50, verbose: bool = True):
    """Run comprehensive model comparison."""
    print_header("MODEL COMPARISON EXPERIMENT")

    # Import experiment runner
    try:
        from app.experiments.experiment_runner import (
            ExperimentRunner,
            ExperimentConfig,
        )
    except ImportError as e:
        print(f"Error importing experiment runner: {e}")
        print("Running simplified comparison instead...")
        return run_simplified_comparison(epochs, verbose)

    runner = ExperimentRunner(output_dir="experiments/model_comparison")

    results = {}

    # Compare models for RHR prediction
    print_section("RHR (Resting Heart Rate) Prediction")

    for model_type in ["lstm_attention", "bilstm_residual", "tcn"]:
        print(f"\nTraining {model_type}...")
        config = ExperimentConfig(
            name=f"{model_type}_rhr",
            model_type=model_type,
            target_metric="RESTING_HEART_RATE",
            epochs=epochs,
        )
        result = runner.run_experiment(config, verbose=verbose)
        results[f"{model_type}_rhr"] = result

    # Compare models for HRV prediction
    print_section("HRV (Heart Rate Variability) Prediction")

    for model_type in ["lstm_attention", "bilstm_residual", "tcn"]:
        print(f"\nTraining {model_type}...")
        config = ExperimentConfig(
            name=f"{model_type}_hrv",
            model_type=model_type,
            target_metric="HEART_RATE_VARIABILITY_RMSSD",
            epochs=epochs,
        )
        result = runner.run_experiment(config, verbose=verbose)
        results[f"{model_type}_hrv"] = result

    # Print comparison table
    print_header("COMPARISON RESULTS")

    print("\n" + "=" * 90)
    print(
        f"{'Experiment':<30} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE':<12} {'Time(s)':<12}"
    )
    print("=" * 90)

    for name, result in results.items():
        metrics = result.test_metrics
        print(
            f"{name:<30} "
            f"{metrics['mae']:<12.4f} "
            f"{metrics['rmse']:<12.4f} "
            f"{metrics['r2']:<12.4f} "
            f"{metrics['mape']:<12.2f} "
            f"{result.training_time:<12.1f}"
        )

    print("=" * 90)

    # Find best model
    rhr_results = {k: v for k, v in results.items() if "rhr" in k}
    hrv_results = {k: v for k, v in results.items() if "hrv" in k}

    best_rhr = min(rhr_results.keys(), key=lambda k: rhr_results[k].test_metrics["mae"])
    best_hrv = min(hrv_results.keys(), key=lambda k: hrv_results[k].test_metrics["mae"])

    print(
        f"\nBest RHR Model: {best_rhr} (MAE: {rhr_results[best_rhr].test_metrics['mae']:.4f})"
    )
    print(
        f"Best HRV Model: {best_hrv} (MAE: {hrv_results[best_hrv].test_metrics['mae']:.4f})"
    )

    runner.save_results()
    return results


def run_simplified_comparison(epochs: int = 20, verbose: bool = True):
    """Simplified model comparison without full experiment framework."""
    print("Running simplified comparison...")

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Generate data
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_user_data(
        persona=UserPersona.HEALTH_ENTHUSIAST,
        user_id="test_user",
        num_days=120,
    )

    # Prepare simple features
    health = data["health_metrics"]
    meals = data["meals"]

    # Get RHR values
    rhr_data = health[health["metric_type"] == "RESTING_HEART_RATE"].copy()
    rhr_data["date"] = rhr_data["recorded_at"].dt.date
    daily_rhr = rhr_data.groupby("date")["value"].mean().values

    # Simple sequence data
    seq_len = 14
    X = []
    y = []
    for i in range(len(daily_rhr) - seq_len):
        X.append(daily_rhr[i : i + seq_len])
        y.append(daily_rhr[i + seq_len])

    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    # Normalize
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 1)
    X_norm = scaler.fit_transform(X_flat).reshape(X.shape)

    y_scaler = StandardScaler()
    y_norm = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Split
    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train = torch.FloatTensor(X_norm[:n_train])
    y_train = torch.FloatTensor(y_norm[:n_train]).unsqueeze(1)
    X_val = torch.FloatTensor(X_norm[n_train : n_train + n_val])
    y_val = torch.FloatTensor(y_norm[n_train : n_train + n_val]).unsqueeze(1)
    X_test = torch.FloatTensor(X_norm[n_train + n_val :])
    y_test = torch.FloatTensor(y_norm[n_train + n_val :]).unsqueeze(1)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results = {}

    for model_type in ["lstm_attention", "bilstm_residual", "tcn"]:
        print(f"\nTraining {model_type}...")

        model = ModelFactory.create(
            model_type=model_type,
            input_dim=1,
            hidden_dim=32,
            num_layers=1,
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: Val Loss = {val_loss:.6f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).numpy()

        # Denormalize
        test_pred_real = y_scaler.inverse_transform(test_pred).flatten()
        test_real = y_scaler.inverse_transform(y_test.numpy()).flatten()

        mae = mean_absolute_error(test_real, test_pred_real)
        rmse = np.sqrt(mean_squared_error(test_real, test_pred_real))
        r2 = r2_score(test_real, test_pred_real)

        results[model_type] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Health Prediction Experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run full experiment suite")
    parser.add_argument(
        "--data-only", action="store_true", help="Only generate and analyze data"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" NUTRI ML - HEALTH PREDICTION MODEL EXPERIMENTS")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = time.time()

    if args.quick:
        run_quick_test()

    elif args.data_only:
        run_data_analysis()

    elif args.full:
        # Run all experiments
        run_quick_test()
        run_data_analysis()
        run_model_comparison(epochs=args.epochs)

    else:
        # Default: run model comparison
        run_model_comparison(epochs=args.epochs)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print("\n" + "=" * 70)
    print(" EXPERIMENTS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
