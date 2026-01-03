"""
What-If Scenarios and Counterfactual Explanations

This service allows users to test hypothetical changes:
- "What if I ate 50g more protein?"
- "What if I did a high-intensity workout?"
- "What if I got 8 hours of sleep?"

It also generates counterfactual explanations:
- "To reach your target RHR, you should increase protein by 30g and reduce late-night calories by 200"
"""

import math
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml_models.lstm import HealthMetricLSTM
from app.schemas.interpretability import (
    WhatIfRequest,
    WhatIfResponse,
    WhatIfResult,
    WhatIfScenario,
    CounterfactualRequest,
    CounterfactualResponse,
    CounterfactualExplanation,
    CounterfactualChange,
    CounterfactualTarget,
    TrajectoryRequest,
    TrajectoryResponse,
    SimulationTrajectory,
    TrajectoryPoint,
    NutritionChange,
)
from app.schemas.predictions import PredictionMetric
from app.schemas.correlations import LagAnalysisRequest, HealthMetricTarget
from app.services.data_preparation import DataPreparationService
from app.services.correlation_engine import CorrelationEngineService


# ============================================================================
# Safety Guardrails - Nutrition Change Bounds
# ============================================================================

# Safe bounds for nutrition changes (absolute daily values)
NUTRITION_SAFE_BOUNDS = {
    "nutrition_protein_daily": {"min_delta": -100, "max_delta": 100, "unit": "g"},
    "nutrition_carbs_daily": {"min_delta": -200, "max_delta": 200, "unit": "g"},
    "nutrition_fat_daily": {"min_delta": -80, "max_delta": 80, "unit": "g"},
    "nutrition_fiber_daily": {"min_delta": -30, "max_delta": 30, "unit": "g"},
    "nutrition_sodium_daily": {"min_delta": -2000, "max_delta": 2000, "unit": "mg"},
    "nutrition_sugar_daily": {"min_delta": -100, "max_delta": 100, "unit": "g"},
    "nutrition_late_night_calories": {
        "min_delta": -500,
        "max_delta": 500,
        "unit": "kcal",
    },
    "nutrition_late_night_carbs": {"min_delta": -100, "max_delta": 100, "unit": "g"},
    "nutrition_calories_total": {"min_delta": -1000, "max_delta": 1000, "unit": "kcal"},
}

# Warning thresholds - generate warnings for changes above these percentages
WARNING_THRESHOLD_PERCENT = 50  # Warn if change is > 50% of typical daily intake

# Minimum days of historical data required for reliable predictions
MIN_HISTORICAL_DAYS = 14


class WhatIfService:
    """
    Service for what-if scenarios and counterfactual explanations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_prep_service = DataPreparationService(db)
        self.models_dir = Path("models")

    # ========================================================================
    # Safety Guardrails and Validation
    # ========================================================================

    def validate_nutrition_changes(
        self, changes: List[NutritionChange]
    ) -> tuple[List[NutritionChange], List[str]]:
        """
        Validate nutrition changes against safe bounds.

        Returns:
            Tuple of (validated_changes, warnings)
            - validated_changes: Changes clamped to safe bounds
            - warnings: List of warning messages for changes that were modified
        """
        validated = []
        warnings = []

        for change in changes:
            feature = change.feature_name
            delta = change.delta

            if feature in NUTRITION_SAFE_BOUNDS:
                bounds = NUTRITION_SAFE_BOUNDS[feature]
                min_delta = cast(float, bounds["min_delta"])
                max_delta = cast(float, bounds["max_delta"])
                unit = cast(str, bounds["unit"])

                # Clamp to safe bounds
                if delta < min_delta:
                    warnings.append(
                        f"⚠️ {feature}: Change of {delta}{unit} is too extreme. "
                        f"Clamped to {min_delta}{unit} for safety."
                    )
                    delta = min_delta
                elif delta > max_delta:
                    warnings.append(
                        f"⚠️ {feature}: Change of {delta}{unit} is too extreme. "
                        f"Clamped to {max_delta}{unit} for safety."
                    )
                    delta = max_delta

                validated.append(
                    NutritionChange(
                        feature_name=feature,
                        delta=delta,
                        change_description=change.change_description,
                    )
                )
            else:
                # Unknown feature - allow but warn
                warnings.append(
                    f"⚠️ Unknown nutrition feature: {feature}. "
                    "Proceeding with caution - predictions may be less reliable."
                )
                validated.append(change)

        return validated, warnings

    def check_data_quality(
        self, user_id: str, metrics: List[PredictionMetric]
    ) -> Optional[str]:
        """
        Check if sufficient historical data exists for reliable predictions.

        Returns:
            Warning message if data quality issues are detected, None otherwise.
        """
        # This is a placeholder - in production, would check actual data availability
        # For now, we return None (no warning) since the model training
        # already validates data requirements
        return None

    def generate_safety_warnings(
        self,
        nutrition_changes: List[NutritionChange],
        duration_days: int,
    ) -> List[str]:
        """
        Generate safety warnings for the simulation.

        Returns:
            List of warning messages
        """
        warnings = []

        # Warn about long-term simulations
        if duration_days >= 30:
            warnings.append(
                "📊 Long-term simulations (30+ days) have increased uncertainty. "
                "Predictions become less reliable over time."
            )

        # Warn about multiple significant changes
        significant_changes = 0
        for change in nutrition_changes:
            if abs(change.delta) > 50:  # More than 50 units
                significant_changes += 1

        if significant_changes >= 3:
            warnings.append(
                "⚠️ Multiple significant nutrition changes detected. "
                "Predictions for combined effects may be less accurate than "
                "for individual changes."
            )

        # Warn about opposing changes
        positive_protein = any(
            c.feature_name.endswith("protein") and c.delta > 0
            for c in nutrition_changes
        )
        negative_protein = any(
            c.feature_name.endswith("protein") and c.delta < 0
            for c in nutrition_changes
        )
        if positive_protein and negative_protein:
            warnings.append(
                "⚠️ Conflicting protein changes detected. "
                "Consider simplifying your simulation."
            )

        return warnings

    # ========================================================================
    # What-If Scenarios
    # ========================================================================

    async def test_what_if_scenarios(self, request: WhatIfRequest) -> WhatIfResponse:
        """
        Test multiple what-if scenarios.

        Steps:
        1. Load trained model
        2. Prepare baseline input features
        3. Make baseline prediction
        4. For each scenario:
           - Apply feature changes
           - Make prediction
           - Calculate difference from baseline
        5. Identify best and worst scenarios
        6. Generate summary

        Args:
            request: What-if request with scenarios

        Returns:
            WhatIfResponse with scenario results
        """
        print(f"\n{'='*70}")
        print(f"🔮 Testing what-if scenarios for {request.metric.value}")
        print(f"   User: {request.user_id}")
        print(f"   Date: {request.target_date}")
        print(f"   Scenarios: {len(request.scenarios)}")
        print(f"{'='*70}\n")

        # Step 1: Load model
        print("🧠 Step 1: Loading trained model...")
        model_id = self._find_latest_model(request.user_id, request.metric)

        if not model_id:
            raise ValueError(
                f"No trained model found for user {request.user_id} "
                f"and metric {request.metric.value}"
            )

        model_artifacts = self._load_model_artifacts(model_id)
        print(f"✅ Model loaded: {model_id}")

        # Step 2: Prepare baseline input
        print("\n📊 Step 2: Preparing baseline features...")
        X_baseline = await self.data_prep_service.prepare_prediction_input(
            user_id=request.user_id,
            target_date=request.target_date,
            sequence_length=model_artifacts["config"].sequence_length,
            scaler=model_artifacts["scaler"],
            feature_names=model_artifacts["feature_names"],
        )

        print("✅ Baseline input prepared")

        # Step 3: Make baseline prediction
        print("\n🔮 Step 3: Making baseline prediction...")
        model = model_artifacts["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        X_baseline_device = X_baseline.to(device)

        with torch.no_grad():
            baseline_normalized = model(X_baseline_device).item()

        baseline_prediction = self.data_prep_service.denormalize_prediction(
            baseline_normalized, model_artifacts["label_scaler"]
        )

        print(f"✅ Baseline prediction: {baseline_prediction:.2f}")

        # Step 4: Test each scenario
        print(f"\n🧪 Step 4: Testing {len(request.scenarios)} scenarios...")
        scenario_results = []

        for i, scenario in enumerate(request.scenarios):
            print(
                f"\n   Scenario {i + 1}/{len(request.scenarios)}: {scenario.scenario_name}"
            )

            # Apply changes to features
            X_modified = self._apply_scenario_changes(
                X_baseline.clone(),
                scenario,
                model_artifacts["feature_names"],
                model_artifacts["scaler"],
            )

            # Make prediction with modified features
            X_modified_device = X_modified.to(device)

            with torch.no_grad():
                scenario_normalized = model(X_modified_device).item()

            scenario_prediction = self.data_prep_service.denormalize_prediction(
                scenario_normalized, model_artifacts["label_scaler"]
            )

            # Calculate change
            change = scenario_prediction - baseline_prediction
            percent_change = (
                (change / baseline_prediction) * 100 if baseline_prediction != 0 else 0
            )

            print(
                f"      Prediction: {scenario_prediction:.2f} (change: {change:+.2f}, {percent_change:+.1f}%)"
            )

            # Identify biggest drivers (features that changed)
            biggest_drivers = [c.feature_name for c in scenario.changes[:3]]

            scenario_result = WhatIfResult(
                scenario_name=scenario.scenario_name,
                predicted_value=scenario_prediction,
                change_from_baseline=change,
                percent_change=percent_change,
                confidence_score=0.85,  # TODO: Calculate actual confidence
                biggest_drivers=biggest_drivers,
            )

            scenario_results.append(scenario_result)

        # Step 5: Identify best and worst scenarios
        print("\n📊 Step 5: Analyzing scenarios...")

        # For RHR/HRV: lower RHR and higher HRV are better
        if "RHR" in request.metric.value or "HEART_RATE" in request.metric.value:
            # Lower is better for RHR
            best_scenario = min(scenario_results, key=lambda x: x.predicted_value)
            worst_scenario = max(scenario_results, key=lambda x: x.predicted_value)
        else:
            # Higher is better for HRV
            best_scenario = max(scenario_results, key=lambda x: x.predicted_value)
            worst_scenario = min(scenario_results, key=lambda x: x.predicted_value)

        print(
            f"✅ Best scenario: {best_scenario.scenario_name} ({best_scenario.predicted_value:.2f})"
        )
        print(
            f"   Worst scenario: {worst_scenario.scenario_name} ({worst_scenario.predicted_value:.2f})"
        )

        # Step 6: Generate summary
        summary = self._generate_what_if_summary(
            metric=request.metric,
            baseline_prediction=baseline_prediction,
            best_scenario=best_scenario,
            worst_scenario=worst_scenario,
            scenario_results=scenario_results,
        )

        recommendation = self._generate_what_if_recommendation(
            metric=request.metric,
            best_scenario=best_scenario,
            scenarios=request.scenarios,
        )

        response = WhatIfResponse(
            user_id=request.user_id,
            metric=request.metric,
            target_date=request.target_date,
            baseline_prediction=baseline_prediction,
            baseline_confidence=0.85,  # TODO: Calculate actual
            scenarios=scenario_results,
            best_scenario=best_scenario.scenario_name,
            best_value=best_scenario.predicted_value,
            worst_scenario=worst_scenario.scenario_name,
            worst_value=worst_scenario.predicted_value,
            summary=summary,
            recommendation=recommendation,
        )

        print(f"\n{'='*70}")
        print("✅ What-if analysis complete!")
        print(f"{'='*70}\n")

        return response

    # ========================================================================
    # Counterfactual Explanations
    # ========================================================================

    async def generate_counterfactual(
        self, request: CounterfactualRequest
    ) -> CounterfactualResponse:
        """
        Generate counterfactual explanations.

        A counterfactual answers: "What minimal changes would achieve my target?"

        Example:
        - Current prediction: RHR = 65 BPM
        - Target: RHR = 60 BPM
        - Counterfactual: "Increase protein by 30g AND reduce late-night carbs by 50g"

        This uses a simple search algorithm:
        1. Try modifying individual features
        2. Measure impact on prediction
        3. Select features with highest impact
        4. Combine changes to reach target

        Args:
            request: Counterfactual request

        Returns:
            CounterfactualResponse with suggested changes
        """
        print(f"\n{'='*70}")
        print("🔍 Generating counterfactual explanation")
        print(f"   User: {request.user_id}")
        print(f"   Metric: {request.metric.value}")
        print(f"   Target: {request.target_type.value}")
        print(f"{'='*70}\n")

        # Step 1: Load model and prepare baseline
        print("🧠 Step 1: Loading model and baseline features...")
        model_id = self._find_latest_model(request.user_id, request.metric)

        if not model_id:
            raise ValueError("No trained model found")

        model_artifacts = self._load_model_artifacts(model_id)

        X_baseline = await self.data_prep_service.prepare_prediction_input(
            user_id=request.user_id,
            target_date=request.target_date,
            sequence_length=model_artifacts["config"].sequence_length,
            scaler=model_artifacts["scaler"],
            feature_names=model_artifacts["feature_names"],
        )

        # Get unnormalized features for display
        X_baseline_unnorm = await self._get_unnormalized_features(
            request.user_id,
            request.target_date,
            model_artifacts["config"].sequence_length,
            model_artifacts["feature_names"],
        )

        # Make baseline prediction
        model = model_artifacts["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        X_baseline_device = X_baseline.to(device)

        with torch.no_grad():
            current_normalized = model(X_baseline_device).item()

        current_prediction = self.data_prep_service.denormalize_prediction(
            current_normalized, model_artifacts["label_scaler"]
        )

        print(f"✅ Current prediction: {current_prediction:.2f}")

        # Step 2: Determine target
        if request.target_type == CounterfactualTarget.IMPROVE:
            # Improve by 5%
            if "RHR" in request.metric.value:
                target_prediction = current_prediction * 0.95  # Lower RHR
            else:
                target_prediction = current_prediction * 1.05  # Higher HRV
        elif request.target_type == CounterfactualTarget.TARGET_VALUE:
            target_prediction = request.target_value or current_prediction
        else:
            target_prediction = current_prediction * 1.05

        print(f"   Target prediction: {target_prediction:.2f}")

        # Step 3: Search for counterfactual changes
        print("\n🔍 Step 2: Searching for counterfactual changes...")

        changes = self._search_counterfactual_changes(
            model=model,
            X_baseline=X_baseline,
            X_baseline_unnorm=X_baseline_unnorm,
            current_prediction=current_prediction,
            target_prediction=target_prediction,
            feature_names=model_artifacts["feature_names"],
            scaler=model_artifacts["scaler"],
            label_scaler=model_artifacts["label_scaler"],
            max_changes=request.max_changes,
            allowed_features=request.allowed_features,
            device=device,
        )

        # Apply changes and get achieved prediction
        X_counterfactual = self._apply_counterfactual_changes(
            X_baseline.clone(), changes, model_artifacts["feature_names"]
        )

        X_counterfactual_device = X_counterfactual.to(device)

        with torch.no_grad():
            achieved_normalized = model(X_counterfactual_device).item()

        achieved_prediction = self.data_prep_service.denormalize_prediction(
            achieved_normalized, model_artifacts["label_scaler"]
        )

        print(f"✅ Achieved prediction: {achieved_prediction:.2f}")

        # Calculate plausibility
        plausibility = self._calculate_plausibility(changes)

        # Generate summary
        summary = self._generate_counterfactual_summary(
            metric=request.metric,
            current_prediction=current_prediction,
            target_prediction=target_prediction,
            achieved_prediction=achieved_prediction,
            changes=changes,
        )

        counterfactual = CounterfactualExplanation(
            current_prediction=current_prediction,
            target_prediction=target_prediction,
            achieved_prediction=achieved_prediction,
            changes=changes,
            plausibility_score=plausibility,
            summary=summary,
        )

        response = CounterfactualResponse(
            user_id=request.user_id,
            metric=request.metric,
            target_date=request.target_date,
            current_prediction=current_prediction,
            target_prediction=target_prediction,
            counterfactual=counterfactual,
            alternatives=[],  # TODO: Generate alternative counterfactuals
        )

        print(f"\n{'='*70}")
        print("✅ Counterfactual generated!")
        print(f"{'='*70}\n")

        return response

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _apply_scenario_changes(
        self,
        X_baseline: torch.Tensor,
        scenario: WhatIfScenario,
        feature_names: List[str],
        scaler,
    ) -> torch.Tensor:
        """
        Apply scenario changes to baseline features.

        Args:
            X_baseline: Baseline input (1, seq_len, num_features)
            scenario: Scenario with changes
            feature_names: List of feature names
            scaler: Feature scaler

        Returns:
            Modified input tensor
        """
        X_modified = X_baseline.clone()

        for change in scenario.changes:
            # Find feature index
            if change.feature_name not in feature_names:
                print(
                    f"⚠️ Warning: Feature '{change.feature_name}' not found, skipping"
                )
                continue

            feature_idx = feature_names.index(change.feature_name)

            # Normalize the new value
            # Create a dummy array with the feature value
            dummy = np.zeros((1, len(feature_names)))
            dummy[0, feature_idx] = change.new_value
            normalized_dummy = scaler.transform(dummy)
            new_normalized_value = normalized_dummy[0, feature_idx]

            # Apply change to last day (most recent)
            X_modified[0, -1, feature_idx] = new_normalized_value

        return X_modified

    def _apply_counterfactual_changes(
        self,
        X_baseline: torch.Tensor,
        changes: List[CounterfactualChange],
        feature_names: List[str],
    ) -> torch.Tensor:
        """Apply counterfactual changes to baseline."""
        X_modified = X_baseline.clone()

        for change in changes:
            if change.feature_name not in feature_names:
                continue

            feature_idx = feature_names.index(change.feature_name)

            # Apply change to last day
            X_modified[0, -1, feature_idx] = change.suggested_value

        return X_modified

    def _search_counterfactual_changes(
        self,
        model,
        X_baseline,
        X_baseline_unnorm,
        current_prediction,
        target_prediction,
        feature_names,
        scaler,
        label_scaler,
        max_changes,
        allowed_features,
        device,
    ) -> List[CounterfactualChange]:
        """
        Search for minimal changes to reach target.

        Uses a greedy search approach:
        1. Try modifying each feature individually
        2. Select feature with highest impact toward target
        3. Repeat until target reached or max_changes hit
        """
        changes: List[CounterfactualChange] = []
        X_current = X_baseline.clone()
        current_pred = current_prediction

        # Determine if we need to increase or decrease
        if target_prediction > current_prediction:
            direction = 1  # Increase
        else:
            direction = -1  # Decrease

        for _ in range(max_changes):
            best_feature = None
            best_impact = 0
            best_value = None
            best_unnorm_value = None

            # Try modifying each feature
            for i, feature_name in enumerate(feature_names):
                # Skip if feature not allowed
                if allowed_features and feature_name not in allowed_features:
                    continue

                # Skip if already changed
                if any(c.feature_name == feature_name for c in changes):
                    continue

                # Try increasing/decreasing the feature by 20%
                current_value = X_current[0, -1, i].item()
                current_unnorm = X_baseline_unnorm[-1, i]

                # Try modification
                new_value = current_value + (0.2 * direction)
                new_unnorm = (
                    current_unnorm * 1.2 if direction > 0 else current_unnorm * 0.8
                )

                # Test this change
                X_test = X_current.clone()
                X_test[0, -1, i] = new_value

                X_test_device = X_test.to(device)

                with torch.no_grad():
                    test_normalized = model(X_test_device).item()

                test_pred = self.data_prep_service.denormalize_prediction(
                    test_normalized, label_scaler
                )

                # Calculate impact (move toward target)
                if direction > 0:
                    impact = test_pred - current_pred
                else:
                    impact = current_pred - test_pred

                if impact > best_impact:
                    best_impact = impact
                    best_feature = feature_name
                    best_value = new_value
                    best_unnorm_value = new_unnorm

            # If we found a good change, apply it
            if best_feature and best_impact > 0 and best_unnorm_value is not None:
                feature_idx = feature_names.index(best_feature)
                X_current[0, -1, feature_idx] = best_value

                # Get current unnormalized value
                current_unnorm_value = X_baseline_unnorm[-1, feature_idx]

                change = CounterfactualChange(
                    feature_name=best_feature,
                    current_value=float(current_unnorm_value),
                    suggested_value=float(best_unnorm_value),
                    change_amount=float(best_unnorm_value - current_unnorm_value),
                    change_description=self._describe_change(
                        best_feature,
                        float(current_unnorm_value),
                        float(best_unnorm_value),
                    ),
                )

                changes.append(change)

                # Update current prediction
                current_pred = current_pred + (best_impact * direction)

                # Check if we reached target
                if (direction > 0 and current_pred >= target_prediction) or (
                    direction < 0 and current_pred <= target_prediction
                ):
                    break
            else:
                # No more improvements found
                break

        return changes

    def _describe_change(
        self, feature_name: str, current_value: float, new_value: float
    ) -> str:
        """Generate human-readable change description."""
        change = new_value - current_value

        # Clean feature name
        clean_name = (
            feature_name.replace("_", " ")
            .replace("nutrition ", "")
            .replace("activity ", "")
        )

        if "protein" in feature_name.lower():
            return f"{change:+.1f}g protein"
        elif "calories" in feature_name.lower():
            return f"{change:+.0f} calories"
        elif "workout" in feature_name.lower():
            return f"{change:+.1f} workout intensity"
        else:
            return f"{change:+.1f} {clean_name}"

    def _calculate_plausibility(self, changes: List[CounterfactualChange]) -> float:
        """Calculate how realistic these changes are (0-1)."""
        # For now, simple heuristic: fewer changes = more plausible
        if not changes:
            return 1.0

        # Penalize large changes
        max_change_ratio = max(
            abs(c.change_amount / c.current_value) if c.current_value != 0 else 1
            for c in changes
        )

        if max_change_ratio > 0.5:
            return 0.5  # Large changes are less plausible
        elif max_change_ratio > 0.3:
            return 0.7
        else:
            return 0.9

    def _generate_what_if_summary(
        self,
        metric,
        baseline_prediction,
        best_scenario,
        worst_scenario,
        scenario_results,
    ) -> str:
        """Generate summary of what-if scenarios."""
        summary = (
            f"Testing {len(scenario_results)} scenarios for {metric.value.lower()}, "
            f"your baseline prediction is {baseline_prediction:.1f}. "
            f"The best scenario is '{best_scenario.scenario_name}' with a predicted value of {best_scenario.predicted_value:.1f} "
            f"({best_scenario.change_from_baseline:+.1f} from baseline). "
            f"The worst scenario is '{worst_scenario.scenario_name}' with {worst_scenario.predicted_value:.1f}."
        )
        return summary

    def _generate_what_if_recommendation(self, metric, best_scenario, scenarios) -> str:
        """Generate recommendation based on best scenario."""
        # Find the best scenario details
        best_scenario_obj = next(
            (s for s in scenarios if s.scenario_name == best_scenario.scenario_name),
            None,
        )

        if not best_scenario_obj:
            return "Consider making the changes in the best scenario to improve your prediction."

        # Get top changes
        top_changes = best_scenario_obj.changes[:2]
        change_descriptions = [c.change_description for c in top_changes]

        recommendation = (
            f"To achieve the best outcome ({best_scenario.predicted_value:.1f} {metric.value.lower()}), "
            f"consider: {', '.join(change_descriptions)}."
        )

        return recommendation

    def _generate_counterfactual_summary(
        self,
        metric,
        current_prediction,
        target_prediction,
        achieved_prediction,
        changes,
    ) -> str:
        """Generate summary of counterfactual explanation."""
        if not changes:
            return f"No changes needed - your current prediction ({current_prediction:.1f}) is already at target."

        change_descriptions = [c.change_description for c in changes]

        summary = (
            f"To move from {current_prediction:.1f} to {target_prediction:.1f} {metric.value.lower()}, "
            f"you should: {', '.join(change_descriptions)}. "
            f"This would achieve approximately {achieved_prediction:.1f}."
        )

        return summary

    def _find_latest_model(self, user_id, metric) -> Optional[str]:
        """Find latest model for user and metric."""
        if not self.models_dir.exists():
            return None

        pattern = f"{user_id}_{metric.value}_*"
        matching_models = list(self.models_dir.glob(pattern))

        if not matching_models:
            return None

        matching_models.sort(key=lambda p: p.name, reverse=True)
        return matching_models[0].name

    def _load_model_artifacts(self, model_id: str) -> Dict:
        """Load model artifacts from disk."""
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # Load config
        with open(model_dir / "config.pkl", "rb") as f:
            config = pickle.load(f)

        # Load model
        model = HealthMetricLSTM(config)
        model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

        # Load scalers
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open(model_dir / "label_scaler.pkl", "rb") as f:
            label_scaler = pickle.load(f)

        # Load feature names
        with open(model_dir / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Load metadata
        with open(model_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return {
            "model": model,
            "scaler": scaler,
            "label_scaler": label_scaler,
            "config": config,
            "feature_names": feature_names,
            "metadata": metadata,
        }

    async def _get_unnormalized_features(
        self, user_id, target_date, sequence_length, feature_names
    ) -> np.ndarray:
        """Get unnormalized feature values."""
        import pandas as pd

        start_date = target_date - timedelta(days=sequence_length)

        feature_matrix, _ = await self.data_prep_service._build_feature_matrix(
            user_id, lookback_days=sequence_length + 5
        )

        if feature_matrix.empty:
            return np.zeros((sequence_length, len(feature_names)))

        # Convert dates to pandas datetime for comparison with datetime64 index
        start_dt = pd.to_datetime(start_date)
        target_dt = pd.to_datetime(target_date)
        mask = (feature_matrix.index >= start_dt) & (feature_matrix.index < target_dt)
        sequence_features = feature_matrix[mask]
        sequence_features = sequence_features.tail(sequence_length)
        sequence_features = sequence_features[feature_names]
        sequence_features = sequence_features.fillna(0)

        return sequence_features.values

    # ========================================================================
    # Multi-Day Trajectory Simulation
    # ========================================================================

    async def generate_trajectory(
        self, request: TrajectoryRequest
    ) -> TrajectoryResponse:
        """
        Generate multi-day trajectory projections with confidence intervals.

        This uses autoregressive prediction where each day's prediction
        feeds into the next day's input features.

        Args:
            request: Trajectory simulation request

        Returns:
            TrajectoryResponse with trajectory projections
        """
        print(f"\n{'='*70}")
        print("📈 Generating multi-day trajectory simulation")
        print(f"   User: {request.user_id}")
        print(f"   Duration: {request.duration_days} days")
        print(f"   Metrics: {[m.value for m in request.metrics_to_predict]}")
        print(f"   Changes: {len(request.nutrition_changes)}")
        print(f"{'='*70}\n")

        # ================================================================
        # Step 0: Safety Validation
        # ================================================================
        print("🛡️ Step 0: Validating safety guardrails...")

        # Validate and clamp nutrition changes to safe bounds
        validated_changes, validation_warnings = self.validate_nutrition_changes(
            request.nutrition_changes
        )

        # Generate safety warnings
        safety_warnings = self.generate_safety_warnings(
            validated_changes, request.duration_days
        )

        # Combine all warnings
        all_warnings = validation_warnings + safety_warnings

        if all_warnings:
            print(f"   ⚠️ {len(all_warnings)} warnings generated:")
            for w in all_warnings:
                print(f"      - {w}")
        else:
            print("   ✅ All safety checks passed")

        # Use validated changes for simulation
        nutrition_changes = validated_changes

        # Determine start date
        start_date = request.start_date or date.today()
        end_date = start_date + timedelta(days=request.duration_days)

        trajectories: List[SimulationTrajectory] = []
        baseline_trajectories: Optional[List[SimulationTrajectory]] = None

        if request.include_no_change_baseline:
            baseline_trajectories = []

        # Combine warnings into data quality warning
        data_quality_warning: Optional[str] = None
        if all_warnings:
            data_quality_warning = " ".join(all_warnings)

        # Generate trajectory for each metric
        for metric in request.metrics_to_predict:
            print(f"\n🎯 Processing {metric.value}...")

            # Load model for this metric
            model_id = self._find_latest_model(request.user_id, metric)

            if not model_id:
                print(f"⚠️ No trained model found for {metric.value}, skipping")
                continue

            model_artifacts = self._load_model_artifacts(model_id)
            model = model_artifacts["model"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device).eval()

            # Prepare baseline features
            X_baseline = await self.data_prep_service.prepare_prediction_input(
                user_id=request.user_id,
                target_date=start_date,
                sequence_length=model_artifacts["config"].sequence_length,
                scaler=model_artifacts["scaler"],
                feature_names=model_artifacts["feature_names"],
            )

            # Run simulation WITH nutrition changes (using validated changes)
            trajectory = await self._run_autoregressive_simulation(
                model=model,
                X_baseline=X_baseline,
                nutrition_changes=nutrition_changes,
                duration_days=request.duration_days,
                start_date=start_date,
                metric=metric,
                model_artifacts=model_artifacts,
                device=device,
            )
            trajectories.append(trajectory)

            # Run baseline simulation WITHOUT changes if requested
            if request.include_no_change_baseline and baseline_trajectories is not None:
                baseline_trajectory = await self._run_autoregressive_simulation(
                    model=model,
                    X_baseline=X_baseline,
                    nutrition_changes=[],  # No changes
                    duration_days=request.duration_days,
                    start_date=start_date,
                    metric=metric,
                    model_artifacts=model_artifacts,
                    device=device,
                )
                baseline_trajectories.append(baseline_trajectory)

        # Calculate overall model confidence
        model_confidence = self._calculate_model_confidence(trajectories)

        # Generate summary and recommendation
        summary = self._generate_trajectory_summary(
            request, trajectories, baseline_trajectories
        )
        recommendation = self._generate_trajectory_recommendation(
            trajectories, baseline_trajectories
        )

        response = TrajectoryResponse(
            user_id=request.user_id,
            start_date=start_date,
            end_date=end_date,
            duration_days=request.duration_days,
            nutrition_changes=nutrition_changes,  # Use validated changes
            trajectories=trajectories,
            baseline_trajectories=baseline_trajectories,
            model_confidence=model_confidence,
            data_quality_warning=data_quality_warning,
            summary=summary,
            recommendation=recommendation,
        )

        print(f"\n{'='*70}")
        print("✅ Trajectory simulation complete!")
        print(f"{'='*70}\n")

        return response

    async def _run_autoregressive_simulation(
        self,
        model,
        X_baseline: torch.Tensor,
        nutrition_changes: List[NutritionChange],
        duration_days: int,
        start_date: date,
        metric: PredictionMetric,
        model_artifacts: Dict,
        device,
    ) -> SimulationTrajectory:
        """
        Run autoregressive day-by-day simulation.

        Each day's prediction is used as input for the next day's prediction.
        """
        trajectory_points: List[TrajectoryPoint] = []
        values: List[float] = []

        # Clone baseline for modification
        X_current = X_baseline.clone()

        # Get model MAE for confidence intervals
        metadata = model_artifacts.get("metadata", {})
        mae = metadata.get("val_mae", 3.0)  # Default MAE if not available

        # Get optimal lag for nutrition changes
        optimal_lag_hours = await self._get_optimal_lag(
            self.db, metric, nutrition_changes
        )
        lag_days = (optimal_lag_hours or 24) // 24

        for day in range(duration_days + 1):
            current_date = start_date + timedelta(days=day)

            # Apply nutrition changes after lag period
            if day >= lag_days and nutrition_changes:
                X_current = self._apply_nutrition_deltas(
                    X_current,
                    nutrition_changes,
                    model_artifacts["feature_names"],
                    model_artifacts["scaler"],
                )

            # Make prediction
            X_current_device = X_current.to(device)

            with torch.no_grad():
                pred_normalized = model(X_current_device).item()

            predicted_value = self.data_prep_service.denormalize_prediction(
                pred_normalized, model_artifacts["label_scaler"]
            )

            # Calculate confidence interval (widens with time)
            # Using sqrt(day) factor for uncertainty accumulation
            uncertainty_factor = 1.0 + 0.1 * math.sqrt(max(day, 1))
            confidence_margin = 1.96 * mae * uncertainty_factor

            trajectory_point = TrajectoryPoint(
                day=day,
                timestamp=current_date,
                predicted_value=predicted_value,
                confidence_lower=predicted_value - confidence_margin,
                confidence_upper=predicted_value + confidence_margin,
            )
            trajectory_points.append(trajectory_point)
            values.append(predicted_value)

            # Shift sequence forward for next day (autoregressive)
            if day < duration_days:
                X_current = self._shift_sequence_forward(
                    X_current, pred_normalized, model_artifacts
                )

        # Calculate trajectory statistics
        baseline_value = values[0]
        final_value = values[-1]
        change = final_value - baseline_value
        percent_change = (change / baseline_value * 100) if baseline_value != 0 else 0

        # Generate lag description
        lag_description = None
        if optimal_lag_hours:
            lag_days_desc = optimal_lag_hours // 24
            if lag_days_desc == 1:
                lag_description = "Effects typically appear after 1 day"
            else:
                lag_description = f"Effects typically appear after {lag_days_desc} days"

        return SimulationTrajectory(
            metric=metric,
            baseline_value=baseline_value,
            projected_final_value=final_value,
            change_from_baseline=change,
            percent_change=percent_change,
            trajectory=trajectory_points,
            min_value=min(values),
            max_value=max(values),
            average_value=sum(values) / len(values),
            optimal_lag_hours=optimal_lag_hours,
            lag_description=lag_description,
        )

    def _apply_nutrition_deltas(
        self,
        X_current: torch.Tensor,
        nutrition_changes: List[NutritionChange],
        feature_names: List[str],
        scaler,
    ) -> torch.Tensor:
        """
        Apply nutrition delta changes to current feature sequence.
        """
        X_modified = X_current.clone()

        for change in nutrition_changes:
            if change.feature_name not in feature_names:
                continue

            feature_idx = feature_names.index(change.feature_name)

            # Get current unnormalized value (approximate from last timestep)
            # Then add delta and renormalize
            current_normalized = X_modified[0, -1, feature_idx].item()

            # Approximate: add normalized delta
            # Create dummy arrays to compute normalized delta
            dummy_current = np.zeros((1, len(feature_names)))
            dummy_with_delta = np.zeros((1, len(feature_names)))

            # Set a baseline value and compute difference
            baseline_val = 100.0  # arbitrary baseline
            dummy_current[0, feature_idx] = baseline_val
            dummy_with_delta[0, feature_idx] = baseline_val + change.delta

            normalized_current = scaler.transform(dummy_current)[0, feature_idx]
            normalized_with_delta = scaler.transform(dummy_with_delta)[0, feature_idx]
            normalized_delta = normalized_with_delta - normalized_current

            # Apply delta
            new_value = current_normalized + normalized_delta
            X_modified[0, -1, feature_idx] = new_value

        return X_modified

    def _shift_sequence_forward(
        self,
        X_current: torch.Tensor,
        pred_normalized: float,
        model_artifacts: Dict,
    ) -> torch.Tensor:
        """
        Shift sequence forward by one day for autoregressive prediction.

        Drops oldest day, shifts all days left, uses prediction in new day.
        """
        X_shifted = X_current.clone()

        # Shift: drop first day, move everything left
        X_shifted[0, :-1, :] = X_current[0, 1:, :]

        # Copy last day to new last position (will be modified by nutrition deltas)
        X_shifted[0, -1, :] = X_current[0, -1, :]

        # If there's a lagged health metric feature, update it with prediction
        feature_names = model_artifacts["feature_names"]
        for i, name in enumerate(feature_names):
            if "lag_1" in name.lower() and "hrv" in name.lower():
                X_shifted[0, -1, i] = pred_normalized
            elif "lag_1" in name.lower() and "rhr" in name.lower():
                X_shifted[0, -1, i] = pred_normalized

        return X_shifted

    async def _get_optimal_lag(
        self,
        db: AsyncSession,
        metric: PredictionMetric,
        nutrition_changes: List[NutritionChange],
    ) -> Optional[int]:
        """
        Get optimal lag hours from correlation engine for nutrition changes.
        """
        if not nutrition_changes:
            return None

        try:
            correlation_service = CorrelationEngineService(db)

            # Try to get lag for first nutrition change
            # This is a simplified approach - could be enhanced to consider all changes
            first_change = nutrition_changes[0]

            # Map feature name to a simpler key
            feature_key = first_change.feature_name.replace("nutrition_", "").replace(
                "_daily", ""
            )

            # Map PredictionMetric to HealthMetricTarget
            target_metric = HealthMetricTarget(metric.value)

            # Get correlation analysis using LagAnalysisRequest
            lag_request = LagAnalysisRequest(
                user_id="",  # Not used in basic analysis
                feature_name=feature_key,
                target_metric=target_metric,
                max_lag_hours=72,
            )
            result = await correlation_service.analyze_lag(lag_request)

            # Check if we have a significant result (optimal_correlation exists and is strong)
            if (
                result
                and result.optimal_lag_hours is not None
                and result.optimal_correlation is not None
            ):
                if abs(result.optimal_correlation) > 0.3:  # Significance threshold
                    return result.optimal_lag_hours

        except Exception as e:
            print(f"⚠️ Could not get optimal lag: {e}")

        # Default lag: 24 hours
        return 24

    def _calculate_model_confidence(
        self, trajectories: List[SimulationTrajectory]
    ) -> float:
        """Calculate overall model confidence from trajectory results."""
        if not trajectories:
            return 0.5

        # Base confidence on trajectory stability
        # Lower variance = higher confidence
        confidences = []

        for traj in trajectories:
            values = [p.predicted_value for p in traj.trajectory]
            if len(values) > 1:
                variance = np.var(values)
                mean_val = np.mean(values)
                cv = np.sqrt(variance) / mean_val if mean_val != 0 else 1
                # Convert coefficient of variation to confidence
                conf = max(0.5, min(0.95, 1 - cv))
                confidences.append(conf)

        return np.mean(confidences) if confidences else 0.75

    def _generate_trajectory_summary(
        self,
        request: TrajectoryRequest,
        trajectories: List[SimulationTrajectory],
        baseline_trajectories: Optional[List[SimulationTrajectory]],
    ) -> str:
        """Generate natural language summary of trajectory simulation."""
        if not trajectories:
            return "No trajectory projections could be generated."

        changes_desc = ", ".join(
            [c.change_description for c in request.nutrition_changes]
        )

        parts = [
            f"Simulating {request.duration_days}-day projections with changes: {changes_desc}."
        ]

        for traj in trajectories:
            direction = "increase" if traj.change_from_baseline > 0 else "decrease"
            parts.append(
                f"{traj.metric.value}: Projected to {direction} from "
                f"{traj.baseline_value:.1f} to {traj.projected_final_value:.1f} "
                f"({traj.percent_change:+.1f}%)."
            )

            if traj.lag_description:
                parts.append(traj.lag_description + ".")

            # Compare to baseline if available
            if baseline_trajectories:
                baseline = next(
                    (b for b in baseline_trajectories if b.metric == traj.metric), None
                )
                if baseline:
                    diff = traj.projected_final_value - baseline.projected_final_value
                    better_worse = (
                        "better" if self._is_improvement(traj.metric, diff) else "worse"
                    )
                    parts.append(
                        f"Compared to no changes: {abs(diff):.1f} {better_worse}."
                    )

        return " ".join(parts)

    def _generate_trajectory_recommendation(
        self,
        trajectories: List[SimulationTrajectory],
        baseline_trajectories: Optional[List[SimulationTrajectory]],
    ) -> str:
        """Generate recommendation based on trajectory outcomes."""
        if not trajectories:
            return "Unable to generate recommendations without trajectory data."

        improvements = 0
        for traj in trajectories:
            if baseline_trajectories:
                baseline = next(
                    (b for b in baseline_trajectories if b.metric == traj.metric), None
                )
                if baseline:
                    diff = traj.projected_final_value - baseline.projected_final_value
                    if self._is_improvement(traj.metric, diff):
                        improvements += 1

        if improvements == len(trajectories):
            return (
                "These changes are projected to improve all tracked metrics. "
                "Consider implementing these nutrition adjustments."
            )
        elif improvements > 0:
            return (
                f"These changes show mixed results: {improvements}/{len(trajectories)} "
                "metrics improved. Review individual trajectories before implementing."
            )
        else:
            return (
                "These changes may not improve your health metrics. "
                "Consider alternative adjustments or consult the simulation with different values."
            )

    def _is_improvement(self, metric: PredictionMetric, diff: float) -> bool:
        """Check if a difference represents an improvement for the given metric."""
        # For RHR, lower is better (negative diff is improvement)
        if "RHR" in metric.value or "RESTING_HEART_RATE" in metric.value:
            return diff < 0
        # For HRV, Recovery, etc., higher is better
        return diff > 0
