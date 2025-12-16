"""
What-If Scenarios and Counterfactual Explanations

This service allows users to test hypothetical changes:
- "What if I ate 50g more protein?"
- "What if I did a high-intensity workout?"
- "What if I got 8 hours of sleep?"

It also generates counterfactual explanations:
- "To reach your target RHR, you should increase protein by 30g and reduce late-night calories by 200"
"""

import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

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
)
from app.services.data_preparation import DataPreparationService


class WhatIfService:
    """
    Service for what-if scenarios and counterfactual explanations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_prep_service = DataPreparationService(db)
        self.models_dir = Path("models")

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
