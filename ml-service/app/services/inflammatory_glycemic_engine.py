"""
Inflammatory Load Calculator & Glycemic Response Predictor

Research-backed models for:
1. Dietary Inflammatory Index (DII) calculation with 45 food parameters
2. Glycemic Load (GL) and Glycemic Index (GI) predictions
3. Inflammatory-HRV correlation modeling
4. Blood glucose variability impact on autonomic function
5. Personalized inflammatory/glycemic thresholds

Research Sources:
- Shivappa et al.: DII development and validation (1943 articles reviewed)
- Frontiers Endocrinology 2024: HbA1c variability and CVD events
- Atherosclerosis 2020: DII meta-analysis (15 cohort studies, RR=1.41)
- Cardiovasc Diabetology 2020: Glycemic variability clinical outcomes
- JACC 2020: DII and CVD risk in US populations
- Wiley J Diabetes Investigation 2024: GV measurement and targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import math
import numpy as np
from collections import defaultdict


# =============================================================================
# DIETARY INFLAMMATORY INDEX CONSTANTS
# =============================================================================

class DIIComponent(str, Enum):
    """DII food parameters (45 parameters from original research)"""
    # Macronutrients
    ENERGY = "energy"
    CARBOHYDRATE = "carbohydrate"
    PROTEIN = "protein"
    TOTAL_FAT = "total_fat"
    SATURATED_FAT = "saturated_fat"
    TRANS_FAT = "trans_fat"
    MUFA = "mufa"
    PUFA = "pufa"
    OMEGA_3 = "omega_3"
    OMEGA_6 = "omega_6"
    CHOLESTEROL = "cholesterol"
    FIBER = "fiber"

    # Vitamins
    VITAMIN_A = "vitamin_a"
    VITAMIN_C = "vitamin_c"
    VITAMIN_D = "vitamin_d"
    VITAMIN_E = "vitamin_e"
    VITAMIN_B1 = "vitamin_b1"
    VITAMIN_B2 = "vitamin_b2"
    VITAMIN_B6 = "vitamin_b6"
    VITAMIN_B12 = "vitamin_b12"
    FOLATE = "folate"
    NIACIN = "niacin"
    BETA_CAROTENE = "beta_carotene"

    # Minerals
    IRON = "iron"
    MAGNESIUM = "magnesium"
    ZINC = "zinc"
    SELENIUM = "selenium"

    # Bioactive Compounds
    FLAVONOIDS = "flavonoids"
    ANTHOCYANIDINS = "anthocyanidins"
    FLAVAN_3_OLS = "flavan_3_ols"
    FLAVONES = "flavones"
    FLAVONOLS = "flavonols"
    FLAVANONES = "flavanones"
    ISOFLAVONES = "isoflavones"

    # Spices/Herbs
    GARLIC = "garlic"
    GINGER = "ginger"
    TURMERIC = "turmeric"
    ONION = "onion"
    PEPPER = "pepper"
    ROSEMARY = "rosemary"
    OREGANO = "oregano"
    SAFFRON = "saffron"

    # Other
    TEA = "tea"
    CAFFEINE = "caffeine"
    ALCOHOL = "alcohol"


# DII scoring coefficients from Shivappa et al.
# Negative = anti-inflammatory, Positive = pro-inflammatory
DII_PARAMETERS: Dict[str, Dict[str, float]] = {
    # Parameter: {effect_score, global_mean, global_std}
    "energy": {"effect": 0.180, "mean": 2056, "std": 338, "unit": "kcal"},
    "carbohydrate": {"effect": 0.097, "mean": 272.2, "std": 40.0, "unit": "g"},
    "protein": {"effect": 0.021, "mean": 79.4, "std": 13.9, "unit": "g"},
    "total_fat": {"effect": 0.298, "mean": 71.4, "std": 19.4, "unit": "g"},
    "saturated_fat": {"effect": 0.373, "mean": 28.6, "std": 8.0, "unit": "g"},
    "trans_fat": {"effect": 0.229, "mean": 3.15, "std": 1.7, "unit": "g"},
    "mufa": {"effect": -0.009, "mean": 27.0, "std": 6.1, "unit": "g"},
    "pufa": {"effect": -0.159, "mean": 13.9, "std": 3.4, "unit": "g"},
    "omega_3": {"effect": -0.436, "mean": 1.06, "std": 1.06, "unit": "g"},
    "omega_6": {"effect": -0.159, "mean": 10.8, "std": 7.5, "unit": "g"},
    "cholesterol": {"effect": 0.110, "mean": 279.4, "std": 51.2, "unit": "mg"},
    "fiber": {"effect": -0.663, "mean": 18.8, "std": 4.9, "unit": "g"},

    # Vitamins
    "vitamin_a": {"effect": -0.401, "mean": 983.9, "std": 518.6, "unit": "RE"},
    "vitamin_c": {"effect": -0.424, "mean": 118.2, "std": 43.5, "unit": "mg"},
    "vitamin_d": {"effect": -0.446, "mean": 6.26, "std": 2.21, "unit": "mcg"},
    "vitamin_e": {"effect": -0.419, "mean": 8.73, "std": 1.49, "unit": "mg"},
    "vitamin_b1": {"effect": -0.098, "mean": 1.70, "std": 0.66, "unit": "mg"},
    "vitamin_b2": {"effect": -0.068, "mean": 1.70, "std": 0.79, "unit": "mg"},
    "vitamin_b6": {"effect": -0.365, "mean": 1.47, "std": 0.74, "unit": "mg"},
    "vitamin_b12": {"effect": -0.106, "mean": 5.15, "std": 2.70, "unit": "mcg"},
    "folate": {"effect": -0.190, "mean": 273.0, "std": 70.7, "unit": "mcg"},
    "niacin": {"effect": -0.246, "mean": 25.9, "std": 11.8, "unit": "mg"},
    "beta_carotene": {"effect": -0.584, "mean": 3718, "std": 1720, "unit": "mcg"},

    # Minerals
    "iron": {"effect": 0.032, "mean": 13.35, "std": 3.71, "unit": "mg"},
    "magnesium": {"effect": -0.484, "mean": 310.1, "std": 139.4, "unit": "mg"},
    "zinc": {"effect": -0.313, "mean": 9.84, "std": 2.19, "unit": "mg"},
    "selenium": {"effect": -0.191, "mean": 67.0, "std": 25.1, "unit": "mcg"},

    # Flavonoids and polyphenols
    "flavonoids": {"effect": -0.467, "mean": 159.0, "std": 159.0, "unit": "mg"},
    "anthocyanidins": {"effect": -0.131, "mean": 18.4, "std": 21.1, "unit": "mg"},
    "flavan_3_ols": {"effect": -0.159, "mean": 95.8, "std": 85.9, "unit": "mg"},
    "flavones": {"effect": -0.160, "mean": 1.55, "std": 0.07, "unit": "mg"},
    "flavonols": {"effect": -0.467, "mean": 17.2, "std": 6.79, "unit": "mg"},
    "flavanones": {"effect": -0.250, "mean": 11.7, "std": 3.16, "unit": "mg"},
    "isoflavones": {"effect": -0.593, "mean": 1.20, "std": 1.18, "unit": "mg"},

    # Spices and herbs
    "garlic": {"effect": -0.412, "mean": 4.35, "std": 2.90, "unit": "g"},
    "ginger": {"effect": -0.453, "mean": 0.50, "std": 0.50, "unit": "g"},
    "turmeric": {"effect": -0.785, "mean": 0.50, "std": 0.50, "unit": "g"},
    "onion": {"effect": -0.301, "mean": 35.9, "std": 18.4, "unit": "g"},
    "pepper": {"effect": -0.131, "mean": 10.0, "std": 7.07, "unit": "g"},
    "rosemary": {"effect": -0.013, "mean": 0.50, "std": 0.50, "unit": "g"},
    "oregano": {"effect": -0.102, "mean": 0.50, "std": 0.50, "unit": "g"},
    "saffron": {"effect": -0.140, "mean": 0.10, "std": 0.10, "unit": "g"},

    # Other
    "tea": {"effect": -0.536, "mean": 1.69, "std": 1.53, "unit": "cups"},
    "caffeine": {"effect": -0.110, "mean": 8.05, "std": 6.67, "unit": "mg/kg"},
    "alcohol": {"effect": -0.278, "mean": 13.98, "std": 3.72, "unit": "g"},  # J-curve
}

# DII score interpretation
DII_RISK_CATEGORIES = {
    "highly_anti_inflammatory": {"min": -8.87, "max": -4.0, "cvd_rr": 0.70},
    "moderately_anti_inflammatory": {"min": -4.0, "max": -1.5, "cvd_rr": 0.82},
    "mildly_anti_inflammatory": {"min": -1.5, "max": 0.0, "cvd_rr": 0.90},
    "neutral": {"min": 0.0, "max": 1.5, "cvd_rr": 1.0},
    "mildly_pro_inflammatory": {"min": 1.5, "max": 3.0, "cvd_rr": 1.15},
    "moderately_pro_inflammatory": {"min": 3.0, "max": 4.5, "cvd_rr": 1.30},
    "highly_pro_inflammatory": {"min": 4.5, "max": 7.98, "cvd_rr": 1.41},
}


# =============================================================================
# GLYCEMIC INDEX DATABASE
# =============================================================================

# GI values for common foods (glucose = 100)
GLYCEMIC_INDEX_DATABASE: Dict[str, Dict[str, Any]] = {
    # Low GI (≤55)
    "oats_rolled": {"gi": 55, "gl_per_100g": 21, "fiber_factor": 0.85},
    "lentils": {"gi": 29, "gl_per_100g": 5, "fiber_factor": 0.7},
    "chickpeas": {"gi": 28, "gl_per_100g": 8, "fiber_factor": 0.75},
    "sweet_potato": {"gi": 54, "gl_per_100g": 11, "fiber_factor": 0.8},
    "apple": {"gi": 38, "gl_per_100g": 5, "fiber_factor": 0.8},
    "orange": {"gi": 43, "gl_per_100g": 5, "fiber_factor": 0.85},
    "berries": {"gi": 25, "gl_per_100g": 3, "fiber_factor": 0.7},
    "yogurt_plain": {"gi": 36, "gl_per_100g": 3, "fiber_factor": 1.0},
    "milk": {"gi": 32, "gl_per_100g": 4, "fiber_factor": 1.0},
    "quinoa": {"gi": 53, "gl_per_100g": 13, "fiber_factor": 0.8},
    "brown_rice": {"gi": 50, "gl_per_100g": 16, "fiber_factor": 0.85},
    "whole_wheat_bread": {"gi": 51, "gl_per_100g": 15, "fiber_factor": 0.85},

    # Medium GI (56-69)
    "banana_ripe": {"gi": 62, "gl_per_100g": 14, "fiber_factor": 0.9},
    "pineapple": {"gi": 66, "gl_per_100g": 8, "fiber_factor": 0.95},
    "honey": {"gi": 61, "gl_per_100g": 50, "fiber_factor": 1.0},
    "basmati_rice": {"gi": 58, "gl_per_100g": 22, "fiber_factor": 0.9},
    "pita_bread": {"gi": 68, "gl_per_100g": 33, "fiber_factor": 0.95},

    # High GI (≥70)
    "white_rice": {"gi": 73, "gl_per_100g": 28, "fiber_factor": 1.0},
    "white_bread": {"gi": 75, "gl_per_100g": 38, "fiber_factor": 1.0},
    "potato_baked": {"gi": 85, "gl_per_100g": 18, "fiber_factor": 0.95},
    "corn_flakes": {"gi": 81, "gl_per_100g": 67, "fiber_factor": 1.0},
    "glucose": {"gi": 100, "gl_per_100g": 100, "fiber_factor": 1.0},
    "white_pasta": {"gi": 71, "gl_per_100g": 32, "fiber_factor": 0.95},
    "watermelon": {"gi": 76, "gl_per_100g": 6, "fiber_factor": 0.95},
}

# Glycemic variability thresholds
GV_THRESHOLDS = {
    "cv_glucose": {  # Coefficient of variation
        "excellent": 20,
        "good": 30,
        "fair": 36,
        "poor": 50,
    },
    "mage": {  # Mean amplitude of glycemic excursions
        "excellent": 40,
        "good": 60,
        "fair": 80,
        "poor": 100,
    },
    "time_in_range": {  # % time 70-180 mg/dL
        "excellent": 90,
        "good": 70,
        "fair": 50,
        "poor": 30,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DIICalculation:
    """Complete DII calculation result"""
    total_score: float
    category: str
    cvd_relative_risk: float
    component_scores: Dict[str, float]
    top_pro_inflammatory: List[Tuple[str, float]]
    top_anti_inflammatory: List[Tuple[str, float]]
    missing_components: List[str]
    confidence: float  # Based on data completeness


@dataclass
class GlycemicMeal:
    """Glycemic analysis of a single meal"""
    foods: List[str]
    total_glycemic_load: float
    weighted_glycemic_index: float
    carbohydrate_g: float
    fiber_g: float
    protein_g: float  # Protein blunts glucose response
    fat_g: float      # Fat slows absorption
    predicted_peak_glucose_rise: float  # mg/dL
    predicted_time_to_peak_min: int
    predicted_return_to_baseline_min: int


@dataclass
class GlycemicDay:
    """Daily glycemic load analysis"""
    total_glycemic_load: float
    mean_glycemic_index: float
    meals: List[GlycemicMeal]
    predicted_glucose_variability: str
    predicted_hrv_impact: float
    predicted_energy_stability: float
    recommendations: List[str]


@dataclass
class InflammatoryTrend:
    """Multi-day inflammatory trend"""
    period_days: int
    mean_dii: float
    trend_direction: str  # "improving", "stable", "worsening"
    trend_slope: float
    projected_dii_7d: float
    hrv_correlation: float
    inflammation_markers_predicted: Dict[str, str]


@dataclass
class CombinedHealthPrediction:
    """Combined inflammatory + glycemic health prediction"""
    dii_score: float
    daily_glycemic_load: float
    combined_risk_score: float  # 0-100
    autonomic_impact: str
    predicted_hrv_change: float
    predicted_crp_trend: str
    predicted_energy_pattern: str
    priority_interventions: List[Dict[str, Any]]


# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class InflammatoryResponsePredictor(nn.Module):
    """
    Neural network predicting inflammatory markers from dietary intake.

    Predicts:
    - CRP (C-reactive protein) trends
    - IL-6 levels
    - TNF-α trends
    - HRV impact
    """

    def __init__(
        self,
        input_dim: int = 50,  # DII components + meal patterns
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Temporal attention for multi-day patterns
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Output heads
        self.crp_head = nn.Linear(hidden_dim, 3)  # low/medium/high
        self.hrv_head = nn.Linear(hidden_dim, 1)  # continuous
        self.energy_head = nn.Linear(hidden_dim, 3)  # low/medium/high
        self.inflammation_head = nn.Linear(hidden_dim, 1)  # 0-1 score

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)

        return {
            "crp_class": F.softmax(self.crp_head(h), dim=-1),
            "hrv_change": torch.tanh(self.hrv_head(h)),
            "energy_class": F.softmax(self.energy_head(h), dim=-1),
            "inflammation_score": torch.sigmoid(self.inflammation_head(h)),
        }


class GlycemicResponsePredictor(nn.Module):
    """
    Neural network predicting glucose response curves.

    Uses meal composition to predict:
    - Peak glucose rise
    - Time to peak
    - Area under curve
    - Return to baseline time
    """

    def __init__(
        self,
        input_dim: int = 20,  # Meal composition features
        hidden_dim: int = 64,
        sequence_length: int = 24  # 2-hour response in 5-min intervals
    ):
        super().__init__()
        self.sequence_length = sequence_length

        # Meal encoder
        self.meal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Glucose curve generator (decoder)
        self.curve_generator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Output projection
        self.glucose_proj = nn.Linear(hidden_dim, 1)

        # Summary statistics head
        self.stats_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # peak, time_to_peak, auc, return_time
        )

    def forward(
        self,
        meal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict glucose response curve from meal.

        Args:
            meal_features: [batch, input_dim] meal composition

        Returns:
            glucose_curve: [batch, sequence_length] predicted glucose changes
            stats: [batch, 4] summary statistics
        """
        batch_size = meal_features.size(0)

        # Encode meal
        h = self.meal_encoder(meal_features)

        # Generate curve autoregressively
        h_expanded = h.unsqueeze(1).expand(-1, self.sequence_length, -1)
        curve_hidden, _ = self.curve_generator(h_expanded)

        # Project to glucose values
        glucose_curve = self.glucose_proj(curve_hidden).squeeze(-1)

        # Summary statistics
        stats = self.stats_head(h)

        return glucose_curve, stats


# =============================================================================
# MAIN ENGINE CLASSES
# =============================================================================

class DietaryInflammatoryIndexCalculator:
    """
    Calculates Dietary Inflammatory Index (DII) from food intake.

    Based on Shivappa et al. methodology validated against
    inflammatory biomarkers (CRP, IL-6, TNF-α).

    Range: approximately -8.87 (most anti-inflammatory) to +7.98 (most pro-inflammatory)
    """

    def __init__(self):
        self.parameters = DII_PARAMETERS
        self.risk_categories = DII_RISK_CATEGORIES

    def calculate(
        self,
        daily_intake: Dict[str, float]
    ) -> DIICalculation:
        """
        Calculate DII from daily nutrient intake.

        Args:
            daily_intake: Dict mapping nutrient names to amounts

        Returns:
            Complete DII calculation with breakdown
        """
        component_scores = {}
        missing = []

        for nutrient, params in self.parameters.items():
            if nutrient in daily_intake:
                # Z-score calculation
                z_score = (daily_intake[nutrient] - params["mean"]) / params["std"]

                # Convert to percentile (centered)
                percentile = self._percentile_from_z(z_score)

                # Multiply by effect score
                score = percentile * params["effect"]
                component_scores[nutrient] = score
            else:
                missing.append(nutrient)

        # Total DII score
        total_score = sum(component_scores.values())

        # Determine category
        category = self._get_category(total_score)
        cvd_rr = self.risk_categories[category]["cvd_rr"]

        # Sort components
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1]
        )

        # Top contributors
        top_pro = [(k, v) for k, v in sorted_components if v > 0][-5:][::-1]
        top_anti = [(k, v) for k, v in sorted_components if v < 0][:5]

        # Confidence based on data completeness
        confidence = 1 - (len(missing) / len(self.parameters))

        return DIICalculation(
            total_score=total_score,
            category=category,
            cvd_relative_risk=cvd_rr,
            component_scores=component_scores,
            top_pro_inflammatory=top_pro,
            top_anti_inflammatory=top_anti,
            missing_components=missing,
            confidence=confidence
        )

    def _percentile_from_z(self, z: float) -> float:
        """Convert z-score to centered percentile (-1 to 1)."""
        # Approximate standard normal CDF
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        # Center around 0
        return 2 * (cdf - 0.5)

    def _get_category(self, score: float) -> str:
        """Get DII category from score."""
        for category, bounds in self.risk_categories.items():
            if bounds["min"] <= score < bounds["max"]:
                return category
        return "neutral"

    def predict_hrv_impact(self, dii: DIICalculation) -> float:
        """
        Predict HRV impact from DII score.

        Research: Higher DII associated with:
        - Reduced HRV (especially HF power)
        - Increased LF/HF ratio
        - Elevated resting HR
        """
        # Approximately -0.02 per DII point
        base_impact = -0.02 * dii.total_score

        # Additional penalty for high pro-inflammatory scores
        if dii.category in ["moderately_pro_inflammatory", "highly_pro_inflammatory"]:
            base_impact *= 1.3

        return max(-0.15, min(0.1, base_impact))

    def get_recommendations(
        self,
        dii: DIICalculation
    ) -> List[Dict[str, Any]]:
        """Generate recommendations to improve DII score."""
        recommendations = []

        # Check for high pro-inflammatory components
        for nutrient, score in dii.top_pro_inflammatory:
            if score > 0.1:
                recommendations.append({
                    "type": "reduce",
                    "nutrient": nutrient,
                    "current_impact": score,
                    "action": f"Reduce {nutrient.replace('_', ' ')} intake",
                    "alternatives": self._get_alternatives(nutrient),
                    "expected_dii_improvement": -score * 0.5
                })

        # Suggest anti-inflammatory foods
        low_anti = [
            k for k, v in dii.component_scores.items()
            if v > -0.1 and self.parameters[k]["effect"] < -0.3
        ]

        for nutrient in low_anti[:3]:
            recommendations.append({
                "type": "increase",
                "nutrient": nutrient,
                "current_impact": dii.component_scores.get(nutrient, 0),
                "action": f"Increase {nutrient.replace('_', ' ')} intake",
                "food_sources": self._get_food_sources(nutrient),
                "expected_dii_improvement": self.parameters[nutrient]["effect"] * 0.5
            })

        return recommendations

    def _get_alternatives(self, nutrient: str) -> List[str]:
        """Get healthier alternatives for pro-inflammatory foods."""
        alternatives = {
            "saturated_fat": ["olive oil", "avocado", "nuts", "fatty fish"],
            "trans_fat": ["butter (in moderation)", "coconut oil", "olive oil"],
            "cholesterol": ["plant sterols", "legumes", "whole grains"],
            "total_fat": ["lean proteins", "vegetables", "fruit"],
        }
        return alternatives.get(nutrient, ["whole foods", "vegetables"])

    def _get_food_sources(self, nutrient: str) -> List[str]:
        """Get food sources for anti-inflammatory nutrients."""
        sources = {
            "omega_3": ["salmon", "mackerel", "sardines", "walnuts", "flaxseed"],
            "fiber": ["oats", "legumes", "vegetables", "berries", "whole grains"],
            "turmeric": ["curries", "golden milk", "turmeric supplements"],
            "vitamin_c": ["citrus", "berries", "bell peppers", "broccoli"],
            "vitamin_d": ["fatty fish", "egg yolks", "mushrooms", "sunlight"],
            "magnesium": ["dark chocolate", "almonds", "spinach", "avocado"],
            "flavonoids": ["berries", "dark chocolate", "tea", "red wine"],
            "beta_carotene": ["carrots", "sweet potato", "spinach", "kale"],
        }
        return sources.get(nutrient, ["various whole foods"])


class GlycemicResponseCalculator:
    """
    Calculates and predicts glycemic response from meals.

    Features:
    - Glycemic Index (GI) weighted calculations
    - Glycemic Load (GL) computation
    - Glucose curve prediction
    - HRV impact estimation
    """

    def __init__(self):
        self.gi_database = GLYCEMIC_INDEX_DATABASE
        self.gv_thresholds = GV_THRESHOLDS

    def calculate_meal_glycemic_response(
        self,
        foods: List[Dict[str, Any]]
    ) -> GlycemicMeal:
        """
        Calculate glycemic response for a meal.

        Args:
            foods: List of {name, amount_g, carbs_g, protein_g, fat_g, fiber_g}

        Returns:
            Complete meal glycemic analysis
        """
        total_carbs = 0
        total_gl = 0
        weighted_gi = 0
        total_fiber = 0
        total_protein = 0
        total_fat = 0
        food_names = []

        for food in foods:
            name = food.get("name", "unknown").lower().replace(" ", "_")
            amount_g = food.get("amount_g", 100)
            carbs_g = food.get("carbs_g", 0)

            food_names.append(name)

            # Get GI from database or estimate
            if name in self.gi_database:
                gi_data = self.gi_database[name]
                gi = gi_data["gi"]
                fiber_factor = gi_data["fiber_factor"]
            else:
                # Default medium GI
                gi = 50
                fiber_factor = 0.9

            # Fiber from food or database
            fiber = food.get("fiber_g", 0)
            total_fiber += fiber

            # Apply fiber factor to GI
            effective_gi = gi * fiber_factor

            # Calculate glycemic load for this food
            gl = (effective_gi * carbs_g) / 100
            total_gl += gl

            # Weight GI by carb contribution
            weighted_gi += effective_gi * carbs_g
            total_carbs += carbs_g

            total_protein += food.get("protein_g", 0)
            total_fat += food.get("fat_g", 0)

        # Finalize weighted GI
        if total_carbs > 0:
            weighted_gi /= total_carbs
        else:
            weighted_gi = 0

        # Predict glucose response
        peak_rise, time_to_peak, return_time = self._predict_glucose_curve(
            total_gl, weighted_gi, total_protein, total_fat, total_fiber
        )

        return GlycemicMeal(
            foods=food_names,
            total_glycemic_load=total_gl,
            weighted_glycemic_index=weighted_gi,
            carbohydrate_g=total_carbs,
            fiber_g=total_fiber,
            protein_g=total_protein,
            fat_g=total_fat,
            predicted_peak_glucose_rise=peak_rise,
            predicted_time_to_peak_min=time_to_peak,
            predicted_return_to_baseline_min=return_time
        )

    def _predict_glucose_curve(
        self,
        gl: float,
        gi: float,
        protein: float,
        fat: float,
        fiber: float
    ) -> Tuple[float, int, int]:
        """
        Predict glucose response curve parameters.

        Returns:
            (peak_rise_mg_dL, time_to_peak_min, return_to_baseline_min)
        """
        # Base peak rise from glycemic load
        # Approximately 1-3 mg/dL per GL unit depending on individual
        base_peak = gl * 2.0

        # Protein blunts response (~10-30%)
        protein_factor = 1 - min(0.3, protein / 100)

        # Fat slows absorption
        fat_factor = 1 - min(0.2, fat / 80)

        # Fiber moderates
        fiber_factor = 1 - min(0.25, fiber / 20)

        peak_rise = base_peak * protein_factor * fat_factor * fiber_factor
        peak_rise = min(80, max(5, peak_rise))  # Cap at reasonable range

        # Time to peak (higher GI = faster peak)
        base_time = 45  # minutes
        gi_factor = gi / 50  # Normalize around medium GI
        fat_delay = min(30, fat / 2)  # Fat delays peak

        time_to_peak = int(base_time / gi_factor + fat_delay)
        time_to_peak = max(20, min(90, time_to_peak))

        # Return to baseline (lower GI = longer sustained energy)
        return_time = int(time_to_peak + 60 + (100 - gi) * 0.8 + fiber * 3)
        return_time = max(60, min(180, return_time))

        return (peak_rise, time_to_peak, return_time)

    def calculate_daily_glycemic_profile(
        self,
        meals: List[GlycemicMeal]
    ) -> GlycemicDay:
        """
        Calculate daily glycemic profile from all meals.
        """
        if not meals:
            return GlycemicDay(
                total_glycemic_load=0,
                mean_glycemic_index=0,
                meals=[],
                predicted_glucose_variability="unknown",
                predicted_hrv_impact=0,
                predicted_energy_stability=0.5,
                recommendations=["Log meals to get glycemic analysis"]
            )

        total_gl = sum(m.total_glycemic_load for m in meals)
        mean_gi = sum(m.weighted_glycemic_index for m in meals) / len(meals)

        # Predict glucose variability
        variability = self._predict_variability(meals)

        # HRV impact (research: GV negatively correlates with HRV)
        hrv_impact = self._calculate_hrv_impact(total_gl, variability)

        # Energy stability (lower GI = more stable energy)
        energy_stability = 1 - (mean_gi / 100) * 0.6 - (total_gl / 200) * 0.4
        energy_stability = max(0, min(1, energy_stability))

        # Recommendations
        recommendations = self._generate_glycemic_recommendations(
            total_gl, mean_gi, variability, meals
        )

        return GlycemicDay(
            total_glycemic_load=total_gl,
            mean_glycemic_index=mean_gi,
            meals=meals,
            predicted_glucose_variability=variability,
            predicted_hrv_impact=hrv_impact,
            predicted_energy_stability=energy_stability,
            recommendations=recommendations
        )

    def _predict_variability(self, meals: List[GlycemicMeal]) -> str:
        """Predict glucose variability category."""
        # High GL meals + high GI = high variability
        max_gl = max(m.total_glycemic_load for m in meals)
        max_gi = max(m.weighted_glycemic_index for m in meals)
        mean_gi = sum(m.weighted_glycemic_index for m in meals) / len(meals)

        variability_score = (max_gl / 30) + (max_gi / 70) + (mean_gi / 70)

        if variability_score < 1.5:
            return "low"
        elif variability_score < 2.5:
            return "moderate"
        elif variability_score < 3.5:
            return "elevated"
        return "high"

    def _calculate_hrv_impact(
        self,
        total_gl: float,
        variability: str
    ) -> float:
        """
        Calculate predicted HRV impact from glycemic profile.

        Research: GV negatively correlated with HRV independent of
        metabolic syndrome components.
        """
        # Base impact from total GL
        gl_impact = -0.01 * (total_gl / 100)

        # Variability penalty
        variability_factors = {
            "low": 0,
            "moderate": -0.02,
            "elevated": -0.05,
            "high": -0.08,
        }
        var_impact = variability_factors.get(variability, -0.03)

        return max(-0.15, gl_impact + var_impact)

    def _generate_glycemic_recommendations(
        self,
        total_gl: float,
        mean_gi: float,
        variability: str,
        meals: List[GlycemicMeal]
    ) -> List[str]:
        """Generate recommendations for glycemic optimization."""
        recommendations = []

        if total_gl > 150:
            recommendations.append(
                "Consider reducing total carbohydrate intake "
                f"(current GL: {total_gl:.0f}, target: <120)"
            )

        if mean_gi > 60:
            recommendations.append(
                "Swap high-GI foods for low-GI alternatives "
                "(whole grains, legumes, non-starchy vegetables)"
            )

        if variability in ["elevated", "high"]:
            recommendations.append(
                "Add protein and healthy fats to carb-heavy meals "
                "to reduce glucose spikes"
            )

        # Check individual meals
        for i, meal in enumerate(meals):
            if meal.total_glycemic_load > 40:
                recommendations.append(
                    f"Meal {i+1} has high GL ({meal.total_glycemic_load:.0f}). "
                    "Consider reducing portion sizes or adding fiber."
                )

        if not recommendations:
            recommendations.append(
                "Glycemic profile looks good! "
                "Continue balanced eating patterns."
            )

        return recommendations


class CombinedInflammatoryGlycemicEngine:
    """
    Combined engine integrating inflammatory and glycemic analysis.

    Provides holistic assessment of dietary impact on:
    - Systemic inflammation
    - Glucose homeostasis
    - Autonomic function (HRV)
    - Cardiovascular risk
    """

    def __init__(self):
        self.dii_calculator = DietaryInflammatoryIndexCalculator()
        self.glycemic_calculator = GlycemicResponseCalculator()

        # Neural models (optional)
        self.inflammatory_predictor = InflammatoryResponsePredictor()
        self.glycemic_predictor = GlycemicResponsePredictor()

    def analyze_full_day(
        self,
        nutrient_intake: Dict[str, float],
        meals: List[Dict[str, Any]]
    ) -> CombinedHealthPrediction:
        """
        Complete daily analysis combining inflammatory and glycemic factors.

        Args:
            nutrient_intake: Full day nutrient totals for DII
            meals: List of meals with food composition

        Returns:
            Combined health prediction
        """
        # Calculate DII
        dii = self.dii_calculator.calculate(nutrient_intake)

        # Calculate glycemic profile
        glycemic_meals = [
            self.glycemic_calculator.calculate_meal_glycemic_response(
                meal.get("foods", [])
            )
            for meal in meals
        ]
        glycemic_day = self.glycemic_calculator.calculate_daily_glycemic_profile(
            glycemic_meals
        )

        # Combined risk score (0-100)
        # Higher DII and higher GV both increase risk
        dii_risk = (dii.total_score + 5) / 13 * 50  # Normalize to 0-50
        gv_risk = {
            "low": 10, "moderate": 20, "elevated": 35, "high": 50
        }.get(glycemic_day.predicted_glucose_variability, 25)

        combined_risk = dii_risk + gv_risk
        combined_risk = max(0, min(100, combined_risk))

        # Autonomic impact assessment
        total_hrv_impact = (
            self.dii_calculator.predict_hrv_impact(dii) +
            glycemic_day.predicted_hrv_impact
        )

        if total_hrv_impact > 0.05:
            autonomic_impact = "parasympathetic_supporting"
        elif total_hrv_impact > -0.02:
            autonomic_impact = "neutral"
        elif total_hrv_impact > -0.08:
            autonomic_impact = "mildly_sympathetic"
        else:
            autonomic_impact = "sympathetic_dominant"

        # CRP trend prediction
        if dii.category in ["highly_anti_inflammatory", "moderately_anti_inflammatory"]:
            crp_trend = "likely_decreasing"
        elif dii.category in ["highly_pro_inflammatory", "moderately_pro_inflammatory"]:
            crp_trend = "likely_increasing"
        else:
            crp_trend = "stable"

        # Energy pattern prediction
        if glycemic_day.predicted_energy_stability > 0.7:
            energy_pattern = "stable_throughout_day"
        elif glycemic_day.predicted_energy_stability > 0.4:
            energy_pattern = "moderate_fluctuations"
        else:
            energy_pattern = "energy_rollercoaster"

        # Priority interventions
        interventions = self._get_priority_interventions(dii, glycemic_day)

        return CombinedHealthPrediction(
            dii_score=dii.total_score,
            daily_glycemic_load=glycemic_day.total_glycemic_load,
            combined_risk_score=combined_risk,
            autonomic_impact=autonomic_impact,
            predicted_hrv_change=total_hrv_impact,
            predicted_crp_trend=crp_trend,
            predicted_energy_pattern=energy_pattern,
            priority_interventions=interventions
        )

    def _get_priority_interventions(
        self,
        dii: DIICalculation,
        glycemic: GlycemicDay
    ) -> List[Dict[str, Any]]:
        """Get prioritized interventions based on analysis."""
        interventions = []

        # DII-based interventions
        if dii.total_score > 2:
            interventions.append({
                "priority": 1,
                "category": "anti_inflammatory",
                "intervention": "Increase omega-3 and fiber intake",
                "specific_actions": [
                    "Add fatty fish 2-3x per week",
                    "Include turmeric/ginger in cooking",
                    "Add berries or dark leafy greens daily"
                ],
                "expected_impact": f"DII reduction of 1-2 points",
                "evidence": "Meta-analysis: lowest DII quartile has RR=0.70 for CVD"
            })

        # Glycemic interventions
        if glycemic.total_glycemic_load > 120:
            interventions.append({
                "priority": 2,
                "category": "glycemic_control",
                "intervention": "Reduce glycemic load",
                "specific_actions": [
                    "Replace white rice with quinoa or cauliflower rice",
                    "Add protein to every meal",
                    "Increase fiber intake to 30g/day"
                ],
                "expected_impact": "Reduced glucose variability, more stable energy",
                "evidence": "GV independently associated with CVD risk"
            })

        # Combined intervention for synergistic effect
        if dii.total_score > 0 and glycemic.total_glycemic_load > 100:
            interventions.append({
                "priority": 1,
                "category": "synergistic",
                "intervention": "Mediterranean diet pattern",
                "specific_actions": [
                    "Olive oil as primary fat",
                    "Legumes 3-4x per week",
                    "Whole grains over refined",
                    "Abundant vegetables and moderate fish"
                ],
                "expected_impact": "Addresses both inflammation and glucose control",
                "evidence": "Mediterranean diet reduces CVD risk by 30%"
            })

        return sorted(interventions, key=lambda x: x["priority"])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_dii(daily_nutrients: Dict[str, float]) -> DIICalculation:
    """Quick DII calculation from nutrient intake."""
    calculator = DietaryInflammatoryIndexCalculator()
    return calculator.calculate(daily_nutrients)


def analyze_meal_glycemic(foods: List[Dict[str, Any]]) -> GlycemicMeal:
    """Quick glycemic analysis for a single meal."""
    calculator = GlycemicResponseCalculator()
    return calculator.calculate_meal_glycemic_response(foods)


def full_dietary_analysis(
    nutrients: Dict[str, float],
    meals: List[Dict[str, Any]]
) -> CombinedHealthPrediction:
    """Complete dietary health analysis."""
    engine = CombinedInflammatoryGlycemicEngine()
    return engine.analyze_full_day(nutrients, meals)
