"""
Compound Quantification Service

Advanced quantification and dose-response modeling for:
- Histamine (mg/100g thresholds, DAO inhibitors)
- Tyramine (MAOI interactions, migraine triggers)
- FODMAP (stacking effects, cumulative load)
- Salicylates, oxalates, lectins

Based on clinical research thresholds:
- Histamine: <25mg/meal safe for most, <10mg for sensitive
- Tyramine: <6mg/meal for MAOI users, <25mg for migraine prone
- FODMAP: <0.5g fructans, <0.2g excess fructose per meal (low FODMAP)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.data.allergen_database import (
    INGREDIENT_DATABASE,
    IngredientData,
    CompoundLevel,
    FodmapLevel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CLINICAL THRESHOLDS (from research)
# =============================================================================

# Histamine thresholds (mg per meal)
HISTAMINE_THRESHOLDS = {
    "safe_general": 50.0,  # Most people tolerate up to 50mg/meal
    "caution_general": 25.0,  # General caution threshold
    "safe_sensitive": 10.0,  # For histamine-sensitive individuals
    "danger_sensitive": 5.0,  # Likely to trigger symptoms
    "daily_limit_general": 100.0,
    "daily_limit_sensitive": 20.0,
}

# Tyramine thresholds (mg per meal)
TYRAMINE_THRESHOLDS = {
    "safe_general": 50.0,  # Most people tolerate well
    "safe_migraine": 25.0,  # For migraine-prone individuals
    "danger_maoi": 6.0,  # MAOI users must stay below this
    "daily_limit_general": 150.0,
    "daily_limit_maoi": 15.0,
}

# FODMAP thresholds (grams per meal) - Monash University guidelines
FODMAP_THRESHOLDS = {
    "fructans": {"low": 0.3, "high": 0.5},
    "galactans": {"low": 0.2, "high": 0.3},
    "lactose": {"low": 1.0, "high": 4.0},
    "polyols_sorbitol": {"low": 0.3, "high": 0.5},
    "polyols_mannitol": {"low": 0.3, "high": 0.5},
    "excess_fructose": {"low": 0.15, "high": 0.2},
}

# DAO inhibitor foods (reduce histamine metabolism)
DAO_INHIBITORS = {
    "alcohol",
    "black_tea",
    "green_tea",
    "energy_drink",
    "mate",
    "beer",
    "wine_red",
    "wine_white",
    "champagne",
}

# Histamine liberators (trigger histamine release)
HISTAMINE_LIBERATORS = {
    "citrus",
    "strawberry",
    "papaya",
    "pineapple",
    "tomato",
    "spinach",
    "cocoa",
    "chocolate",
    "shellfish",
    "egg_white",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


class RiskLevel(str, Enum):
    """Risk levels for compound exposure."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CompoundQuantification:
    """Quantification result for a single compound."""

    compound_name: str
    total_mg: float
    level: CompoundLevel
    risk_level: RiskLevel
    threshold_used: str
    threshold_value: float
    percentage_of_threshold: float
    sources: List[Dict[str, float]] = field(default_factory=list)  # ingredient -> mg
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FodmapQuantification:
    """FODMAP quantification with per-type breakdown."""

    total_fodmap_types: int
    high_fodmap_count: int
    risk_level: RiskLevel
    by_type: Dict[str, Dict[str, any]] = field(default_factory=dict)
    stacking_warning: bool = False
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MealCompoundProfile:
    """Complete compound profile for a meal."""

    histamine: CompoundQuantification
    tyramine: CompoundQuantification
    fodmap: FodmapQuantification
    salicylate: Optional[CompoundQuantification] = None
    oxalate: Optional[CompoundQuantification] = None
    overall_risk: RiskLevel = RiskLevel.LOW
    interaction_warnings: List[str] = field(default_factory=list)
    meal_timing_recommendation: Optional[str] = None


@dataclass
class DailyCompoundAccumulation:
    """Track compound accumulation across meals."""

    date: datetime
    histamine_total_mg: float = 0.0
    tyramine_total_mg: float = 0.0
    fodmap_exposures: int = 0
    meals_analyzed: int = 0
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# MAIN SERVICE
# =============================================================================


class CompoundQuantificationService:
    """
    Service for quantifying bioactive compounds in meals and
    calculating dose-response risk levels.

    Features:
    - Histamine quantification with DAO inhibitor detection
    - Tyramine quantification with MAOI warnings
    - FODMAP stacking and cumulative load calculation
    - Cross-compound interaction detection
    - Temporal risk factors (aged, fermented foods)
    """

    def __init__(self):
        """Initialize with default user sensitivity profile."""
        logger.info("Initialized CompoundQuantificationService")

    def quantify_meal_compounds(
        self,
        ingredients: List[str],
        portion_weights: Optional[Dict[str, float]] = None,
        user_sensitivities: Optional[List[str]] = None,
        is_maoi_user: bool = False,
        is_histamine_sensitive: bool = False,
        is_migraine_prone: bool = False,
    ) -> MealCompoundProfile:
        """
        Quantify all bioactive compounds in a meal.

        Args:
            ingredients: List of ingredient keys from INGREDIENT_DATABASE
            portion_weights: Optional dict of ingredient -> weight in grams
            user_sensitivities: List of user's known sensitivities
            is_maoi_user: Whether user takes MAO inhibitors
            is_histamine_sensitive: Whether user has histamine intolerance
            is_migraine_prone: Whether user is prone to migraines

        Returns:
            Complete compound profile with risk assessments
        """
        user_sensitivities = user_sensitivities or []
        portion_weights = portion_weights or {}

        # Gather ingredient data
        ingredient_data = self._gather_ingredient_data(ingredients)

        # Calculate individual compounds
        histamine = self._quantify_histamine(
            ingredient_data, portion_weights, is_histamine_sensitive
        )

        tyramine = self._quantify_tyramine(
            ingredient_data, portion_weights, is_maoi_user, is_migraine_prone
        )

        fodmap = self._quantify_fodmap(
            ingredient_data, portion_weights, "fodmap" in user_sensitivities
        )

        # Detect cross-compound interactions
        interaction_warnings = self._detect_interactions(
            ingredients,
            ingredient_data,
            histamine,
            tyramine,
        )

        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(
            histamine.risk_level,
            tyramine.risk_level,
            fodmap.risk_level,
            len(interaction_warnings),
        )

        # Meal timing recommendation
        timing_rec = self._get_timing_recommendation(
            histamine, tyramine, fodmap, overall_risk
        )

        return MealCompoundProfile(
            histamine=histamine,
            tyramine=tyramine,
            fodmap=fodmap,
            overall_risk=overall_risk,
            interaction_warnings=interaction_warnings,
            meal_timing_recommendation=timing_rec,
        )

    def _gather_ingredient_data(
        self, ingredients: List[str]
    ) -> Dict[str, IngredientData]:
        """Gather IngredientData for all ingredients."""
        data = {}
        for ing in ingredients:
            ing_lower = ing.lower().replace(" ", "_")
            if ing_lower in INGREDIENT_DATABASE:
                data[ing] = INGREDIENT_DATABASE[ing_lower]
            elif ing in INGREDIENT_DATABASE:
                data[ing] = INGREDIENT_DATABASE[ing]
        return data

    def _quantify_histamine(
        self,
        ingredient_data: Dict[str, IngredientData],
        portion_weights: Dict[str, float],
        is_sensitive: bool,
    ) -> CompoundQuantification:
        """Quantify total histamine with risk assessment."""
        total_mg = 0.0
        sources = []
        warnings = []
        has_dao_inhibitor = False
        has_liberator = False

        for ing_name, data in ingredient_data.items():
            weight = portion_weights.get(ing_name, 100.0)  # Default 100g

            if data.histamine_mg:
                # Scale to portion weight
                amount = (data.histamine_mg / 100.0) * weight
                total_mg += amount
                sources.append(
                    {"ingredient": data.display_name, "mg": round(amount, 2)}
                )

            # Check for DAO inhibitors
            ing_key = ing_name.lower().replace(" ", "_")
            if ing_key in DAO_INHIBITORS:
                has_dao_inhibitor = True

            # Check for histamine liberators
            if ing_key in HISTAMINE_LIBERATORS:
                has_liberator = True

        # Adjust for DAO inhibitors (they reduce histamine metabolism by ~30%)
        effective_mg = total_mg
        if has_dao_inhibitor:
            effective_mg = total_mg * 1.3
            warnings.append(
                "Contains DAO inhibitor - histamine will be metabolized more slowly"
            )

        if has_liberator:
            warnings.append(
                "Contains histamine liberator - may trigger additional histamine release"
            )

        # Determine threshold and risk
        if is_sensitive:
            threshold = HISTAMINE_THRESHOLDS["safe_sensitive"]
            threshold_name = "sensitive"
        else:
            threshold = HISTAMINE_THRESHOLDS["caution_general"]
            threshold_name = "general"

        pct_of_threshold = (effective_mg / threshold) * 100 if threshold > 0 else 0

        # Determine risk level
        if effective_mg < threshold * 0.3:
            risk = RiskLevel.NEGLIGIBLE
            level = CompoundLevel.LOW
        elif effective_mg < threshold * 0.6:
            risk = RiskLevel.LOW
            level = CompoundLevel.MEDIUM
        elif effective_mg < threshold:
            risk = RiskLevel.MODERATE
            level = CompoundLevel.HIGH
        elif effective_mg < threshold * 1.5:
            risk = RiskLevel.HIGH
            level = CompoundLevel.HIGH
            warnings.append(f"Histamine exceeds {threshold_name} threshold")
        else:
            risk = RiskLevel.CRITICAL
            level = CompoundLevel.VERY_HIGH
            warnings.append("Histamine significantly exceeds safe levels")

        # Recommendations
        recommendations = []
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            recommendations.append(
                "Consider reducing portion size or choosing alternatives"
            )
            if is_sensitive:
                recommendations.append("Take DAO supplement 15 minutes before eating")
        elif risk == RiskLevel.MODERATE:
            recommendations.append("Monitor for symptoms within 2 hours of eating")

        return CompoundQuantification(
            compound_name="histamine",
            total_mg=round(total_mg, 2),
            level=level,
            risk_level=risk,
            threshold_used=threshold_name,
            threshold_value=threshold,
            percentage_of_threshold=round(pct_of_threshold, 1),
            sources=sources,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _quantify_tyramine(
        self,
        ingredient_data: Dict[str, IngredientData],
        portion_weights: Dict[str, float],
        is_maoi_user: bool,
        is_migraine_prone: bool,
    ) -> CompoundQuantification:
        """Quantify total tyramine with risk assessment."""
        total_mg = 0.0
        sources = []
        warnings = []
        has_aged_food = False

        for ing_name, data in ingredient_data.items():
            weight = portion_weights.get(ing_name, 100.0)

            if data.tyramine_mg:
                amount = (data.tyramine_mg / 100.0) * weight
                total_mg += amount
                sources.append(
                    {"ingredient": data.display_name, "mg": round(amount, 2)}
                )

            if data.is_aged:
                has_aged_food = True

        if has_aged_food:
            warnings.append(
                "Contains aged food - tyramine content may vary significantly"
            )

        # Determine threshold based on user profile
        if is_maoi_user:
            threshold = TYRAMINE_THRESHOLDS["danger_maoi"]
            threshold_name = "MAOI user"
            if total_mg > threshold:
                warnings.append(
                    "CRITICAL: Tyramine level dangerous for MAOI users. "
                    "Risk of hypertensive crisis."
                )
        elif is_migraine_prone:
            threshold = TYRAMINE_THRESHOLDS["safe_migraine"]
            threshold_name = "migraine-prone"
        else:
            threshold = TYRAMINE_THRESHOLDS["safe_general"]
            threshold_name = "general"

        pct_of_threshold = (total_mg / threshold) * 100 if threshold > 0 else 0

        # Determine risk level
        if total_mg < threshold * 0.3:
            risk = RiskLevel.NEGLIGIBLE
            level = CompoundLevel.LOW
        elif total_mg < threshold * 0.6:
            risk = RiskLevel.LOW
            level = CompoundLevel.MEDIUM
        elif total_mg < threshold:
            risk = RiskLevel.MODERATE
            level = CompoundLevel.HIGH
        elif total_mg < threshold * 1.5:
            risk = RiskLevel.HIGH
            level = CompoundLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL
            level = CompoundLevel.VERY_HIGH

        # MAOI users get elevated risk regardless
        if is_maoi_user and total_mg > threshold * 0.5:
            risk = max(risk, RiskLevel.HIGH)
            if total_mg > threshold:
                risk = RiskLevel.CRITICAL

        # Recommendations
        recommendations = []
        if is_maoi_user and risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            recommendations.append("AVOID this meal - hypertensive crisis risk")
        elif risk == RiskLevel.HIGH:
            recommendations.append("Reduce aged/fermented foods in this meal")
        elif is_migraine_prone and risk >= RiskLevel.MODERATE:
            recommendations.append("Monitor for migraine symptoms in next 6-12 hours")

        return CompoundQuantification(
            compound_name="tyramine",
            total_mg=round(total_mg, 2),
            level=level,
            risk_level=risk,
            threshold_used=threshold_name,
            threshold_value=threshold,
            percentage_of_threshold=round(pct_of_threshold, 1),
            sources=sources,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _quantify_fodmap(
        self,
        ingredient_data: Dict[str, IngredientData],
        portion_weights: Dict[str, float],
        is_ibs_sensitive: bool,
    ) -> FodmapQuantification:
        """Quantify FODMAP content with stacking detection."""
        by_type = {}
        warnings = []
        recommendations = []
        high_fodmap_count = 0

        # Count FODMAP types present
        fodmap_types_present = set()

        for ing_name, data in ingredient_data.items():
            if data.fodmap_level == FodmapLevel.HIGH:
                high_fodmap_count += 1

            if data.fodmap_types:
                for fodmap_type in data.fodmap_types:
                    fodmap_types_present.add(fodmap_type)

                    if fodmap_type not in by_type:
                        by_type[fodmap_type] = {
                            "level": "low",
                            "sources": [],
                            "estimated_g": 0.0,
                        }

                    by_type[fodmap_type]["sources"].append(data.display_name)

                    # Upgrade level if high FODMAP
                    if data.fodmap_level == FodmapLevel.HIGH:
                        by_type[fodmap_type]["level"] = "high"
                    elif data.fodmap_level == FodmapLevel.MEDIUM:
                        if by_type[fodmap_type]["level"] == "low":
                            by_type[fodmap_type]["level"] = "medium"

        # Detect FODMAP stacking (multiple types in one meal)
        stacking_warning = len(fodmap_types_present) >= 3

        if stacking_warning:
            warnings.append(
                f"FODMAP stacking detected: {len(fodmap_types_present)} types present. "
                "Cumulative effect may exceed tolerance."
            )

        # Determine overall risk
        if high_fodmap_count == 0:
            risk = RiskLevel.NEGLIGIBLE
        elif high_fodmap_count == 1 and not stacking_warning:
            risk = RiskLevel.LOW
        elif high_fodmap_count <= 2 and not stacking_warning:
            risk = RiskLevel.MODERATE
        elif stacking_warning or high_fodmap_count >= 3:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.MODERATE

        # Elevated risk for IBS-sensitive users
        if is_ibs_sensitive:
            if risk == RiskLevel.MODERATE:
                risk = RiskLevel.HIGH
            elif risk == RiskLevel.LOW:
                risk = RiskLevel.MODERATE
            warnings.append("IBS sensitivity increases FODMAP reaction risk")

        # Recommendations
        if risk >= RiskLevel.HIGH:
            recommendations.append(
                "Consider removing one or more high-FODMAP ingredients"
            )
            recommendations.append(
                "Spacing FODMAP intake by 3+ hours may reduce symptoms"
            )
        elif risk == RiskLevel.MODERATE:
            recommendations.append("Monitor for GI symptoms within 6 hours")

        return FodmapQuantification(
            total_fodmap_types=len(fodmap_types_present),
            high_fodmap_count=high_fodmap_count,
            risk_level=risk,
            by_type=by_type,
            stacking_warning=stacking_warning,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _detect_interactions(
        self,
        ingredients: List[str],
        ingredient_data: Dict[str, IngredientData],
        histamine: CompoundQuantification,
        tyramine: CompoundQuantification,
    ) -> List[str]:
        """Detect cross-compound interactions."""
        warnings = []

        # High histamine + DAO inhibitor interaction
        has_dao_inhibitor = any(
            ing.lower().replace(" ", "_") in DAO_INHIBITORS for ing in ingredients
        )
        if has_dao_inhibitor and histamine.total_mg > 10:
            warnings.append(
                "DAO inhibitor + histamine foods: Reduced histamine metabolism. "
                "Consider spacing meals or avoiding this combination."
            )

        # High tyramine + alcohol interaction
        has_alcohol = any(
            "wine" in ing.lower() or "beer" in ing.lower() or "alcohol" in ing.lower()
            for ing in ingredients
        )
        if has_alcohol and tyramine.total_mg > 10:
            warnings.append(
                "Alcohol + tyramine: Increased migraine risk. "
                "Alcohol also inhibits tyramine metabolism."
            )

        # High histamine + high tyramine (double whammy)
        if histamine.risk_level in (
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ) and tyramine.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            warnings.append(
                "High histamine AND high tyramine: Combined effect may be severe. "
                "Both compounds can cause similar symptoms."
            )

        # Fermented + aged foods accumulation
        fermented_count = sum(
            1 for data in ingredient_data.values() if data.is_fermented
        )
        aged_count = sum(1 for data in ingredient_data.values() if data.is_aged)
        if fermented_count >= 2 or aged_count >= 2:
            warnings.append(
                "Multiple fermented/aged foods: Biogenic amine content varies widely. "
                "Fresher alternatives recommended."
            )

        return warnings

    def _calculate_overall_risk(
        self,
        histamine_risk: RiskLevel,
        tyramine_risk: RiskLevel,
        fodmap_risk: RiskLevel,
        interaction_count: int,
    ) -> RiskLevel:
        """Calculate overall meal risk from component risks."""
        risk_values = {
            RiskLevel.NEGLIGIBLE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        # Take maximum risk level
        max_risk = max(
            risk_values[histamine_risk],
            risk_values[tyramine_risk],
            risk_values[fodmap_risk],
        )

        # Elevate if multiple interactions
        if interaction_count >= 2:
            max_risk = min(4, max_risk + 1)

        # Map back to RiskLevel
        for level, value in risk_values.items():
            if value == max_risk:
                return level

        return RiskLevel.MODERATE

    def _get_timing_recommendation(
        self,
        histamine: CompoundQuantification,
        tyramine: CompoundQuantification,
        fodmap: FodmapQuantification,
        overall_risk: RiskLevel,
    ) -> Optional[str]:
        """Generate meal timing recommendation."""
        if overall_risk == RiskLevel.CRITICAL:
            return "Consider skipping this meal or significantly reducing portions."

        if overall_risk == RiskLevel.HIGH:
            recommendations = []
            if histamine.risk_level >= RiskLevel.HIGH:
                recommendations.append(
                    "eat earlier in the day when DAO activity is higher"
                )
            if fodmap.stacking_warning:
                recommendations.append("wait 4+ hours after previous FODMAP exposure")
            if tyramine.risk_level >= RiskLevel.HIGH:
                recommendations.append("avoid combining with alcohol")

            if recommendations:
                return "Timing suggestion: " + ", ".join(recommendations)

        if overall_risk == RiskLevel.MODERATE:
            return "Best consumed with other low-risk foods. Avoid stacking sensitive ingredients."

        return None

    def calculate_daily_accumulation(
        self,
        meals_today: List[MealCompoundProfile],
        user_daily_limits: Optional[Dict[str, float]] = None,
    ) -> DailyCompoundAccumulation:
        """
        Calculate daily compound accumulation across meals.

        Args:
            meals_today: List of MealCompoundProfile from today's meals
            user_daily_limits: Optional custom limits for user

        Returns:
            Accumulation data with warnings if limits exceeded
        """
        accumulation = DailyCompoundAccumulation(
            date=datetime.now(),
            meals_analyzed=len(meals_today),
        )

        for meal in meals_today:
            accumulation.histamine_total_mg += meal.histamine.total_mg
            accumulation.tyramine_total_mg += meal.tyramine.total_mg
            if meal.fodmap.high_fodmap_count > 0:
                accumulation.fodmap_exposures += 1

        # Check against daily limits
        histamine_daily = (
            user_daily_limits.get("histamine")
            if user_daily_limits
            else HISTAMINE_THRESHOLDS["daily_limit_general"]
        )
        tyramine_daily = (
            user_daily_limits.get("tyramine")
            if user_daily_limits
            else TYRAMINE_THRESHOLDS["daily_limit_general"]
        )

        if accumulation.histamine_total_mg > histamine_daily:
            accumulation.warnings.append(
                f"Daily histamine ({accumulation.histamine_total_mg:.1f}mg) "
                f"exceeds limit ({histamine_daily}mg)"
            )

        if accumulation.tyramine_total_mg > tyramine_daily:
            accumulation.warnings.append(
                f"Daily tyramine ({accumulation.tyramine_total_mg:.1f}mg) "
                f"exceeds limit ({tyramine_daily}mg)"
            )

        if accumulation.fodmap_exposures >= 3:
            accumulation.warnings.append(
                f"High FODMAP exposure in {accumulation.fodmap_exposures} meals. "
                "Consider FODMAP-free meals for remainder of day."
            )

        return accumulation

    def get_compound_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Return all compound thresholds for reference."""
        return {
            "histamine": HISTAMINE_THRESHOLDS,
            "tyramine": TYRAMINE_THRESHOLDS,
            "fodmap": FODMAP_THRESHOLDS,
        }

    def get_dao_inhibitors(self) -> List[str]:
        """Return list of DAO inhibitor foods."""
        return list(DAO_INHIBITORS)

    def get_histamine_liberators(self) -> List[str]:
        """Return list of histamine liberator foods."""
        return list(HISTAMINE_LIBERATORS)


# Singleton instance
compound_quantification_service = CompoundQuantificationService()
