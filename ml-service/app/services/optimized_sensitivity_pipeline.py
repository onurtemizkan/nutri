"""
Optimized Food Sensitivity Detection Pipeline.

This module integrates all optimized components into a unified pipeline:
- NLP Ingredient Extraction (spaCy-based)
- Advanced Ingredient Matching (Trie + BK-Tree)
- Redis Caching Layer
- Bayesian Sensitivity Inference
- LSTM Temporal Pattern Analysis

Architecture:
1. Input Processing: NLP extraction + ingredient matching
2. Historical Context: Cache lookup + prior retrieval
3. Real-time Analysis: HRV processing + pattern detection
4. Bayesian Update: Evidence integration + belief update
5. Deep Learning: LSTM temporal analysis
6. Result Synthesis: Combined inference + recommendations

Performance Targets:
- End-to-end latency: <500ms (cached), <2s (uncached)
- Throughput: 100+ requests/second
- Accuracy: >85% sensitivity detection
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from collections import defaultdict
import logging

# Import optimized components
from .nlp_ingredient_extractor import (
    NLPIngredientExtractor,
    ExtractedIngredient,
    get_nlp_extractor,
)
from .advanced_ingredient_matcher import (
    AdvancedIngredientMatcher,
)
from .sensitivity_cache_service import (
    SensitivityCacheService,
    get_cache_service,
)
from .bayesian_sensitivity_engine import (
    BayesianSensitivityEngine,
    ExposureEvidence,
    BayesianBelief,
)
from .lstm_temporal_analyzer import (
    LSTMTemporalPatternAnalyzer,
    AnalysisResult as LSTMAnalysisResult,
    TemporalPattern,
    get_lstm_analyzer,
)

# Import existing services
from app.data.allergen_database import (
    IngredientData,
    get_ingredient,
    get_all_ingredients,
)
from app.models.sensitivity import (
    CompoundLevel,
)
from .compound_quantification_service import (
    CompoundQuantificationService,
    compound_quantification_service,
)

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis mode selection."""

    QUICK = "quick"  # Fast, uses cache only
    STANDARD = "standard"  # Balanced speed/accuracy
    COMPREHENSIVE = "comprehensive"  # Full analysis, all components
    REALTIME = "realtime"  # Streaming/continuous analysis


class ConfidenceLevel(Enum):
    """Confidence classification."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class IngredientAnalysis:
    """Analysis result for a single ingredient."""

    name: str
    matched_name: Optional[str]
    match_confidence: float
    allergens: List[str]
    trigger_types: List[str]
    compounds: Dict[str, float]
    is_known_trigger: bool
    sensitivity_probability: float
    historical_reactions: int


@dataclass
class HRVAnalysis:
    """HRV-specific analysis results."""

    baseline_hrv: float
    current_hrv: float
    percent_change: float
    is_significant_change: bool
    pattern_detected: Optional[str]
    temporal_patterns: List[TemporalPattern]
    recovery_prediction_hours: float


@dataclass
class SensitivityAnalysisResult:
    """Complete sensitivity analysis result."""

    # Request info
    request_id: str
    user_id: str
    timestamp: datetime
    analysis_mode: AnalysisMode
    processing_time_ms: float

    # Ingredient analysis
    extracted_ingredients: List[ExtractedIngredient]
    matched_ingredients: List[IngredientAnalysis]
    total_allergens_detected: int
    total_compounds_quantified: Dict[str, float]

    # HRV analysis
    hrv_analysis: Optional[HRVAnalysis]

    # Bayesian inference
    bayesian_beliefs: Dict[str, BayesianBelief]
    posterior_sensitivity_probability: float
    evidence_strength: float

    # LSTM analysis
    lstm_result: Optional[LSTMAnalysisResult]

    # Combined assessment
    overall_risk_score: float
    confidence_level: ConfidenceLevel
    primary_triggers: List[str]
    recommendations: List[str]
    warnings: List[str]

    # Metadata
    cache_hits: int
    components_used: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "analysis_mode": self.analysis_mode.value,
            "processing_time_ms": self.processing_time_ms,
            "ingredients": {
                "extracted_count": len(self.extracted_ingredients),
                "matched": [
                    {
                        "name": i.name,
                        "matched_name": i.matched_name,
                        "confidence": i.match_confidence,
                        "is_trigger": i.is_known_trigger,
                        "allergens": i.allergens,
                        "sensitivity_probability": i.sensitivity_probability,
                    }
                    for i in self.matched_ingredients
                ],
                "total_allergens": self.total_allergens_detected,
            },
            "compounds": self.total_compounds_quantified,
            "hrv_analysis": (
                {
                    "baseline": self.hrv_analysis.baseline_hrv,
                    "current": self.hrv_analysis.current_hrv,
                    "percent_change": self.hrv_analysis.percent_change,
                    "significant_change": self.hrv_analysis.is_significant_change,
                    "pattern": self.hrv_analysis.pattern_detected,
                    "recovery_hours": self.hrv_analysis.recovery_prediction_hours,
                }
                if self.hrv_analysis
                else None
            ),
            "bayesian_inference": {
                "posterior_probability": self.posterior_sensitivity_probability,
                "evidence_strength": self.evidence_strength,
                "beliefs": {
                    k: {
                        "mean": v.mean_probability,
                        "confidence_interval": [v.lower_ci, v.upper_ci],
                    }
                    for k, v in self.bayesian_beliefs.items()
                },
            },
            "lstm_analysis": (self.lstm_result.to_dict() if self.lstm_result else None),
            "assessment": {
                "overall_risk_score": self.overall_risk_score,
                "confidence_level": self.confidence_level.value,
                "primary_triggers": self.primary_triggers,
            },
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "metadata": {
                "cache_hits": self.cache_hits,
                "components_used": self.components_used,
            },
        }


class OptimizedSensitivityPipeline:
    """
    Unified pipeline for optimized food sensitivity detection.

    Orchestrates all components:
    - NLP extraction for ingredient parsing
    - Advanced matching for ingredient lookup
    - Redis caching for fast retrieval
    - Bayesian engine for probabilistic inference
    - LSTM analyzer for temporal patterns
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        enable_lstm: bool = True,
        enable_caching: bool = True,
        enable_nlp: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            redis_url: Redis connection URL
            enable_lstm: Enable LSTM temporal analysis
            enable_caching: Enable Redis caching
            enable_nlp: Enable NLP ingredient extraction
        """
        self.redis_url = redis_url
        self.enable_lstm = enable_lstm
        self.enable_caching = enable_caching
        self.enable_nlp = enable_nlp

        # Component instances (initialized lazily)
        self._nlp_extractor: Optional[NLPIngredientExtractor] = None
        self._ingredient_matcher: Optional[AdvancedIngredientMatcher] = None
        self._cache_service: Optional[SensitivityCacheService] = None
        self._bayesian_engine: Optional[BayesianSensitivityEngine] = None
        self._lstm_analyzer: Optional[LSTMTemporalPatternAnalyzer] = None
        self._ingredient_db: Dict[str, IngredientData] = {}
        self._compound_service: Optional[CompoundQuantificationService] = None

        # User-specific engines (keyed by user_id)
        self._user_engines: Dict[str, BayesianSensitivityEngine] = {}

        # Initialization state
        self._initialized = False

        # Performance metrics
        self._metrics = defaultdict(list)

    async def initialize(self) -> bool:
        """Initialize all pipeline components."""
        logger.info("Initializing optimized sensitivity pipeline...")
        start_time = time.time()

        try:
            # Initialize ingredient database from data module
            self._ingredient_db = get_all_ingredients()
            logger.info(f"Loaded {len(self._ingredient_db)} ingredients")

            # Initialize ingredient matcher
            self._ingredient_matcher = AdvancedIngredientMatcher()
            self._build_matcher_index()
            logger.info("Ingredient matcher initialized")

            # Initialize NLP extractor
            if self.enable_nlp:
                self._nlp_extractor = get_nlp_extractor()
                # Set food vocabulary from ingredient database
                food_terms = list(self._ingredient_db.keys())
                self._nlp_extractor.set_food_vocabulary(food_terms)
                logger.info("NLP extractor initialized")

            # Initialize cache service
            if self.enable_caching:
                self._cache_service = await get_cache_service()
                # Warm cache with common allergens
                await self._warm_cache()
                logger.info("Cache service initialized")

            # Initialize Bayesian engine (default instance)
            self._bayesian_engine = BayesianSensitivityEngine()
            logger.info("Bayesian engine initialized")

            # Initialize LSTM analyzer
            if self.enable_lstm:
                self._lstm_analyzer = get_lstm_analyzer()
                logger.info("LSTM analyzer initialized")

            # Initialize compound service
            self._compound_service = compound_quantification_service
            logger.info("Compound quantification service initialized")

            self._initialized = True
            init_time = (time.time() - start_time) * 1000
            logger.info(f"Pipeline initialized in {init_time:.0f}ms")
            return True

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def _build_matcher_index(self) -> None:
        """Build ingredient matcher index from ingredient database."""
        if not self._ingredient_matcher or not self._ingredient_db:
            return

        # Add all known ingredients
        for name, data in self._ingredient_db.items():
            # Extract allergen info
            allergen_types = (
                [a.allergen.value for a in data.allergens] if data.allergens else []
            )

            self._ingredient_matcher.add_ingredient(
                name,
                {
                    "category": data.category.value if data.category else "unknown",
                    "allergens": allergen_types,
                    "histamine_level": (
                        data.histamine_level.value if data.histamine_level else None
                    ),
                    "fodmap_level": (
                        data.fodmap_level.value if data.fodmap_level else None
                    ),
                },
            )

            # Add name variants as aliases
            for variant in data.name_variants:
                self._ingredient_matcher.add_ingredient(
                    variant,
                    {
                        "canonical_name": name,
                        "category": data.category.value if data.category else "unknown",
                    },
                )

    async def _warm_cache(self) -> None:
        """Pre-populate cache with common data."""
        if not self._cache_service or not self._ingredient_db:
            return

        # Cache ingredients with allergens (common triggers)
        count = 0
        for name, data in self._ingredient_db.items():
            if data.allergens and count < 100:
                await self._cache_service.set_ingredient(
                    name,
                    {
                        "name": name,
                        "category": data.category.value if data.category else "unknown",
                        "allergens": [a.allergen.value for a in data.allergens],
                        "histamine_mg": data.histamine_mg,
                        "histamine_level": (
                            data.histamine_level.value if data.histamine_level else None
                        ),
                    },
                )
                count += 1

    def _get_user_engine(self, user_id: str) -> BayesianSensitivityEngine:
        """Get or create user-specific Bayesian engine."""
        if user_id not in self._user_engines:
            self._user_engines[user_id] = BayesianSensitivityEngine()
        return self._user_engines[user_id]

    async def analyze(
        self,
        user_id: str,
        food_text: Optional[str] = None,
        ingredients: Optional[List[str]] = None,
        hrv_data: Optional[List[Dict[str, float]]] = None,
        meal_timestamp: Optional[datetime] = None,
        mode: AnalysisMode = AnalysisMode.STANDARD,
    ) -> SensitivityAnalysisResult:
        """
        Run complete sensitivity analysis.

        Args:
            user_id: User identifier
            food_text: Free-text food description (parsed by NLP)
            ingredients: Pre-extracted ingredient list
            hrv_data: HRV measurements for temporal analysis
            meal_timestamp: When the meal was consumed
            mode: Analysis mode (quick/standard/comprehensive)

        Returns:
            Complete SensitivityAnalysisResult
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        cache_hits = 0
        components_used = []
        warnings = []

        if not self._initialized:
            if not await self.initialize():
                return self._create_error_result(
                    request_id, user_id, "Pipeline initialization failed"
                )

        # Step 1: Extract ingredients from text
        extracted_ingredients = []
        if food_text and self._nlp_extractor and self.enable_nlp:
            components_used.append("nlp_extractor")
            extraction_result = self._nlp_extractor.extract(food_text)
            extracted_ingredients = extraction_result.ingredients
            if extraction_result.warnings:
                warnings.extend(extraction_result.warnings)

        # Add explicit ingredients
        if ingredients:
            for ing_name in ingredients:
                extracted_ingredients.append(
                    ExtractedIngredient(
                        name=ing_name,
                        original_text=ing_name,
                        confidence=1.0,
                        start_char=0,
                        end_char=len(ing_name),
                    )
                )

        # Step 2: Match ingredients to database
        matched_ingredients = []
        total_allergens = 0
        total_compounds: Dict[str, float] = defaultdict(float)

        if self._ingredient_matcher:
            components_used.append("ingredient_matcher")

            for extracted in extracted_ingredients:
                # Try cache first
                cached_data = None
                if self._cache_service and self.enable_caching:
                    cached_data = await self._cache_service.get_ingredient(
                        extracted.name
                    )
                    if cached_data:
                        cache_hits += 1

                # Match ingredient
                matches = self._ingredient_matcher.match(
                    extracted.name,
                    threshold=0.7 if mode == AnalysisMode.QUICK else 0.6,
                    max_results=3,
                )

                if matches:
                    best_match = matches[0]

                    # Get ingredient data from database
                    ingredient_data = get_ingredient(best_match.ingredient)

                    # Extract allergen labels
                    allergen_labels = []
                    trigger_types = []
                    if ingredient_data and ingredient_data.allergens:
                        allergen_labels = [
                            a.allergen.value for a in ingredient_data.allergens
                        ]
                        trigger_types = allergen_labels  # Same as allergen types

                    # Quantify compounds
                    compound_quantities = {}
                    if (
                        self._compound_service
                        and extracted.quantity
                        and mode != AnalysisMode.QUICK
                    ):
                        components_used.append("compound_quantification")
                        compound_quantities = self._compound_service.quantify_compounds(
                            best_match.ingredient,
                            extracted.quantity.normalized_grams or 100.0,
                        )
                        for compound, amount in compound_quantities.items():
                            total_compounds[compound] += amount

                    # Get historical reaction count from Bayesian engine
                    user_engine = self._get_user_engine(user_id)
                    belief = user_engine.get_belief("ingredient", best_match.ingredient)
                    historical_reactions = int(belief.alpha - 1)  # α - 1 = successes

                    # Determine if this is a known trigger
                    is_trigger = bool(
                        ingredient_data
                        and (
                            ingredient_data.allergens
                            or ingredient_data.histamine_level
                            in [CompoundLevel.HIGH, CompoundLevel.VERY_HIGH]
                            or ingredient_data.is_histamine_liberator
                        )
                    )

                    ing_analysis = IngredientAnalysis(
                        name=extracted.name,
                        matched_name=best_match.ingredient,
                        match_confidence=best_match.similarity,
                        allergens=allergen_labels,
                        trigger_types=trigger_types,
                        compounds=compound_quantities,
                        is_known_trigger=is_trigger,
                        sensitivity_probability=belief.mean_probability,
                        historical_reactions=historical_reactions,
                    )
                    matched_ingredients.append(ing_analysis)

                    if is_trigger:
                        total_allergens += 1
                else:
                    # No match found
                    matched_ingredients.append(
                        IngredientAnalysis(
                            name=extracted.name,
                            matched_name=None,
                            match_confidence=0.0,
                            allergens=[],
                            trigger_types=[],
                            compounds={},
                            is_known_trigger=False,
                            sensitivity_probability=0.0,
                            historical_reactions=0,
                        )
                    )

        # Step 3: HRV analysis
        hrv_analysis = None
        lstm_result = None

        if hrv_data and len(hrv_data) >= 5:
            components_used.append("hrv_analysis")

            # Calculate basic HRV metrics
            hrv_values = [
                d.get("hrv_sdnn", 0) or d.get("hrv_rmssd", 0) for d in hrv_data
            ]
            if hrv_values:
                baseline = sum(hrv_values[: min(5, len(hrv_values))]) / min(
                    5, len(hrv_values)
                )
                current = hrv_values[-1]
                percent_change = (
                    ((current - baseline) / baseline * 100) if baseline > 0 else 0
                )

                # Significant change threshold: >15% drop
                is_significant = percent_change < -15

                hrv_analysis = HRVAnalysis(
                    baseline_hrv=baseline,
                    current_hrv=current,
                    percent_change=percent_change,
                    is_significant_change=is_significant,
                    pattern_detected=None,
                    temporal_patterns=[],
                    recovery_prediction_hours=0.0,
                )

                # LSTM temporal analysis (for comprehensive mode)
                if (
                    self._lstm_analyzer
                    and self.enable_lstm
                    and mode in [AnalysisMode.STANDARD, AnalysisMode.COMPREHENSIVE]
                    and len(hrv_data) >= 10
                ):
                    components_used.append("lstm_analyzer")
                    lstm_result = self._lstm_analyzer.analyze(hrv_data)

                    if lstm_result:
                        hrv_analysis.temporal_patterns = lstm_result.detected_patterns
                        hrv_analysis.recovery_prediction_hours = (
                            lstm_result.predicted_recovery_time_hours
                        )
                        if lstm_result.detected_patterns:
                            hrv_analysis.pattern_detected = (
                                lstm_result.detected_patterns[0].pattern_type.value
                            )

        # Step 4: Bayesian inference
        bayesian_beliefs: Dict[str, BayesianBelief] = {}
        posterior_probability = 0.0
        evidence_strength = 0.0
        components_used.append("bayesian_engine")

        user_engine = self._get_user_engine(user_id)

        # Update beliefs for each trigger
        for ing in matched_ingredients:
            if ing.is_known_trigger and ing.matched_name:
                # Create evidence from HRV
                hrv_response = False
                if hrv_analysis and hrv_analysis.is_significant_change:
                    hrv_response = True

                evidence = ExposureEvidence(
                    timestamp=meal_timestamp or datetime.now(),
                    trigger_type="ingredient",
                    trigger_name=ing.matched_name,
                    dose_grams=100.0,  # Default if not specified
                    hrv_response=hrv_response,
                    confounders={},
                )

                # Update belief
                belief = user_engine.update_belief(evidence, ing.matched_name)
                bayesian_beliefs[ing.matched_name] = belief

                # Track max probability
                if belief.mean_probability > posterior_probability:
                    posterior_probability = belief.mean_probability

        # Calculate evidence strength
        if bayesian_beliefs:
            evidence_strength = user_engine.compute_evidence_strength()

        # Step 5: Synthesize results
        overall_risk = self._calculate_overall_risk(
            matched_ingredients,
            hrv_analysis,
            bayesian_beliefs,
            lstm_result,
        )

        confidence_level = self._assess_confidence(
            len(extracted_ingredients),
            len(matched_ingredients),
            hrv_analysis,
            bayesian_beliefs,
            cache_hits,
        )

        primary_triggers = self._identify_primary_triggers(
            matched_ingredients,
            bayesian_beliefs,
        )

        recommendations = self._generate_recommendations(
            matched_ingredients,
            hrv_analysis,
            bayesian_beliefs,
            lstm_result,
            overall_risk,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        self._record_metric("processing_time_ms", processing_time)

        return SensitivityAnalysisResult(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(),
            analysis_mode=mode,
            processing_time_ms=processing_time,
            extracted_ingredients=extracted_ingredients,
            matched_ingredients=matched_ingredients,
            total_allergens_detected=total_allergens,
            total_compounds_quantified=dict(total_compounds),
            hrv_analysis=hrv_analysis,
            bayesian_beliefs=bayesian_beliefs,
            posterior_sensitivity_probability=posterior_probability,
            evidence_strength=evidence_strength,
            lstm_result=lstm_result,
            overall_risk_score=overall_risk,
            confidence_level=confidence_level,
            primary_triggers=primary_triggers,
            recommendations=recommendations,
            warnings=warnings,
            cache_hits=cache_hits,
            components_used=list(set(components_used)),
        )

    def _calculate_overall_risk(
        self,
        matched_ingredients: List[IngredientAnalysis],
        hrv_analysis: Optional[HRVAnalysis],
        bayesian_beliefs: Dict[str, BayesianBelief],
        lstm_result: Optional[LSTMAnalysisResult],
    ) -> float:
        """Calculate overall risk score (0-1)."""
        scores = []
        weights = []

        # Ingredient-based risk
        if matched_ingredients:
            max_ing_prob = max(
                (i.sensitivity_probability for i in matched_ingredients), default=0.0
            )
            scores.append(max_ing_prob)
            weights.append(0.3)

        # HRV-based risk
        if hrv_analysis:
            hrv_risk = 0.0
            if hrv_analysis.is_significant_change:
                hrv_risk = min(1.0, abs(hrv_analysis.percent_change) / 30)
            scores.append(hrv_risk)
            weights.append(0.25)

        # Bayesian-based risk
        if bayesian_beliefs:
            max_belief = max(
                (b.mean_probability for b in bayesian_beliefs.values()), default=0.0
            )
            scores.append(max_belief)
            weights.append(0.3)

        # LSTM-based risk
        if lstm_result:
            scores.append(lstm_result.overall_sensitivity_score)
            weights.append(0.15)

        # Weighted average
        if scores and weights:
            total_weight = sum(weights)
            return sum(s * w for s, w in zip(scores, weights)) / total_weight

        return 0.0

    def _assess_confidence(
        self,
        num_extracted: int,
        num_matched: int,
        hrv_analysis: Optional[HRVAnalysis],
        bayesian_beliefs: Dict[str, BayesianBelief],
        cache_hits: int,
    ) -> ConfidenceLevel:
        """Assess confidence level of analysis."""
        score = 0.0

        # Ingredient matching quality
        if num_extracted > 0:
            match_rate = num_matched / num_extracted
            score += match_rate * 0.25

        # HRV data quality
        if hrv_analysis:
            score += 0.25

        # Bayesian evidence
        if bayesian_beliefs:
            avg_confidence = sum(
                1 - (b.upper_ci - b.lower_ci) for b in bayesian_beliefs.values()
            ) / len(bayesian_beliefs)
            score += avg_confidence * 0.25

        # Historical data (cache hits indicate established patterns)
        if cache_hits > 0:
            score += min(0.25, cache_hits * 0.05)

        # Map to confidence level
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MODERATE
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _identify_primary_triggers(
        self,
        matched_ingredients: List[IngredientAnalysis],
        bayesian_beliefs: Dict[str, BayesianBelief],
    ) -> List[str]:
        """Identify primary sensitivity triggers."""
        triggers = []

        # Combine ingredient and Bayesian signals
        candidates = []

        for ing in matched_ingredients:
            if ing.is_known_trigger and ing.matched_name:
                belief = bayesian_beliefs.get(ing.matched_name)
                prob = (
                    belief.mean_probability if belief else ing.sensitivity_probability
                )

                if prob > 0.3:  # Threshold for consideration
                    candidates.append((ing.matched_name, prob))

        # Sort by probability and take top 3
        candidates.sort(key=lambda x: x[1], reverse=True)
        triggers = [name for name, _ in candidates[:3]]

        return triggers

    def _generate_recommendations(
        self,
        matched_ingredients: List[IngredientAnalysis],
        hrv_analysis: Optional[HRVAnalysis],
        bayesian_beliefs: Dict[str, BayesianBelief],
        lstm_result: Optional[LSTMAnalysisResult],
        overall_risk: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Risk-based recommendations
        if overall_risk > 0.7:
            recommendations.append(
                "High sensitivity risk detected. Consider avoiding identified triggers."
            )
        elif overall_risk > 0.4:
            recommendations.append(
                "Moderate sensitivity signals observed. Monitor symptoms carefully."
            )

        # Ingredient-specific recommendations
        high_risk_ingredients = [
            i for i in matched_ingredients if i.sensitivity_probability > 0.5
        ]
        if high_risk_ingredients:
            names = [i.matched_name for i in high_risk_ingredients[:3]]
            recommendations.append(f"Likely triggers: {', '.join(filter(None, names))}")

        # HRV-based recommendations
        if hrv_analysis and hrv_analysis.is_significant_change:
            recommendations.append(
                f"HRV dropped {abs(hrv_analysis.percent_change):.0f}% - "
                "indicates possible stress response"
            )
            if hrv_analysis.recovery_prediction_hours > 0:
                recommendations.append(
                    f"Expected recovery: ~{hrv_analysis.recovery_prediction_hours:.1f} hours"
                )

        # LSTM-based recommendations
        if lstm_result and lstm_result.recommendations:
            recommendations.extend(lstm_result.recommendations[:2])

        # Compound-specific recommendations
        for ing in matched_ingredients:
            if "histamine" in ing.compounds and ing.compounds["histamine"] > 10:
                recommendations.append(
                    "High histamine content detected - relevant for histamine intolerance"
                )
                break

        # Default if no specific recommendations
        if not recommendations:
            recommendations.append(
                "No significant sensitivity patterns detected. Continue monitoring."
            )

        return recommendations

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()[:12]

    def _create_error_result(
        self, request_id: str, user_id: str, error: str
    ) -> SensitivityAnalysisResult:
        """Create error result."""
        return SensitivityAnalysisResult(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(),
            analysis_mode=AnalysisMode.QUICK,
            processing_time_ms=0,
            extracted_ingredients=[],
            matched_ingredients=[],
            total_allergens_detected=0,
            total_compounds_quantified={},
            hrv_analysis=None,
            bayesian_beliefs={},
            posterior_sensitivity_probability=0.0,
            evidence_strength=0.0,
            lstm_result=None,
            overall_risk_score=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            primary_triggers=[],
            recommendations=[],
            warnings=[error],
            cache_hits=0,
            components_used=[],
        )

    def _record_metric(self, name: str, value: float) -> None:
        """Record performance metric."""
        self._metrics[name].append(value)
        # Keep last 1000 measurements
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get pipeline metrics."""
        result = {}
        for name, values in self._metrics.items():
            if values:
                result[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health status."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "components": {},
        }

        # Check each component
        if self._nlp_extractor:
            health["components"]["nlp_extractor"] = "ok"
        else:
            health["components"]["nlp_extractor"] = (
                "disabled" if not self.enable_nlp else "error"
            )

        if self._ingredient_matcher:
            health["components"]["ingredient_matcher"] = "ok"

        if self._cache_service:
            cache_health = await self._cache_service.health_check()
            health["components"]["cache_service"] = cache_health["status"]
        else:
            health["components"]["cache_service"] = "disabled"

        if self._bayesian_engine:
            health["components"]["bayesian_engine"] = "ok"

        if self._lstm_analyzer:
            health["components"]["lstm_analyzer"] = "ok"
        else:
            health["components"]["lstm_analyzer"] = (
                "disabled" if not self.enable_lstm else "error"
            )

        # Overall status
        if any(v == "error" for v in health["components"].values()):
            health["status"] = "degraded"

        if not self._initialized:
            health["status"] = "not_initialized"

        health["metrics"] = self.get_metrics()

        return health


# ==================== Singleton Instance ====================

_pipeline: Optional[OptimizedSensitivityPipeline] = None


async def get_pipeline() -> OptimizedSensitivityPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline

    if _pipeline is None:
        _pipeline = OptimizedSensitivityPipeline()
        await _pipeline.initialize()

    return _pipeline


# ==================== Convenience Functions ====================


async def analyze_food_sensitivity(
    user_id: str,
    food_text: Optional[str] = None,
    ingredients: Optional[List[str]] = None,
    hrv_data: Optional[List[Dict[str, float]]] = None,
    mode: str = "standard",
) -> Dict[str, Any]:
    """
    Convenience function for food sensitivity analysis.

    Args:
        user_id: User identifier
        food_text: Free-text food description
        ingredients: Pre-extracted ingredient list
        hrv_data: HRV measurements
        mode: Analysis mode (quick/standard/comprehensive)

    Returns:
        Analysis result dictionary
    """
    pipeline = await get_pipeline()

    mode_enum = AnalysisMode(mode.lower()) if mode else AnalysisMode.STANDARD

    result = await pipeline.analyze(
        user_id=user_id,
        food_text=food_text,
        ingredients=ingredients,
        hrv_data=hrv_data,
        mode=mode_enum,
    )

    return result.to_dict()
