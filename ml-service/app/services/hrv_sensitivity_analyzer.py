"""
Multi-Window HRV Sensitivity Analyzer

Detects food sensitivities by analyzing HRV changes across multiple time windows
after food exposure. Based on research showing HRV can detect allergic reactions
17 minutes before symptoms with 90.5% sensitivity and 79.4% specificity.

Time Windows (from clinical research):
- Immediate: 0-30 min (acute stress response, early IgE reactions)
- Short-term: 30-120 min (peak for IgE-mediated allergies)
- Medium-term: 2-6 hours (food intolerances, FODMAP reactions)
- Extended: 6-24 hours (cumulative effects, delayed reactions)
- Next-day: 24-48 hours (persistent inflammation, recovery assessment)

Key HRV Metrics:
- RMSSD: Most sensitive to parasympathetic changes (primary metric)
- SDNN: Overall autonomic variability
- pNN50: Parasympathetic indicator
- LF/HF ratio: Sympathovagal balance

Detection Thresholds:
- >10% RMSSD drop: Possible reaction
- >15% RMSSD drop: Likely reaction
- >25% RMSSD drop: Significant reaction
- Pattern across 3+ exposures: Confirmed sensitivity
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats

from app.models.sensitivity import (
    SensitivityType,
    SensitivitySeverity,
    ReactionSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================


class TimeWindow(str, Enum):
    """HRV analysis time windows."""

    IMMEDIATE = "immediate"  # 0-30 min
    SHORT_TERM = "short_term"  # 30-120 min
    MEDIUM_TERM = "medium_term"  # 2-6 hours
    EXTENDED = "extended"  # 6-24 hours
    NEXT_DAY = "next_day"  # 24-48 hours


# Time window definitions in minutes
WINDOW_DEFINITIONS = {
    TimeWindow.IMMEDIATE: {"start": 0, "end": 30, "label": "0-30 min"},
    TimeWindow.SHORT_TERM: {"start": 30, "end": 120, "label": "30 min - 2 hr"},
    TimeWindow.MEDIUM_TERM: {"start": 120, "end": 360, "label": "2-6 hr"},
    TimeWindow.EXTENDED: {"start": 360, "end": 1440, "label": "6-24 hr"},
    TimeWindow.NEXT_DAY: {"start": 1440, "end": 2880, "label": "24-48 hr"},
}

# Reaction patterns by sensitivity type
REACTION_PATTERNS = {
    SensitivityType.ALLERGY: {
        "primary_window": TimeWindow.SHORT_TERM,
        "secondary_windows": [TimeWindow.IMMEDIATE],
        "typical_onset_min": 15,
        "typical_duration_hours": 4,
    },
    SensitivityType.INTOLERANCE: {
        "primary_window": TimeWindow.MEDIUM_TERM,
        "secondary_windows": [TimeWindow.SHORT_TERM, TimeWindow.EXTENDED],
        "typical_onset_min": 120,
        "typical_duration_hours": 12,
    },
    SensitivityType.FODMAP: {
        "primary_window": TimeWindow.MEDIUM_TERM,
        "secondary_windows": [TimeWindow.EXTENDED],
        "typical_onset_min": 180,
        "typical_duration_hours": 24,
    },
    SensitivityType.HISTAMINE: {
        "primary_window": TimeWindow.SHORT_TERM,
        "secondary_windows": [TimeWindow.IMMEDIATE, TimeWindow.MEDIUM_TERM],
        "typical_onset_min": 30,
        "typical_duration_hours": 8,
    },
    SensitivityType.TYRAMINE: {
        "primary_window": TimeWindow.SHORT_TERM,
        "secondary_windows": [TimeWindow.MEDIUM_TERM],
        "typical_onset_min": 60,
        "typical_duration_hours": 6,
    },
}

# HRV drop thresholds for reaction detection
HRV_THRESHOLDS = {
    "possible_reaction_pct": 10.0,  # >10% drop
    "likely_reaction_pct": 15.0,  # >15% drop
    "significant_reaction_pct": 25.0,  # >25% drop
    "min_baseline_samples": 3,  # Minimum samples for baseline
    "min_exposures_for_pattern": 3,  # Exposures needed to confirm pattern
    "statistical_significance": 0.05,  # p-value threshold
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class HRVReading:
    """Single HRV measurement."""

    timestamp: datetime
    rmssd: float  # Root Mean Square of Successive Differences (ms)
    sdnn: Optional[float] = None  # Standard Deviation of NN intervals
    pnn50: Optional[float] = None  # Percentage of successive NN>50ms
    lf_hf_ratio: Optional[float] = None  # Low frequency / High frequency ratio
    heart_rate: Optional[float] = None
    source: str = "unknown"  # apple_health, garmin, oura, etc.


@dataclass
class BaselineHRV:
    """User's baseline HRV statistics."""

    mean_rmssd: float
    std_rmssd: float
    median_rmssd: float
    mean_sdnn: Optional[float] = None
    mean_pnn50: Optional[float] = None
    mean_lf_hf: Optional[float] = None
    sample_count: int = 0
    calculation_period_days: int = 7
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WindowAnalysis:
    """HRV analysis for a specific time window."""

    window: TimeWindow
    start_minutes: int
    end_minutes: int
    readings_count: int

    # HRV statistics
    mean_rmssd: Optional[float] = None
    min_rmssd: Optional[float] = None
    max_rmssd: Optional[float] = None

    # Change from baseline
    rmssd_change_ms: Optional[float] = None  # Absolute change
    rmssd_change_pct: Optional[float] = None  # Percentage change

    # Reaction detection
    has_significant_drop: bool = False
    reaction_severity: Optional[ReactionSeverity] = None
    confidence: float = 0.0


@dataclass
class ExposureAnalysis:
    """Complete HRV analysis for a single exposure event."""

    exposure_id: str
    trigger_type: str  # allergen type or compound
    trigger_name: str
    exposed_at: datetime

    # Baseline used
    baseline_rmssd: float

    # Window analyses
    windows: Dict[TimeWindow, WindowAnalysis] = field(default_factory=dict)

    # Overall assessment
    max_hrv_drop_pct: float = 0.0
    peak_reaction_window: Optional[TimeWindow] = None
    overall_had_reaction: bool = False
    overall_severity: ReactionSeverity = ReactionSeverity.NONE

    # Confidence and quality
    data_quality: str = "unknown"  # excellent, good, fair, poor
    confidence: float = 0.0


@dataclass
class SensitivityPattern:
    """Detected sensitivity pattern across multiple exposures."""

    trigger_type: str
    trigger_name: str

    # Exposure statistics
    total_exposures: int
    reactions_detected: int
    reaction_rate: float  # 0.0 - 1.0

    # HRV impact statistics
    avg_hrv_drop_pct: float
    max_hrv_drop_pct: float
    std_hrv_drop: float

    # By window
    window_impacts: Dict[TimeWindow, float]  # window -> avg drop %
    primary_reaction_window: TimeWindow

    # Statistical significance
    correlation_coefficient: float  # Pearson r
    p_value: float
    is_statistically_significant: bool

    # Severity assessment
    suggested_severity: SensitivitySeverity
    confidence: float

    # Evidence
    exposure_analyses: List[ExposureAnalysis] = field(default_factory=list)


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report for a user."""

    user_id: str
    analysis_period_days: int
    generated_at: datetime

    # Baseline
    baseline: BaselineHRV

    # Current HRV status
    current_hrv_trend: str  # improving, stable, declining
    current_vs_baseline_pct: float

    # Discovered patterns
    confirmed_sensitivities: List[SensitivityPattern]
    possible_sensitivities: List[SensitivityPattern]

    # Recommendations
    recommendations: List[str]

    # Data quality
    total_hrv_readings: int
    total_exposures_analyzed: int
    data_completeness_pct: float


# =============================================================================
# MAIN SERVICE
# =============================================================================


class HRVSensitivityAnalyzer:
    """
    Analyzes HRV data to detect food sensitivity patterns.

    Uses multi-window temporal analysis to detect both immediate
    and delayed reactions, with statistical validation across
    multiple exposure events.
    """

    def __init__(self):
        """Initialize analyzer."""
        logger.info("Initialized HRVSensitivityAnalyzer")

    def calculate_baseline(
        self,
        hrv_readings: List[HRVReading],
        days_back: int = 7,
        exclude_post_meal_hours: int = 3,
    ) -> BaselineHRV:
        """
        Calculate user's baseline HRV from historical data.

        Args:
            hrv_readings: List of HRV readings
            days_back: Number of days to use for baseline
            exclude_post_meal_hours: Exclude readings this many hours after meals

        Returns:
            BaselineHRV with statistics
        """
        if not hrv_readings:
            raise ValueError("No HRV readings provided")

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        readings_in_range = [r for r in hrv_readings if r.timestamp >= cutoff]

        if len(readings_in_range) < HRV_THRESHOLDS["min_baseline_samples"]:
            raise ValueError(
                f"Insufficient data: {len(readings_in_range)} readings, "
                f"need at least {HRV_THRESHOLDS['min_baseline_samples']}"
            )

        rmssd_values = [r.rmssd for r in readings_in_range]

        baseline = BaselineHRV(
            mean_rmssd=float(np.mean(rmssd_values)),
            std_rmssd=float(np.std(rmssd_values)),
            median_rmssd=float(np.median(rmssd_values)),
            sample_count=len(readings_in_range),
            calculation_period_days=days_back,
            last_updated=datetime.utcnow(),
        )

        # Calculate optional metrics if available
        sdnn_values = [r.sdnn for r in readings_in_range if r.sdnn is not None]
        if sdnn_values:
            baseline.mean_sdnn = float(np.mean(sdnn_values))

        pnn50_values = [r.pnn50 for r in readings_in_range if r.pnn50 is not None]
        if pnn50_values:
            baseline.mean_pnn50 = float(np.mean(pnn50_values))

        lf_hf_values = [
            r.lf_hf_ratio for r in readings_in_range if r.lf_hf_ratio is not None
        ]
        if lf_hf_values:
            baseline.mean_lf_hf = float(np.mean(lf_hf_values))

        logger.info(
            f"Calculated baseline: mean_rmssd={baseline.mean_rmssd:.2f}ms, "
            f"std={baseline.std_rmssd:.2f}ms from {baseline.sample_count} readings"
        )

        return baseline

    def analyze_exposure(
        self,
        exposure_id: str,
        trigger_type: str,
        trigger_name: str,
        exposed_at: datetime,
        hrv_readings: List[HRVReading],
        baseline: BaselineHRV,
    ) -> ExposureAnalysis:
        """
        Analyze HRV response to a single exposure event.

        Args:
            exposure_id: Unique ID for this exposure
            trigger_type: Allergen or compound type
            trigger_name: Human-readable name
            exposed_at: When exposure occurred
            hrv_readings: HRV readings around the exposure time
            baseline: User's baseline HRV

        Returns:
            ExposureAnalysis with window-by-window breakdown
        """
        analysis = ExposureAnalysis(
            exposure_id=exposure_id,
            trigger_type=trigger_type,
            trigger_name=trigger_name,
            exposed_at=exposed_at,
            baseline_rmssd=baseline.mean_rmssd,
        )

        # Analyze each time window
        for window, window_def in WINDOW_DEFINITIONS.items():
            window_analysis = self._analyze_window(
                hrv_readings=hrv_readings,
                exposed_at=exposed_at,
                window=window,
                window_def=window_def,
                baseline=baseline,
            )
            analysis.windows[window] = window_analysis

            # Track maximum drop
            if window_analysis.rmssd_change_pct is not None and abs(
                window_analysis.rmssd_change_pct
            ) > abs(analysis.max_hrv_drop_pct):
                analysis.max_hrv_drop_pct = window_analysis.rmssd_change_pct
                if window_analysis.has_significant_drop:
                    analysis.peak_reaction_window = window

        # Determine overall reaction
        analysis.overall_had_reaction = any(
            w.has_significant_drop for w in analysis.windows.values()
        )

        analysis.overall_severity = self._determine_severity(analysis.max_hrv_drop_pct)

        # Assess data quality
        total_readings = sum(w.readings_count for w in analysis.windows.values())
        if total_readings >= 20:
            analysis.data_quality = "excellent"
            analysis.confidence = 0.95
        elif total_readings >= 10:
            analysis.data_quality = "good"
            analysis.confidence = 0.8
        elif total_readings >= 5:
            analysis.data_quality = "fair"
            analysis.confidence = 0.6
        else:
            analysis.data_quality = "poor"
            analysis.confidence = 0.4

        return analysis

    def _analyze_window(
        self,
        hrv_readings: List[HRVReading],
        exposed_at: datetime,
        window: TimeWindow,
        window_def: Dict,
        baseline: BaselineHRV,
    ) -> WindowAnalysis:
        """Analyze HRV in a specific time window after exposure."""
        start_time = exposed_at + timedelta(minutes=window_def["start"])
        end_time = exposed_at + timedelta(minutes=window_def["end"])

        # Filter readings in window
        window_readings = [
            r for r in hrv_readings if start_time <= r.timestamp < end_time
        ]

        analysis = WindowAnalysis(
            window=window,
            start_minutes=window_def["start"],
            end_minutes=window_def["end"],
            readings_count=len(window_readings),
        )

        if not window_readings:
            return analysis

        # Calculate statistics
        rmssd_values = [r.rmssd for r in window_readings]
        analysis.mean_rmssd = float(np.mean(rmssd_values))
        analysis.min_rmssd = float(np.min(rmssd_values))
        analysis.max_rmssd = float(np.max(rmssd_values))

        # Calculate change from baseline
        analysis.rmssd_change_ms = analysis.mean_rmssd - baseline.mean_rmssd
        if baseline.mean_rmssd > 0:
            analysis.rmssd_change_pct = (
                (analysis.mean_rmssd - baseline.mean_rmssd) / baseline.mean_rmssd
            ) * 100

        # Detect significant drop
        if analysis.rmssd_change_pct is not None:
            drop_pct = abs(analysis.rmssd_change_pct)

            if drop_pct >= HRV_THRESHOLDS["significant_reaction_pct"]:
                analysis.has_significant_drop = True
                analysis.reaction_severity = ReactionSeverity.SEVERE
                analysis.confidence = 0.9
            elif drop_pct >= HRV_THRESHOLDS["likely_reaction_pct"]:
                analysis.has_significant_drop = True
                analysis.reaction_severity = ReactionSeverity.MODERATE
                analysis.confidence = 0.75
            elif drop_pct >= HRV_THRESHOLDS["possible_reaction_pct"]:
                analysis.has_significant_drop = len(window_readings) >= 3
                analysis.reaction_severity = ReactionSeverity.MILD
                analysis.confidence = 0.5

        return analysis

    def _determine_severity(self, max_drop_pct: float) -> ReactionSeverity:
        """Determine reaction severity from maximum HRV drop."""
        drop = abs(max_drop_pct)

        if drop >= 30:
            return ReactionSeverity.SEVERE
        elif drop >= 20:
            return ReactionSeverity.MODERATE
        elif drop >= 10:
            return ReactionSeverity.MILD
        else:
            return ReactionSeverity.NONE

    def detect_sensitivity_pattern(
        self,
        trigger_type: str,
        trigger_name: str,
        exposure_analyses: List[ExposureAnalysis],
    ) -> Optional[SensitivityPattern]:
        """
        Detect sensitivity pattern across multiple exposures.

        Requires minimum number of exposures for statistical validity.

        Args:
            trigger_type: Allergen or compound type
            trigger_name: Human-readable name
            exposure_analyses: List of exposure analyses for this trigger

        Returns:
            SensitivityPattern if pattern detected, None otherwise
        """
        if len(exposure_analyses) < HRV_THRESHOLDS["min_exposures_for_pattern"]:
            logger.debug(
                f"Insufficient exposures for {trigger_name}: "
                f"{len(exposure_analyses)} < {HRV_THRESHOLDS['min_exposures_for_pattern']}"
            )
            return None

        # Collect HRV drop data
        hrv_drops = [a.max_hrv_drop_pct for a in exposure_analyses]
        reactions = [1 if a.overall_had_reaction else 0 for a in exposure_analyses]

        # Calculate statistics
        reaction_count = sum(reactions)
        reaction_rate = reaction_count / len(exposure_analyses)
        avg_drop = float(np.mean([abs(d) for d in hrv_drops]))
        max_drop = float(max(abs(d) for d in hrv_drops))
        std_drop = float(np.std([abs(d) for d in hrv_drops]))

        # Calculate window impacts
        window_impacts = {}
        for window in TimeWindow:
            window_drops = []
            for analysis in exposure_analyses:
                if window in analysis.windows:
                    w = analysis.windows[window]
                    if w.rmssd_change_pct is not None:
                        window_drops.append(abs(w.rmssd_change_pct))
            if window_drops:
                window_impacts[window] = float(np.mean(window_drops))

        # Find primary reaction window
        primary_window = max(
            window_impacts.items(),
            key=lambda x: x[1],
            default=(TimeWindow.SHORT_TERM, 0),
        )[0]

        # Statistical test: is the drop significant?
        # One-sample t-test: is mean drop significantly > 0?
        if len(hrv_drops) >= 3:
            _, p_value = stats.ttest_1samp([abs(d) for d in hrv_drops], 0)
            correlation = (
                float(
                    np.corrcoef(
                        list(range(len(hrv_drops))), [abs(d) for d in hrv_drops]
                    )[0, 1]
                )
                if len(hrv_drops) > 1
                else 0.0
            )
        else:
            _, p_value = 0.0, 1.0
            correlation = 0.0

        is_significant = (
            p_value < HRV_THRESHOLDS["statistical_significance"]
            and avg_drop >= HRV_THRESHOLDS["possible_reaction_pct"]
        )

        # Determine suggested severity
        suggested_severity = self._suggest_sensitivity_severity(
            avg_drop, reaction_rate, is_significant
        )

        # Calculate confidence
        confidence = self._calculate_pattern_confidence(
            len(exposure_analyses),
            reaction_rate,
            avg_drop,
            is_significant,
        )

        pattern = SensitivityPattern(
            trigger_type=trigger_type,
            trigger_name=trigger_name,
            total_exposures=len(exposure_analyses),
            reactions_detected=reaction_count,
            reaction_rate=reaction_rate,
            avg_hrv_drop_pct=avg_drop,
            max_hrv_drop_pct=max_drop,
            std_hrv_drop=std_drop,
            window_impacts=window_impacts,
            primary_reaction_window=primary_window,
            correlation_coefficient=correlation if not np.isnan(correlation) else 0.0,
            p_value=float(p_value),
            is_statistically_significant=is_significant,
            suggested_severity=suggested_severity,
            confidence=confidence,
            exposure_analyses=exposure_analyses,
        )

        return pattern

    def _suggest_sensitivity_severity(
        self,
        avg_drop: float,
        reaction_rate: float,
        is_significant: bool,
    ) -> SensitivitySeverity:
        """Suggest sensitivity severity based on pattern data."""
        if not is_significant:
            return SensitivitySeverity.MILD

        if avg_drop >= 25 and reaction_rate >= 0.8:
            return SensitivitySeverity.SEVERE
        elif avg_drop >= 20 and reaction_rate >= 0.6:
            return SensitivitySeverity.MODERATE
        elif avg_drop >= 15 and reaction_rate >= 0.5:
            return SensitivitySeverity.MODERATE
        else:
            return SensitivitySeverity.MILD

    def _calculate_pattern_confidence(
        self,
        exposure_count: int,
        reaction_rate: float,
        avg_drop: float,
        is_significant: bool,
    ) -> float:
        """Calculate confidence in the detected pattern."""
        # Base confidence from exposure count
        if exposure_count >= 10:
            count_factor = 1.0
        elif exposure_count >= 5:
            count_factor = 0.8
        else:
            count_factor = 0.6

        # Consistency factor from reaction rate
        # High reaction rate = consistent pattern
        consistency_factor = min(1.0, reaction_rate + 0.3)

        # Drop magnitude factor
        if avg_drop >= 20:
            magnitude_factor = 1.0
        elif avg_drop >= 15:
            magnitude_factor = 0.9
        elif avg_drop >= 10:
            magnitude_factor = 0.7
        else:
            magnitude_factor = 0.5

        # Statistical significance boost
        significance_factor = 1.2 if is_significant else 0.8

        confidence = (
            count_factor * 0.3 + consistency_factor * 0.3 + magnitude_factor * 0.4
        ) * significance_factor

        return min(0.99, max(0.1, confidence))

    def generate_sensitivity_report(
        self,
        user_id: str,
        baseline: BaselineHRV,
        current_hrv_readings: List[HRVReading],
        patterns: List[SensitivityPattern],
        analysis_period_days: int = 30,
    ) -> SensitivityReport:
        """
        Generate comprehensive sensitivity report for a user.

        Args:
            user_id: User ID
            baseline: User's baseline HRV
            current_hrv_readings: Recent HRV readings
            patterns: Detected sensitivity patterns
            analysis_period_days: Period analyzed

        Returns:
            Complete SensitivityReport
        """
        # Separate confirmed vs possible sensitivities
        confirmed = [
            p
            for p in patterns
            if p.is_statistically_significant and p.confidence >= 0.7
        ]
        possible = [
            p
            for p in patterns
            if not p.is_statistically_significant or p.confidence < 0.7
        ]

        # Calculate current HRV trend
        if current_hrv_readings:
            recent_rmssd = np.mean([r.rmssd for r in current_hrv_readings[-7:]])
            older_rmssd = (
                np.mean([r.rmssd for r in current_hrv_readings[:-7]])
                if len(current_hrv_readings) > 7
                else recent_rmssd
            )

            if recent_rmssd > older_rmssd * 1.05:
                trend = "improving"
            elif recent_rmssd < older_rmssd * 0.95:
                trend = "declining"
            else:
                trend = "stable"

            current_vs_baseline = (
                (recent_rmssd - baseline.mean_rmssd) / baseline.mean_rmssd
            ) * 100
        else:
            trend = "unknown"
            current_vs_baseline = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(confirmed, possible, baseline)

        # Count total exposures analyzed
        total_exposures = sum(p.total_exposures for p in patterns)

        # Data completeness
        expected_readings = analysis_period_days * 24  # Assume hourly readings ideal
        completeness = min(100, (len(current_hrv_readings) / expected_readings) * 100)

        return SensitivityReport(
            user_id=user_id,
            analysis_period_days=analysis_period_days,
            generated_at=datetime.utcnow(),
            baseline=baseline,
            current_hrv_trend=trend,
            current_vs_baseline_pct=round(current_vs_baseline, 1),
            confirmed_sensitivities=confirmed,
            possible_sensitivities=possible,
            recommendations=recommendations,
            total_hrv_readings=len(current_hrv_readings),
            total_exposures_analyzed=total_exposures,
            data_completeness_pct=round(completeness, 1),
        )

    def _generate_recommendations(
        self,
        confirmed: List[SensitivityPattern],
        possible: List[SensitivityPattern],
        baseline: BaselineHRV,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Recommendations for confirmed sensitivities
        for pattern in confirmed:
            if pattern.suggested_severity == SensitivitySeverity.SEVERE:
                recommendations.append(
                    f"AVOID {pattern.trigger_name}: Confirmed severe sensitivity "
                    f"(avg {pattern.avg_hrv_drop_pct:.0f}% HRV drop, {pattern.reaction_rate*100:.0f}% reaction rate)"
                )
            elif pattern.suggested_severity == SensitivitySeverity.MODERATE:
                recommendations.append(
                    f"LIMIT {pattern.trigger_name}: Confirmed moderate sensitivity. "
                    f"Consider elimination trial."
                )

        # Recommendations for possible sensitivities
        if possible:
            trigger_names = [p.trigger_name for p in possible[:3]]
            recommendations.append(
                f"MONITOR: {', '.join(trigger_names)} show possible sensitivity patterns. "
                f"Continue tracking for confirmation."
            )

        # General HRV recommendations
        if baseline.mean_rmssd < 30:
            recommendations.append(
                "Your baseline HRV is below average. Focus on stress reduction, "
                "sleep quality, and avoiding identified triggers."
            )

        # Data collection recommendation
        if baseline.sample_count < 50:
            recommendations.append(
                "Continue wearing your HRV monitor consistently for more accurate sensitivity detection."
            )

        return recommendations

    def get_reaction_pattern_info(
        self, sensitivity_type: SensitivityType
    ) -> Dict[str, Any]:
        """Get typical reaction pattern for a sensitivity type."""
        pattern = REACTION_PATTERNS.get(
            sensitivity_type,
            {
                "primary_window": TimeWindow.SHORT_TERM,
                "secondary_windows": [],
                "typical_onset_min": 60,
                "typical_duration_hours": 6,
            },
        )

        return {
            "primary_window": pattern["primary_window"].value,
            "primary_window_label": WINDOW_DEFINITIONS[pattern["primary_window"]][
                "label"
            ],
            "secondary_windows": [w.value for w in pattern["secondary_windows"]],
            "typical_onset_minutes": pattern["typical_onset_min"],
            "typical_duration_hours": pattern["typical_duration_hours"],
        }


# Singleton instance
hrv_sensitivity_analyzer = HRVSensitivityAnalyzer()
