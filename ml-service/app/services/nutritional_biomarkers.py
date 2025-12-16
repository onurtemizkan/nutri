"""
Advanced Nutritional Biomarker Prediction Engine

Based on comprehensive scientific literature review covering:
- Amino acid metabolism and HRV correlations (tryptophan, tyrosine, glycine, BCAAs)
- Dietary Inflammatory Index (DII) and cardiovascular outcomes
- Omega-3 fatty acids (EPA/DHA) and autonomic function
- Magnesium and autonomic nervous system regulation
- Vitamin D and cardiac autonomic function
- Glycemic variability and HRV correlations
- Histamine/tyramine biogenic amines and autonomic effects
- Gut-brain axis and vagal tone modulation
- Polyphenols/flavonoids and cardiovascular protection

Key Research Sources:
- Nutrition & Metabolism 2024: Tryptophan-CVD meta-analysis (34,370 subjects)
- Frontiers in Molecular Biosciences 2025: Blood metabolome-HRV correlations
- ScienceDirect 2024: CGRP-histamine-migraine gut-brain connections
- Journal of Sleep Research 2024: Caffeine concentration-HRV relationships
- Gut Microbes 2025: Probiotic-HRV clinical trial results
- Frontiers in Nutrition 2024: DII-cardiometabolic parameters
"""
# mypy: ignore-errors

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import math


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class AminoAcidType(str, Enum):
    """Essential and conditionally essential amino acids"""

    # Essential
    TRYPTOPHAN = "tryptophan"
    TYROSINE = "tyrosine"
    PHENYLALANINE = "phenylalanine"
    LEUCINE = "leucine"
    ISOLEUCINE = "isoleucine"
    VALINE = "valine"
    METHIONINE = "methionine"
    THREONINE = "threonine"
    LYSINE = "lysine"
    HISTIDINE = "histidine"

    # Conditionally essential
    GLYCINE = "glycine"
    ARGININE = "arginine"
    GLUTAMINE = "glutamine"
    TAURINE = "taurine"
    CYSTEINE = "cysteine"


class NeurotransmitterType(str, Enum):
    """Neurotransmitters synthesized from amino acid precursors"""

    SEROTONIN = "serotonin"  # From tryptophan
    MELATONIN = "melatonin"  # From serotonin (via tryptophan)
    DOPAMINE = "dopamine"  # From tyrosine
    NOREPINEPHRINE = "norepinephrine"  # From dopamine
    EPINEPHRINE = "epinephrine"  # From norepinephrine
    GABA = "gaba"  # From glutamate
    ACETYLCHOLINE = "acetylcholine"  # From choline
    HISTAMINE = "histamine"  # From histidine


class MicronutrientType(str, Enum):
    """Key micronutrients affecting autonomic function"""

    # Minerals
    MAGNESIUM = "magnesium"
    ZINC = "zinc"
    IRON = "iron"
    SELENIUM = "selenium"
    POTASSIUM = "potassium"
    CALCIUM = "calcium"

    # Vitamins
    VITAMIN_D = "vitamin_d"
    VITAMIN_B6 = "vitamin_b6"
    VITAMIN_B12 = "vitamin_b12"
    FOLATE = "folate"
    VITAMIN_C = "vitamin_c"
    VITAMIN_E = "vitamin_e"

    # Fatty Acids
    EPA = "epa"
    DHA = "dha"
    ALA = "ala"


class InflammatoryCategory(str, Enum):
    """Food inflammatory potential categories"""

    HIGHLY_ANTI_INFLAMMATORY = "highly_anti_inflammatory"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    NEUTRAL = "neutral"
    PRO_INFLAMMATORY = "pro_inflammatory"
    HIGHLY_PRO_INFLAMMATORY = "highly_pro_inflammatory"


class BiogenicAmineType(str, Enum):
    """Biogenic amines in foods"""

    HISTAMINE = "histamine"
    TYRAMINE = "tyramine"
    PHENYLETHYLAMINE = "phenylethylamine"
    PUTRESCINE = "putrescine"
    CADAVERINE = "cadaverine"
    SPERMINE = "spermine"
    SPERMIDINE = "spermidine"


class GutBrainPathway(str, Enum):
    """Gut-brain axis communication pathways"""

    VAGAL_AFFERENT = "vagal_afferent"
    IMMUNE_MEDIATED = "immune_mediated"
    METABOLITE_SIGNALING = "metabolite_signaling"  # SCFAs, etc.
    NEUROTRANSMITTER = "neurotransmitter"
    HPA_AXIS = "hpa_axis"


# =============================================================================
# RESEARCH-BASED CONSTANTS
# =============================================================================

# Amino acid content per 100g for common foods (mg)
AMINO_ACID_DATABASE: Dict[str, Dict[str, float]] = {
    "chicken_breast": {
        "tryptophan": 267,
        "tyrosine": 893,
        "phenylalanine": 1030,
        "leucine": 2013,
        "isoleucine": 1362,
        "valine": 1325,
        "glycine": 1221,
        "arginine": 1545,
        "glutamine": 3610,
    },
    "salmon": {
        "tryptophan": 250,
        "tyrosine": 759,
        "phenylalanine": 869,
        "leucine": 1770,
        "isoleucine": 1006,
        "valine": 1120,
        "glycine": 1040,
        "arginine": 1300,
        "glutamine": 2980,
    },
    "eggs": {
        "tryptophan": 167,
        "tyrosine": 499,
        "phenylalanine": 680,
        "leucine": 1088,
        "isoleucine": 672,
        "valine": 859,
        "glycine": 432,
        "arginine": 755,
        "glutamine": 1670,
    },
    "milk": {
        "tryptophan": 46,
        "tyrosine": 159,
        "phenylalanine": 163,
        "leucine": 323,
        "isoleucine": 200,
        "valine": 220,
        "glycine": 72,
        "arginine": 119,
        "glutamine": 680,
    },
    "beef": {
        "tryptophan": 223,
        "tyrosine": 751,
        "phenylalanine": 872,
        "leucine": 1760,
        "isoleucine": 1001,
        "valine": 1085,
        "glycine": 1260,
        "arginine": 1400,
        "glutamine": 3250,
    },
    "tofu": {
        "tryptophan": 148,
        "tyrosine": 400,
        "phenylalanine": 520,
        "leucine": 820,
        "isoleucine": 510,
        "valine": 530,
        "glycine": 450,
        "arginine": 830,
        "glutamine": 2100,
    },
    "lentils": {
        "tryptophan": 81,
        "tyrosine": 265,
        "phenylalanine": 481,
        "leucine": 654,
        "isoleucine": 390,
        "valine": 448,
        "glycine": 380,
        "arginine": 780,
        "glutamine": 1520,
    },
    "oats": {
        "tryptophan": 182,
        "tyrosine": 447,
        "phenylalanine": 663,
        "leucine": 980,
        "isoleucine": 503,
        "valine": 688,
        "glycine": 642,
        "arginine": 850,
        "glutamine": 2890,
    },
    "almonds": {
        "tryptophan": 211,
        "tyrosine": 452,
        "phenylalanine": 1132,
        "leucine": 1461,
        "isoleucine": 745,
        "valine": 848,
        "glycine": 1430,
        "arginine": 2490,
        "glutamine": 4140,
    },
    "banana": {
        "tryptophan": 9,
        "tyrosine": 9,
        "phenylalanine": 49,
        "leucine": 68,
        "isoleucine": 28,
        "valine": 47,
        "glycine": 38,
        "arginine": 49,
        "glutamine": 152,
    },
    "spinach": {
        "tryptophan": 39,
        "tyrosine": 108,
        "phenylalanine": 129,
        "leucine": 223,
        "isoleucine": 147,
        "valine": 161,
        "glycine": 134,
        "arginine": 162,
        "glutamine": 343,
    },
}

# Dietary Inflammatory Index (DII) coefficients
# Based on Shivappa et al. meta-analysis of 1943 articles
DII_COEFFICIENTS: Dict[str, float] = {
    # Anti-inflammatory (negative values)
    "fiber": -0.663,
    "omega_3": -0.436,
    "omega_6": -0.159,  # Context-dependent
    "beta_carotene": -0.584,
    "vitamin_a": -0.401,
    "vitamin_c": -0.424,
    "vitamin_d": -0.446,
    "vitamin_e": -0.419,
    "vitamin_b6": -0.365,
    "vitamin_b12": -0.106,
    "folate": -0.190,
    "magnesium": -0.484,
    "zinc": -0.313,
    "selenium": -0.191,
    "flavonoids": -0.467,
    "anthocyanidins": -0.131,
    "flavan_3_ols": -0.159,
    "flavones": -0.160,
    "flavonols": -0.467,
    "isoflavones": -0.593,
    "green_tea": -0.536,
    "garlic": -0.412,
    "ginger": -0.453,
    "turmeric": -0.785,
    "onion": -0.301,
    "pepper": -0.131,
    "thyme_oregano": -0.102,
    "rosemary": -0.013,
    "saffron": -0.140,
    "alcohol_moderate": -0.278,  # J-curve effect
    "caffeine": -0.110,
    # Pro-inflammatory (positive values)
    "saturated_fat": 0.373,
    "trans_fat": 0.229,
    "cholesterol": 0.110,
    "total_fat_excess": 0.298,
    "carbs_refined": 0.097,
    "protein_excess": 0.021,
    "iron_excess": 0.032,
    "energy_excess": 0.180,
}

# Omega-3 effects on HRV (based on research)
# 3.4g/day EPA+DHA showed significant HRV improvement
OMEGA3_HRV_EFFECTS: Dict[str, Dict[str, float]] = {
    "low_dose": {  # < 1g/day
        "hf_power_change": 0.0,
        "lf_hf_ratio_change": 0.0,
        "rmssd_change": 0.0,
    },
    "moderate_dose": {  # 1-2g/day
        "hf_power_change": 0.05,
        "lf_hf_ratio_change": -0.03,
        "rmssd_change": 2.0,
    },
    "therapeutic_dose": {  # 2-4g/day
        "hf_power_change": 0.15,
        "lf_hf_ratio_change": -0.08,
        "rmssd_change": 5.0,
    },
    "high_dose": {  # > 4g/day
        "hf_power_change": 0.12,  # Diminishing returns
        "lf_hf_ratio_change": -0.06,
        "rmssd_change": 4.0,
    },
}

# Magnesium effects on HRV (400mg/day study)
MAGNESIUM_HRV_EFFECTS = {
    "pnn50_increase": 0.15,  # 15% increase
    "lf_hf_ratio_decrease": 0.10,
    "stress_index_decrease": 0.12,
    "vagal_activity_increase": 0.18,
}

# Vitamin D - HRV relationships
VITAMIN_D_HRV_THRESHOLDS = {
    "deficient": {"level": 20, "hrv_impact": -0.15},  # < 20 ng/mL
    "insufficient": {"level": 30, "hrv_impact": -0.05},  # 20-30 ng/mL
    "optimal": {"level": 50, "hrv_impact": 0.0},  # 30-50 ng/mL
    "high_normal": {"level": 80, "hrv_impact": 0.02},  # 50-80 ng/mL
}

# Caffeine concentration-effect relationships (μmol/L)
CAFFEINE_HRV_THRESHOLDS = {
    "hr_reduction_threshold": 4.3,  # μmol/L for HR reduction
    "hf_hrv_increase_threshold": 4.9,  # μmol/L for parasympathetic increase
    "eeg_delta_reduction": 7.4,  # μmol/L for sleep disruption
}

# Biogenic amine thresholds (mg/day)
BIOGENIC_AMINE_THRESHOLDS = {
    "histamine": {
        "safe": 10,
        "moderate_risk": 50,
        "high_risk": 100,
        "dao_capacity_normal": 50,  # mg/day DAO can process
    },
    "tyramine": {
        "safe": 100,
        "maoi_risk": 25,  # With MAO inhibitors
        "migraine_trigger": 150,
    },
}

# Gut microbiome - HRV relationships
GUT_BRAIN_EFFECTS = {
    "akkermansia_muciniphila": {
        "morning_vn_improvement": 0.12,
        "sleep_improvement": 0.08,
    },
    "lactobacillales": {
        "vagal_correlation": 0.25,
        "scfa_production": "high",
    },
    "ruminococcaceae": {
        "vagal_correlation": 0.22,
        "scfa_production": "high",
    },
    "christensenellaceae": {
        "metabolic_health": 0.15,
        "anti_inflammatory": 0.10,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class AminoAcidProfile:
    """Complete amino acid intake profile"""

    tryptophan_mg: float = 0.0
    tyrosine_mg: float = 0.0
    phenylalanine_mg: float = 0.0
    leucine_mg: float = 0.0
    isoleucine_mg: float = 0.0
    valine_mg: float = 0.0
    glycine_mg: float = 0.0
    arginine_mg: float = 0.0
    glutamine_mg: float = 0.0
    taurine_mg: float = 0.0
    histidine_mg: float = 0.0
    methionine_mg: float = 0.0

    @property
    def bcaa_total(self) -> float:
        """Total branched-chain amino acids"""
        return self.leucine_mg + self.isoleucine_mg + self.valine_mg

    @property
    def bcaa_ratio(self) -> str:
        """BCAA ratio (leucine:isoleucine:valine)"""
        if self.isoleucine_mg > 0:
            leu = self.leucine_mg / self.isoleucine_mg
            val = self.valine_mg / self.isoleucine_mg
            return f"{leu:.1f}:1:{val:.1f}"
        return "N/A"

    @property
    def serotonin_precursor_score(self) -> float:
        """Score for serotonin synthesis potential (0-100)"""
        # Tryptophan RDA ~280mg, optimal ~500mg for mood
        score = min(100, (self.tryptophan_mg / 500) * 100)
        # Competing amino acids reduce tryptophan brain uptake
        if self.bcaa_total > 5000:  # High BCAA competes
            score *= 0.8
        return score

    @property
    def dopamine_precursor_score(self) -> float:
        """Score for dopamine synthesis potential (0-100)"""
        # Tyrosine + phenylalanine pathway
        precursor = self.tyrosine_mg + (self.phenylalanine_mg * 0.5)
        return min(100, (precursor / 1000) * 100)


@dataclass
class MicronutrientProfile:
    """Micronutrient intake and status"""

    # Minerals (mg unless specified)
    magnesium_mg: float = 0.0
    zinc_mg: float = 0.0
    iron_mg: float = 0.0
    selenium_mcg: float = 0.0
    potassium_mg: float = 0.0
    calcium_mg: float = 0.0

    # Vitamins
    vitamin_d_iu: float = 0.0
    vitamin_b6_mg: float = 0.0
    vitamin_b12_mcg: float = 0.0
    folate_mcg: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_e_mg: float = 0.0

    # Fatty acids (g)
    epa_g: float = 0.0
    dha_g: float = 0.0
    ala_g: float = 0.0

    @property
    def omega3_total_g(self) -> float:
        return self.epa_g + self.dha_g + self.ala_g

    @property
    def omega3_dose_category(self) -> str:
        total = self.epa_g + self.dha_g
        if total < 1.0:
            return "low_dose"
        elif total < 2.0:
            return "moderate_dose"
        elif total < 4.0:
            return "therapeutic_dose"
        return "high_dose"

    @property
    def vitamin_d_ng_ml_estimated(self) -> float:
        """Rough estimate of serum 25(OH)D from intake"""
        # ~40 IU raises serum by 1 ng/mL (simplified)
        return min(100, 20 + (self.vitamin_d_iu / 40))


@dataclass
class InflammatoryProfile:
    """Dietary inflammatory assessment"""

    dii_score: float = 0.0
    anti_inflammatory_foods: int = 0
    pro_inflammatory_foods: int = 0
    polyphenol_score: float = 0.0
    fiber_g: float = 0.0
    omega6_omega3_ratio: float = 10.0

    @property
    def category(self) -> InflammatoryCategory:
        if self.dii_score < -4:
            return InflammatoryCategory.HIGHLY_ANTI_INFLAMMATORY
        elif self.dii_score < -1:
            return InflammatoryCategory.ANTI_INFLAMMATORY
        elif self.dii_score < 1:
            return InflammatoryCategory.NEUTRAL
        elif self.dii_score < 4:
            return InflammatoryCategory.PRO_INFLAMMATORY
        return InflammatoryCategory.HIGHLY_PRO_INFLAMMATORY

    @property
    def cvd_risk_multiplier(self) -> float:
        """Based on meta-analysis: highest DII = 1.41x CVD risk"""
        # Linear interpolation from research
        if self.dii_score <= -4:
            return 0.85
        elif self.dii_score >= 4:
            return 1.41
        return 1.0 + (self.dii_score * 0.05)


@dataclass
class BiogenicAmineProfile:
    """Biogenic amine intake tracking"""

    histamine_mg: float = 0.0
    tyramine_mg: float = 0.0
    phenylethylamine_mg: float = 0.0

    # DAO enzyme factors
    dao_inhibitor_consumed: bool = False
    alcohol_consumed: bool = False

    @property
    def histamine_risk_level(self) -> str:
        threshold = BIOGENIC_AMINE_THRESHOLDS["histamine"]
        effective_capacity = threshold["dao_capacity_normal"]
        if self.dao_inhibitor_consumed:
            effective_capacity *= 0.5
        if self.alcohol_consumed:
            effective_capacity *= 0.7

        if self.histamine_mg < threshold["safe"]:
            return "safe"
        elif self.histamine_mg < effective_capacity:
            return "moderate"
        elif self.histamine_mg < threshold["high_risk"]:
            return "elevated"
        return "high_risk"

    @property
    def migraine_trigger_risk(self) -> float:
        """Probability of migraine trigger (0-1)"""
        risk = 0.0

        # Histamine contribution
        if self.histamine_mg > 50:
            risk += min(0.4, (self.histamine_mg - 50) / 100)

        # Tyramine contribution
        if self.tyramine_mg > 100:
            risk += min(0.3, (self.tyramine_mg - 100) / 150)

        # PEA contribution
        if self.phenylethylamine_mg > 30:
            risk += 0.1

        # Synergistic effects
        if self.histamine_mg > 30 and self.tyramine_mg > 80:
            risk *= 1.3

        return min(1.0, risk)


@dataclass
class GlycemicProfile:
    """Glycemic load and variability assessment"""

    glycemic_load: float = 0.0
    glycemic_index_avg: float = 0.0
    fiber_g: float = 0.0
    added_sugar_g: float = 0.0
    complex_carbs_g: float = 0.0
    simple_carbs_g: float = 0.0
    protein_with_carbs: bool = True  # Protein blunts glucose spike
    fat_with_carbs: bool = True  # Fat slows absorption

    @property
    def predicted_glucose_variability(self) -> str:
        """Predict glycemic variability category"""
        # High GL + low fiber + no protein/fat = high variability
        gl_factor = self.glycemic_load / 100
        fiber_factor = max(0.5, 1 - (self.fiber_g / 30))

        variability = gl_factor * fiber_factor
        if not self.protein_with_carbs:
            variability *= 1.2
        if not self.fat_with_carbs:
            variability *= 1.1

        if variability < 0.5:
            return "low"
        elif variability < 1.0:
            return "moderate"
        elif variability < 1.5:
            return "elevated"
        return "high"

    @property
    def hrv_impact_prediction(self) -> float:
        """Predicted HRV impact from glycemic load (-1 to 0)"""
        # Research: GV negatively correlated with HRV
        base_impact = -0.05 * (self.glycemic_load / 100)

        # Fiber mitigates
        fiber_mitigation = min(0.03, self.fiber_g / 1000)

        # Sugar exacerbates
        sugar_penalty = -0.02 * (self.added_sugar_g / 50)

        return max(-0.15, base_impact + fiber_mitigation + sugar_penalty)


@dataclass
class GutBrainProfile:
    """Gut-brain axis assessment"""

    probiotic_cfu: float = 0.0  # Colony forming units (billions)
    prebiotic_fiber_g: float = 0.0
    fermented_foods_servings: int = 0
    scfa_precursors_g: float = 0.0  # Short-chain fatty acid precursors

    # Beneficial bacteria indicators
    has_akkermansia_support: bool = False
    has_lactobacillus_support: bool = False
    has_bifidobacterium_support: bool = False

    @property
    def vagal_tone_support_score(self) -> float:
        """Score for vagal tone support (0-100)"""
        score = 0.0

        # Probiotics (multi-species shown to improve VN function after 3 months)
        if self.probiotic_cfu >= 10:  # 10 billion CFU
            score += 25
        elif self.probiotic_cfu >= 1:
            score += 15

        # Prebiotic fiber supports SCFA production
        score += min(25, self.prebiotic_fiber_g * 2.5)

        # Fermented foods
        score += min(20, self.fermented_foods_servings * 5)

        # Specific bacteria support
        if self.has_akkermansia_support:
            score += 10
        if self.has_lactobacillus_support:
            score += 10
        if self.has_bifidobacterium_support:
            score += 10

        return min(100, score)


@dataclass
class NeurotransmitterPrediction:
    """Predicted neurotransmitter synthesis capacity"""

    serotonin_score: float = 0.0
    dopamine_score: float = 0.0
    norepinephrine_score: float = 0.0
    gaba_score: float = 0.0
    acetylcholine_score: float = 0.0
    melatonin_score: float = 0.0

    # Limiting factors
    b6_adequate: bool = True  # Required cofactor
    iron_adequate: bool = True  # Required for synthesis
    magnesium_adequate: bool = True  # GABA modulation

    @property
    def mood_support_score(self) -> float:
        """Overall mood support from precursor availability"""
        base = (self.serotonin_score + self.dopamine_score + self.gaba_score) / 3

        # Cofactor penalties
        if not self.b6_adequate:
            base *= 0.7
        if not self.iron_adequate:
            base *= 0.85
        if not self.magnesium_adequate:
            base *= 0.9

        return base

    @property
    def sleep_support_score(self) -> float:
        """Sleep support from neurotransmitter precursors"""
        return (
            self.serotonin_score * 0.4
            + self.melatonin_score * 0.3
            + self.gaba_score * 0.3
        )

    @property
    def stress_resilience_score(self) -> float:
        """Stress resilience prediction"""
        return (
            self.gaba_score * 0.35
            + self.serotonin_score * 0.25
            + self.dopamine_score * 0.2
            + self.acetylcholine_score * 0.2
        )


@dataclass
class HRVPrediction:
    """Comprehensive HRV prediction from nutritional factors"""

    # Individual factor contributions
    amino_acid_contribution: float = 0.0
    omega3_contribution: float = 0.0
    magnesium_contribution: float = 0.0
    vitamin_d_contribution: float = 0.0
    inflammatory_contribution: float = 0.0
    glycemic_contribution: float = 0.0
    biogenic_amine_contribution: float = 0.0
    gut_brain_contribution: float = 0.0
    caffeine_contribution: float = 0.0

    # Predicted changes
    predicted_rmssd_change_ms: float = 0.0
    predicted_hf_power_change_pct: float = 0.0
    predicted_lf_hf_ratio_change: float = 0.0

    confidence: float = 0.0

    @property
    def total_hrv_impact(self) -> float:
        """Net HRV impact from all factors (-1 to 1)"""
        return (
            self.amino_acid_contribution
            + self.omega3_contribution
            + self.magnesium_contribution
            + self.vitamin_d_contribution
            + self.inflammatory_contribution
            + self.glycemic_contribution
            + self.biogenic_amine_contribution
            + self.gut_brain_contribution
            + self.caffeine_contribution
        )

    @property
    def autonomic_balance_prediction(self) -> str:
        """Predicted autonomic balance"""
        impact = self.total_hrv_impact
        if impact > 0.1:
            return "parasympathetic_enhanced"
        elif impact > 0.02:
            return "slightly_parasympathetic"
        elif impact > -0.02:
            return "balanced"
        elif impact > -0.1:
            return "slightly_sympathetic"
        return "sympathetic_dominant"


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================


class NutritionalBiomarkerEngine:
    """
    Advanced engine for predicting health biomarker responses to nutrition.

    Based on peer-reviewed research from:
    - Nutrition & Metabolism, Frontiers, JACC, Sleep journals
    - Meta-analyses covering 30,000+ subjects
    - Clinical trials on omega-3, magnesium, probiotics
    """

    def __init__(self):
        self.amino_acid_db = AMINO_ACID_DATABASE
        self.dii_coefficients = DII_COEFFICIENTS
        self.omega3_effects = OMEGA3_HRV_EFFECTS

    def calculate_amino_acid_profile(
        self, foods: List[Dict[str, Any]]
    ) -> AminoAcidProfile:
        """
        Calculate amino acid profile from food intake.

        Args:
            foods: List of {name, amount_g} dictionaries

        Returns:
            Complete amino acid profile
        """
        profile = AminoAcidProfile()

        for food in foods:
            name = food.get("name", "").lower().replace(" ", "_")
            amount_g = food.get("amount_g", 0)

            if name in self.amino_acid_db:
                aa_data = self.amino_acid_db[name]
                multiplier = amount_g / 100

                profile.tryptophan_mg += aa_data.get("tryptophan", 0) * multiplier
                profile.tyrosine_mg += aa_data.get("tyrosine", 0) * multiplier
                profile.phenylalanine_mg += aa_data.get("phenylalanine", 0) * multiplier
                profile.leucine_mg += aa_data.get("leucine", 0) * multiplier
                profile.isoleucine_mg += aa_data.get("isoleucine", 0) * multiplier
                profile.valine_mg += aa_data.get("valine", 0) * multiplier
                profile.glycine_mg += aa_data.get("glycine", 0) * multiplier
                profile.arginine_mg += aa_data.get("arginine", 0) * multiplier
                profile.glutamine_mg += aa_data.get("glutamine", 0) * multiplier

        return profile

    def calculate_dietary_inflammatory_index(
        self, nutrients: Dict[str, float]
    ) -> InflammatoryProfile:
        """
        Calculate Dietary Inflammatory Index (DII) score.

        Based on Shivappa et al. methodology from 1943 articles.
        Higher scores = more pro-inflammatory.
        Range typically -8 to +8.

        Args:
            nutrients: Dict of nutrient names to amounts

        Returns:
            Complete inflammatory profile
        """
        dii_score = 0.0
        anti_count = 0
        pro_count = 0

        for nutrient, amount in nutrients.items():
            if nutrient in self.dii_coefficients:
                coef = self.dii_coefficients[nutrient]
                # Standardize to z-score (simplified)
                contribution = coef * math.tanh(amount / 100)
                dii_score += contribution

                if coef < 0:
                    anti_count += 1
                else:
                    pro_count += 1

        # Calculate omega-6:omega-3 ratio
        omega6 = nutrients.get("omega_6", 15)  # Average Western diet
        omega3 = nutrients.get("omega_3", 1.5)
        ratio = omega6 / max(omega3, 0.1)

        return InflammatoryProfile(
            dii_score=dii_score,
            anti_inflammatory_foods=anti_count,
            pro_inflammatory_foods=pro_count,
            fiber_g=nutrients.get("fiber", 0),
            omega6_omega3_ratio=ratio,
            polyphenol_score=nutrients.get("flavonoids", 0)
            + nutrients.get("anthocyanidins", 0),
        )

    def calculate_neurotransmitter_prediction(
        self, amino_acids: AminoAcidProfile, micronutrients: MicronutrientProfile
    ) -> NeurotransmitterPrediction:
        """
        Predict neurotransmitter synthesis capacity from precursors.

        Based on research:
        - Tryptophan → 5-HTP → Serotonin → Melatonin
        - Tyrosine → L-DOPA → Dopamine → Norepinephrine → Epinephrine
        - Glutamate → GABA (B6 cofactor required)
        """
        # Cofactor adequacy
        b6_adequate = micronutrients.vitamin_b6_mg >= 1.3
        iron_adequate = micronutrients.iron_mg >= 8
        mg_adequate = micronutrients.magnesium_mg >= 300

        # Serotonin pathway
        # Tryptophan competes with BCAAs for transport
        trp_availability = amino_acids.tryptophan_mg
        bcaa_competition = 1 - min(0.3, amino_acids.bcaa_total / 20000)
        serotonin_score = min(100, (trp_availability / 400) * 100 * bcaa_competition)

        # B6 required for final conversion
        if not b6_adequate:
            serotonin_score *= 0.6

        # Melatonin (downstream from serotonin)
        melatonin_score = serotonin_score * 0.8  # Some loss in pathway

        # Dopamine pathway
        tyr_phe = amino_acids.tyrosine_mg + (amino_acids.phenylalanine_mg * 0.5)
        dopamine_score = min(100, (tyr_phe / 800) * 100)

        # Iron required for tyrosine hydroxylase
        if not iron_adequate:
            dopamine_score *= 0.7

        # Norepinephrine (from dopamine, needs vitamin C)
        norepinephrine_score = dopamine_score * 0.9
        if micronutrients.vitamin_c_mg < 60:
            norepinephrine_score *= 0.8

        # GABA (from glutamate via GAD enzyme, B6 cofactor)
        glutamate = amino_acids.glutamine_mg * 0.8  # Conversion
        gaba_score = min(100, (glutamate / 2000) * 100)
        if not b6_adequate:
            gaba_score *= 0.5  # B6 is critical for GAD
        if mg_adequate:
            gaba_score *= 1.15  # Mg enhances GABA receptor function
        gaba_score = min(100, gaba_score)

        # Acetylcholine (from choline)
        # Simplified - would need choline intake data
        acetylcholine_score = 70  # Default moderate

        return NeurotransmitterPrediction(
            serotonin_score=serotonin_score,
            dopamine_score=dopamine_score,
            norepinephrine_score=norepinephrine_score,
            gaba_score=gaba_score,
            acetylcholine_score=acetylcholine_score,
            melatonin_score=melatonin_score,
            b6_adequate=b6_adequate,
            iron_adequate=iron_adequate,
            magnesium_adequate=mg_adequate,
        )

    def predict_hrv_response(
        self,
        amino_acids: AminoAcidProfile,
        micronutrients: MicronutrientProfile,
        inflammatory: InflammatoryProfile,
        glycemic: GlycemicProfile,
        biogenic_amines: BiogenicAmineProfile,
        gut_brain: GutBrainProfile,
        caffeine_mg: float = 0.0,
    ) -> HRVPrediction:
        """
        Predict HRV response from comprehensive nutritional assessment.

        Research basis:
        - Omega-3: 3.4g/d EPA+DHA increases HF power (parasympathetic)
        - Magnesium: 400mg/d increases pNN50, decreases stress index
        - Vitamin D: Deficiency (<20 ng/mL) reduces RMSSD, HF
        - DII: High scores associated with reduced HRV
        - Glycine/Taurine: Antioxidant, improves HRV in heart failure
        - Probiotics: 3-month supplementation improves morning VN function
        """
        prediction = HRVPrediction()

        # 1. Amino acid contribution (glycine, taurine antioxidant effects)
        glycine_effect = min(0.03, amino_acids.glycine_mg / 10000)
        taurine_effect = min(0.02, amino_acids.taurine_mg / 3000)
        # Tryptophan → serotonin → parasympathetic
        trp_effect = min(0.02, amino_acids.serotonin_precursor_score / 5000)
        prediction.amino_acid_contribution = (
            glycine_effect + taurine_effect + trp_effect
        )

        # 2. Omega-3 contribution
        dose_cat = micronutrients.omega3_dose_category
        effects = self.omega3_effects.get(dose_cat, {})
        prediction.omega3_contribution = effects.get("hf_power_change", 0)
        prediction.predicted_rmssd_change_ms += effects.get("rmssd_change", 0)

        # 3. Magnesium contribution
        mg_factor = min(1.0, micronutrients.magnesium_mg / 400)
        prediction.magnesium_contribution = (
            mg_factor * MAGNESIUM_HRV_EFFECTS["vagal_activity_increase"]
        )
        prediction.predicted_hf_power_change_pct += (
            mg_factor * MAGNESIUM_HRV_EFFECTS["pnn50_increase"] * 100
        )

        # 4. Vitamin D contribution
        vit_d = micronutrients.vitamin_d_ng_ml_estimated
        if vit_d < 20:
            prediction.vitamin_d_contribution = -0.15
        elif vit_d < 30:
            prediction.vitamin_d_contribution = -0.05
        else:
            prediction.vitamin_d_contribution = 0.02

        # 5. Inflammatory contribution (DII)
        # High DII → inflammation → reduced HRV
        dii_effect = -0.02 * inflammatory.dii_score
        prediction.inflammatory_contribution = max(-0.15, min(0.1, dii_effect))

        # 6. Glycemic contribution
        prediction.glycemic_contribution = glycemic.hrv_impact_prediction

        # 7. Biogenic amine contribution
        # Histamine/tyramine can trigger sympathetic activation
        if biogenic_amines.histamine_risk_level == "high_risk":
            prediction.biogenic_amine_contribution = -0.08
        elif biogenic_amines.histamine_risk_level == "elevated":
            prediction.biogenic_amine_contribution = -0.04
        else:
            prediction.biogenic_amine_contribution = 0.0

        # 8. Gut-brain contribution
        # Probiotics improve vagal tone after 3 months
        vagal_support = gut_brain.vagal_tone_support_score / 100
        prediction.gut_brain_contribution = vagal_support * 0.1

        # 9. Caffeine contribution
        # Non-linear: moderate caffeine can increase HF-HRV
        caffeine_umol = caffeine_mg / 194  # Convert mg to μmol/L (rough)
        if caffeine_umol > CAFFEINE_HRV_THRESHOLDS["hf_hrv_increase_threshold"]:
            prediction.caffeine_contribution = 0.03
        elif caffeine_umol > 10:  # High caffeine
            prediction.caffeine_contribution = -0.02

        # Calculate predicted changes
        total_impact = prediction.total_hrv_impact
        prediction.predicted_rmssd_change_ms += total_impact * 10  # Scale factor
        prediction.predicted_hf_power_change_pct += total_impact * 15
        prediction.predicted_lf_hf_ratio_change = -total_impact * 0.5

        # Confidence based on data completeness
        data_points = sum(
            [
                amino_acids.tryptophan_mg > 0,
                micronutrients.omega3_total_g > 0,
                micronutrients.magnesium_mg > 0,
                micronutrients.vitamin_d_iu > 0,
                inflammatory.dii_score != 0,
                glycemic.glycemic_load > 0,
                gut_brain.probiotic_cfu > 0,
            ]
        )
        prediction.confidence = min(0.95, 0.5 + (data_points * 0.06))

        return prediction

    def get_personalized_recommendations(
        self,
        hrv_prediction: HRVPrediction,
        neurotransmitter_prediction: NeurotransmitterPrediction,
        inflammatory: InflammatoryProfile,
        current_hrv_baseline: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized nutritional recommendations.

        Args:
            hrv_prediction: Current HRV prediction
            neurotransmitter_prediction: Neurotransmitter status
            inflammatory: Current inflammatory profile
            current_hrv_baseline: User's baseline RMSSD if known

        Returns:
            List of prioritized recommendations with evidence
        """
        recommendations = []

        # Omega-3 recommendation
        if hrv_prediction.omega3_contribution < 0.1:
            recommendations.append(
                {
                    "priority": 1,
                    "category": "omega_3",
                    "recommendation": "Increase EPA+DHA intake to 2-4g/day",
                    "foods": ["fatty fish (salmon, mackerel)", "fish oil", "algae oil"],
                    "expected_benefit": "+5ms RMSSD, +15% HF power",
                    "evidence": "Meta-analysis: 3.4g/d EPA+DHA improves HRV in 8 weeks",
                    "source": "Frontiers in Physiology 2011",
                }
            )

        # Magnesium recommendation
        if hrv_prediction.magnesium_contribution < 0.15:
            recommendations.append(
                {
                    "priority": 2,
                    "category": "magnesium",
                    "recommendation": "Increase magnesium to 400mg/day",
                    "foods": ["dark chocolate", "almonds", "spinach", "avocado"],
                    "expected_benefit": "+15% pNN50, reduced stress index",
                    "evidence": "400mg/d Mg increases vagal activity",
                    "source": "MMW Fortschr Med 2016",
                }
            )

        # Vitamin D recommendation
        if hrv_prediction.vitamin_d_contribution < 0:
            recommendations.append(
                {
                    "priority": 2,
                    "category": "vitamin_d",
                    "recommendation": "Optimize vitamin D to 40-60 ng/mL",
                    "foods": [
                        "sunlight",
                        "fatty fish",
                        "fortified foods",
                        "supplement",
                    ],
                    "expected_benefit": "Improved SDNN, RMSSD, HF power",
                    "evidence": "VitD deficiency correlates with reduced HRV",
                    "source": "Cardiovasc Ther 2022",
                }
            )

        # Anti-inflammatory recommendation
        if inflammatory.dii_score > 1:
            recommendations.append(
                {
                    "priority": 1,
                    "category": "anti_inflammatory",
                    "recommendation": "Shift to anti-inflammatory diet pattern",
                    "foods": [
                        "berries",
                        "leafy greens",
                        "olive oil",
                        "turmeric",
                        "ginger",
                    ],
                    "expected_benefit": "Reduced CVD risk (RR 0.71 vs high DII)",
                    "evidence": "Meta-analysis of 15 cohort studies",
                    "source": "Atherosclerosis 2020",
                }
            )

        # Neurotransmitter support
        if neurotransmitter_prediction.serotonin_score < 60:
            recommendations.append(
                {
                    "priority": 3,
                    "category": "serotonin_support",
                    "recommendation": "Increase tryptophan-rich foods",
                    "foods": ["turkey", "eggs", "cheese", "nuts", "seeds"],
                    "expected_benefit": "Improved mood, sleep, HRV",
                    "evidence": "Tryptophan depletion reduces HF-HRV",
                    "source": "Biol Psychiatry 2006",
                }
            )

        # Gut-brain axis
        if hrv_prediction.gut_brain_contribution < 0.05:
            recommendations.append(
                {
                    "priority": 3,
                    "category": "gut_brain",
                    "recommendation": "Add probiotics and fermented foods",
                    "foods": ["yogurt", "kefir", "sauerkraut", "kimchi", "kombucha"],
                    "expected_benefit": "Improved vagal tone after 3 months",
                    "evidence": "Multi-species probiotic improves VN function",
                    "source": "Gut Microbes 2025",
                }
            )

        # B6 cofactor
        if not neurotransmitter_prediction.b6_adequate:
            recommendations.append(
                {
                    "priority": 2,
                    "category": "b_vitamins",
                    "recommendation": "Ensure adequate B6 intake (1.3-2mg/day)",
                    "foods": ["chicken", "fish", "potatoes", "bananas", "chickpeas"],
                    "expected_benefit": "Enhanced serotonin and GABA synthesis",
                    "evidence": "B6 required for amino acid decarboxylation",
                    "source": "NCBI StatPearls",
                }
            )

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"])

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_engine() -> NutritionalBiomarkerEngine:
    """Create and return a configured engine instance."""
    return NutritionalBiomarkerEngine()


def quick_hrv_assessment(
    foods: List[Dict[str, Any]], supplements: Optional[Dict[str, float]] = None
) -> HRVPrediction:
    """
    Quick HRV prediction from food list.

    Args:
        foods: List of {name, amount_g} food items
        supplements: Optional dict of supplement amounts

    Returns:
        HRV prediction
    """
    engine = NutritionalBiomarkerEngine()

    # Calculate profiles
    amino_acids = engine.calculate_amino_acid_profile(foods)

    # Default micronutrients (would be calculated from foods in production)
    micronutrients = MicronutrientProfile(
        magnesium_mg=supplements.get("magnesium", 250) if supplements else 250,
        vitamin_d_iu=supplements.get("vitamin_d", 1000) if supplements else 1000,
        epa_g=supplements.get("epa", 0.5) if supplements else 0.5,
        dha_g=supplements.get("dha", 0.5) if supplements else 0.5,
        vitamin_b6_mg=1.5,
        iron_mg=12,
    )

    # Default profiles
    inflammatory = InflammatoryProfile(dii_score=0)
    glycemic = GlycemicProfile(glycemic_load=50)
    biogenic = BiogenicAmineProfile()
    gut_brain = GutBrainProfile()

    return engine.predict_hrv_response(
        amino_acids=amino_acids,
        micronutrients=micronutrients,
        inflammatory=inflammatory,
        glycemic=glycemic,
        biogenic_amines=biogenic,
        gut_brain=gut_brain,
    )
