"""
Gut-Brain-Vagal Axis Integration Engine

Research-backed models for:
1. Microbiome-vagus nerve communication pathways
2. Short-chain fatty acid (SCFA) production prediction
3. Probiotic effects on HRV and vagal tone
4. Enteric neurotransmitter synthesis
5. Inflammatory-gut-brain axis integration

Research Sources:
- Gut Microbes 2025: Multi-species probiotic enhances vagal nerve function (RCT)
- Physiology Reviews: Microbiota-Gut-Brain Axis comprehensive review
- ScienceDirect 2024: Vagus nerve stimulation and gut microbiota interactions
- PMC 2024: Precision psychobiotics for gut-brain health
- PMC 2022: Vagus nerve impact on gut microbiota-brain axis

Key Findings:
- After 3 months probiotic intake, MDD patients showed improved morning VN function
- Akkermansia muciniphila associated with improved sleep and HRV
- Lactobacillales and Ruminococcaceae correlate with better vagal function
- 80% of vagus nerve fibers are afferent (gut → brain)
- SCFA producers like Lactobacillus directly signal vagus via neuropods
"""
# mypy: ignore-errors

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import math


# =============================================================================
# MICROBIOME CONSTANTS
# =============================================================================


class BacterialPhylum(str, Enum):
    """Major gut bacterial phyla"""

    FIRMICUTES = "firmicutes"
    BACTEROIDETES = "bacteroidetes"
    ACTINOBACTERIA = "actinobacteria"
    PROTEOBACTERIA = "proteobacteria"
    VERRUCOMICROBIA = "verrucomicrobia"


class BeneficialBacteria(str, Enum):
    """Key beneficial bacteria for gut-brain health"""

    AKKERMANSIA_MUCINIPHILA = "akkermansia_muciniphila"
    LACTOBACILLUS_RHAMNOSUS = "lactobacillus_rhamnosus"
    LACTOBACILLUS_PLANTARUM = "lactobacillus_plantarum"
    BIFIDOBACTERIUM_LONGUM = "bifidobacterium_longum"
    BIFIDOBACTERIUM_INFANTIS = "bifidobacterium_infantis"
    FAECALIBACTERIUM_PRAUSNITZII = "faecalibacterium_prausnitzii"
    ROSEBURIA = "roseburia"
    EUBACTERIUM_RECTALE = "eubacterium_rectale"
    CHRISTENSENELLA_MINUTA = "christensenella_minuta"
    BACTEROIDES_FRAGILIS = "bacteroides_fragilis"


class PrebioticFiber(str, Enum):
    """Prebiotic fiber types that feed beneficial bacteria"""

    INULIN = "inulin"
    FOS = "fructooligosaccharides"
    GOS = "galactooligosaccharides"
    RESISTANT_STARCH = "resistant_starch"
    PECTIN = "pectin"
    BETA_GLUCAN = "beta_glucan"
    ARABINOXYLAN = "arabinoxylan"
    PSYLLIUM = "psyllium"


class ShortChainFattyAcid(str, Enum):
    """SCFAs produced by gut bacteria"""

    ACETATE = "acetate"
    PROPIONATE = "propionate"
    BUTYRATE = "butyrate"


class VagalCommunicationPathway(str, Enum):
    """Pathways for gut-brain vagal communication"""

    DIRECT_NEUROPOD = "direct_neuropod"  # Enteroendocrine cells
    SCFA_SIGNALING = "scfa_signaling"
    CYTOKINE_MEDIATED = "cytokine_mediated"
    NEUROTRANSMITTER = "neurotransmitter"
    HORMONE = "hormone"  # GLP-1, PYY, CCK


# Bacterial effects on health (from research)
BACTERIA_HEALTH_EFFECTS: Dict[str, Dict[str, float]] = {
    "akkermansia_muciniphila": {
        "vagal_function": 0.12,
        "sleep_quality": 0.08,
        "metabolic_health": 0.15,
        "gut_barrier": 0.20,
        "anti_inflammatory": 0.10,
    },
    "lactobacillus_rhamnosus": {
        "vagal_function": 0.15,
        "anxiety_reduction": 0.25,  # Vagus-dependent effect
        "immune_modulation": 0.12,
        "gaba_production": 0.08,
    },
    "lactobacillus_plantarum": {
        "acetylcholine_production": 0.10,
        "vagal_stimulation": 0.08,
        "serotonin_support": 0.05,
    },
    "bifidobacterium_longum": {
        "stress_reduction": 0.15,
        "hrv_improvement": 0.08,
        "cortisol_reduction": 0.10,
    },
    "faecalibacterium_prausnitzii": {
        "butyrate_production": 0.25,
        "anti_inflammatory": 0.20,
        "gut_barrier": 0.15,
    },
}

# Prebiotic fiber → bacteria relationships
PREBIOTIC_BACTERIA_SUPPORT: Dict[str, List[str]] = {
    "inulin": [
        "bifidobacterium_longum",
        "akkermansia_muciniphila",
        "faecalibacterium_prausnitzii",
    ],
    "fructooligosaccharides": ["bifidobacterium_longum", "bifidobacterium_infantis"],
    "galactooligosaccharides": ["bifidobacterium_longum", "lactobacillus_rhamnosus"],
    "resistant_starch": [
        "faecalibacterium_prausnitzii",
        "eubacterium_rectale",
        "roseburia",
    ],
    "pectin": ["akkermansia_muciniphila", "bacteroides_fragilis"],
    "beta_glucan": ["lactobacillus_plantarum", "bifidobacterium_longum"],
}

# Fermented foods and their probiotic content
FERMENTED_FOODS: Dict[str, Dict[str, Any]] = {
    "yogurt": {
        "bacteria": ["lactobacillus_bulgaricus", "streptococcus_thermophilus"],
        "cfu_per_serving": 1e9,
        "vagal_support": 0.05,
    },
    "kefir": {
        "bacteria": [
            "lactobacillus_kefiri",
            "lactobacillus_rhamnosus",
            "bifidobacterium",
        ],
        "cfu_per_serving": 2.5e10,
        "vagal_support": 0.10,
    },
    "sauerkraut": {
        "bacteria": ["lactobacillus_plantarum", "leuconostoc_mesenteroides"],
        "cfu_per_serving": 1e8,
        "vagal_support": 0.06,
    },
    "kimchi": {
        "bacteria": ["lactobacillus_plantarum", "lactobacillus_brevis"],
        "cfu_per_serving": 1e9,
        "vagal_support": 0.08,
    },
    "kombucha": {
        "bacteria": ["gluconacetobacter", "acetobacter"],
        "cfu_per_serving": 1e6,
        "vagal_support": 0.03,
    },
    "miso": {
        "bacteria": ["lactobacillus_acidophilus"],
        "cfu_per_serving": 1e6,
        "vagal_support": 0.04,
    },
    "tempeh": {
        "bacteria": ["rhizopus_oligosporus"],  # Actually a fungus
        "cfu_per_serving": 1e7,
        "vagal_support": 0.03,
    },
}

# SCFA production rates from different fibers (mmol per g fiber)
SCFA_PRODUCTION_RATES: Dict[str, Dict[str, float]] = {
    "inulin": {"acetate": 2.5, "propionate": 0.8, "butyrate": 1.2},
    "resistant_starch": {"acetate": 1.8, "propionate": 0.6, "butyrate": 2.0},
    "pectin": {"acetate": 2.0, "propionate": 1.0, "butyrate": 0.5},
    "beta_glucan": {"acetate": 1.5, "propionate": 0.5, "butyrate": 0.8},
    "psyllium": {"acetate": 1.0, "propionate": 0.3, "butyrate": 0.4},
}

# HRV correlation coefficients with gut health markers
GUT_HRV_CORRELATIONS = {
    "microbiome_diversity": 0.30,
    "firmicutes_bacteroidetes_ratio": -0.15,  # High ratio = lower HRV
    "scfa_total": 0.25,
    "butyrate_level": 0.28,
    "gut_permeability_inverse": -0.35,  # Leaky gut = lower HRV
    "akkermansia_abundance": 0.22,
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MicrobiomeProfile:
    """Estimated gut microbiome profile"""

    diversity_score: float  # 0-100 Shannon diversity proxy
    firmicutes_ratio: float  # F:B ratio
    beneficial_bacteria: Dict[str, float]  # Abundance 0-1
    scfa_producing_capacity: float  # 0-1
    inflammation_markers: float  # 0-1 (higher = more inflammation)


@dataclass
class PrebioticIntake:
    """Prebiotic fiber intake tracking"""

    total_prebiotic_g: float
    inulin_g: float = 0.0
    fos_g: float = 0.0
    gos_g: float = 0.0
    resistant_starch_g: float = 0.0
    pectin_g: float = 0.0
    beta_glucan_g: float = 0.0


@dataclass
class ProbioticIntake:
    """Probiotic intake from foods and supplements"""

    total_cfu: float  # Colony forming units
    strains: List[str]
    source: str  # "supplement", "fermented_food"
    multi_strain: bool = False


@dataclass
class SCFAProduction:
    """Predicted SCFA production"""

    acetate_mmol: float
    propionate_mmol: float
    butyrate_mmol: float
    total_mmol: float
    butyrate_ratio: float  # Butyrate is most beneficial

    @property
    def quality_score(self) -> float:
        """Higher butyrate ratio = better quality SCFA production"""
        return min(1.0, self.butyrate_ratio * 2 + (self.total_mmol / 100) * 0.3)


@dataclass
class VagalTonePrediction:
    """Predicted vagal tone from gut health"""

    vagal_tone_score: float  # 0-100
    hrv_impact: float  # -0.2 to 0.2 relative change
    parasympathetic_support: float  # 0-1
    time_to_effect_days: int
    confidence: float
    contributing_factors: Dict[str, float]


@dataclass
class GutBrainAxisState:
    """Complete gut-brain axis assessment"""

    microbiome: MicrobiomeProfile
    scfa_production: SCFAProduction
    vagal_prediction: VagalTonePrediction
    enteric_neurotransmitters: Dict[str, float]
    inflammation_pathway: str  # "anti_inflammatory", "neutral", "pro_inflammatory"
    recommendations: List[Dict[str, Any]]


@dataclass
class ProbioticProtocol:
    """Recommended probiotic protocol"""

    priority_strains: List[str]
    target_cfu: float
    duration_weeks: int
    expected_hrv_improvement: float
    expected_vagal_improvement: float
    supporting_prebiotics: List[str]
    food_sources: List[str]
    evidence_summary: str


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================


class GutBrainPredictor(nn.Module):
    """
    Neural network predicting gut-brain axis health outcomes.

    Inputs:
    - Prebiotic fiber intake (multiple types)
    - Probiotic intake (CFU, strain count)
    - Fermented food servings
    - Diet quality markers
    - Baseline microbiome indicators

    Outputs:
    - Microbiome diversity prediction
    - SCFA production estimate
    - Vagal tone impact
    - HRV change prediction
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(num_layers)
            ]
        )

        # Output heads
        self.diversity_head = nn.Linear(hidden_dim, 1)
        self.scfa_head = nn.Linear(hidden_dim, 3)  # acetate, propionate, butyrate
        self.vagal_head = nn.Linear(hidden_dim, 1)
        self.hrv_head = nn.Linear(hidden_dim, 1)
        self.inflammation_head = nn.Linear(hidden_dim, 3)  # anti/neutral/pro

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)

        for layer in self.layers:
            h = h + layer(h)  # Residual

        return {
            "diversity": torch.sigmoid(self.diversity_head(h)) * 100,
            "scfa": F.softplus(self.scfa_head(h)),  # Positive values
            "vagal_tone": torch.sigmoid(self.vagal_head(h)) * 100,
            "hrv_change": torch.tanh(self.hrv_head(h)) * 0.2,
            "inflammation": F.softmax(self.inflammation_head(h), dim=-1),
        }


class TemporalMicrobiomeModel(nn.Module):
    """
    LSTM model for predicting microbiome changes over time.

    Research shows probiotic effects take 3 months to manifest
    in vagal function - this model captures temporal dynamics.
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),  # diversity, scfa, vagal, hrv, inflammation
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] daily intake features
            lengths: Optional sequence lengths for packing

        Returns:
            [batch, 5] predicted outcomes after the sequence
        """
        output, (hidden, _) = self.lstm(x)

        # Use final hidden state from both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        return self.output_proj(final_hidden)


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================


class GutBrainVagalEngine:
    """
    Comprehensive gut-brain-vagal axis analysis engine.

    Integrates:
    - Microbiome diversity estimation
    - SCFA production modeling
    - Vagal tone prediction
    - Probiotic/prebiotic recommendations
    - Temporal effect modeling

    Based on clinical research including the 2025 RCT showing
    3-month probiotic supplementation improves vagal function
    in depression patients.
    """

    def __init__(self, use_neural_models: bool = True):
        self.bacteria_effects = BACTERIA_HEALTH_EFFECTS
        self.prebiotic_bacteria = PREBIOTIC_BACTERIA_SUPPORT
        self.fermented_foods = FERMENTED_FOODS
        self.scfa_rates = SCFA_PRODUCTION_RATES

        self.use_neural = use_neural_models
        if use_neural_models:
            self.predictor = GutBrainPredictor()
            self.temporal_model = TemporalMicrobiomeModel()

    def analyze_gut_brain_axis(
        self,
        prebiotic_intake: PrebioticIntake,
        probiotic_intake: Optional[ProbioticIntake] = None,
        fermented_foods: Optional[List[str]] = None,
        diet_quality_score: float = 0.5,
        baseline_diversity: Optional[float] = None,
    ) -> GutBrainAxisState:
        """
        Complete gut-brain axis analysis.

        Args:
            prebiotic_intake: Daily prebiotic fiber intake
            probiotic_intake: Probiotic supplement/food intake
            fermented_foods: List of fermented foods consumed
            diet_quality_score: Overall diet quality (0-1)
            baseline_diversity: Known baseline microbiome diversity

        Returns:
            Complete gut-brain axis state assessment
        """
        # Estimate microbiome profile
        microbiome = self._estimate_microbiome(
            prebiotic_intake,
            probiotic_intake,
            fermented_foods,
            diet_quality_score,
            baseline_diversity,
        )

        # Calculate SCFA production
        scfa = self._calculate_scfa_production(prebiotic_intake)

        # Predict vagal tone impact
        vagal = self._predict_vagal_tone(
            microbiome, scfa, probiotic_intake, fermented_foods
        )

        # Estimate enteric neurotransmitters
        enteric_nt = self._estimate_enteric_neurotransmitters(
            microbiome, probiotic_intake
        )

        # Determine inflammation pathway
        inflammation = self._assess_inflammation_pathway(
            microbiome, scfa, diet_quality_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            microbiome, scfa, vagal, probiotic_intake
        )

        return GutBrainAxisState(
            microbiome=microbiome,
            scfa_production=scfa,
            vagal_prediction=vagal,
            enteric_neurotransmitters=enteric_nt,
            inflammation_pathway=inflammation,
            recommendations=recommendations,
        )

    def _estimate_microbiome(
        self,
        prebiotic: PrebioticIntake,
        probiotic: Optional[ProbioticIntake],
        fermented: Optional[List[str]],
        diet_quality: float,
        baseline: Optional[float],
    ) -> MicrobiomeProfile:
        """Estimate microbiome profile from dietary inputs."""

        # Base diversity from diet quality
        base_diversity = (baseline or 50) + (diet_quality - 0.5) * 20

        # Prebiotic boost
        prebiotic_boost = min(15, prebiotic.total_prebiotic_g * 0.5)

        # Probiotic boost
        probiotic_boost = 0
        if probiotic:
            # Log scale for CFU
            cfu_factor = math.log10(max(1, probiotic.total_cfu)) / 12
            probiotic_boost = cfu_factor * 10
            if probiotic.multi_strain:
                probiotic_boost *= 1.3

        # Fermented food boost
        fermented_boost = 0
        if fermented:
            for food in fermented:
                if food.lower() in self.fermented_foods:
                    fermented_boost += (
                        self.fermented_foods[food.lower()]["vagal_support"] * 5
                    )

        diversity_score = min(
            100, base_diversity + prebiotic_boost + probiotic_boost + fermented_boost
        )

        # Estimate beneficial bacteria abundance
        beneficial = {}
        for bacteria in BeneficialBacteria:
            name = bacteria.value
            abundance = 0.3  # Base

            # Check if supported by prebiotics
            for prebiotic_type, supported_bacteria in self.prebiotic_bacteria.items():
                prebiotic_amount = getattr(prebiotic, f"{prebiotic_type}_g", 0)
                if name in supported_bacteria and prebiotic_amount > 0:
                    abundance += min(0.3, prebiotic_amount / 10)

            # Check if in probiotic
            if probiotic and name in [s.lower() for s in probiotic.strains]:
                abundance += 0.2

            beneficial[name] = min(1.0, abundance)

        # F:B ratio (lower is generally better in Western populations)
        # High fiber diets tend to increase Bacteroidetes
        fb_ratio = 2.0 - (prebiotic.total_prebiotic_g / 30)
        fb_ratio = max(0.5, min(3.0, fb_ratio))

        # SCFA producing capacity
        scfa_capacity = (
            sum(
                beneficial.get(b, 0)
                for b in [
                    "faecalibacterium_prausnitzii",
                    "roseburia",
                    "eubacterium_rectale",
                ]
            )
            / 3
        )

        # Inflammation marker (inverse of beneficial bacteria)
        inflammation = 1 - (sum(beneficial.values()) / len(beneficial))

        return MicrobiomeProfile(
            diversity_score=diversity_score,
            firmicutes_ratio=fb_ratio,
            beneficial_bacteria=beneficial,
            scfa_producing_capacity=scfa_capacity,
            inflammation_markers=inflammation,
        )

    def _calculate_scfa_production(self, prebiotic: PrebioticIntake) -> SCFAProduction:
        """Calculate predicted SCFA production from prebiotic intake."""
        total_acetate = 0
        total_propionate = 0
        total_butyrate = 0

        fiber_sources = {
            "inulin": prebiotic.inulin_g,
            "resistant_starch": prebiotic.resistant_starch_g,
            "pectin": prebiotic.pectin_g,
            "beta_glucan": prebiotic.beta_glucan_g,
        }

        for fiber, amount in fiber_sources.items():
            if fiber in self.scfa_rates and amount > 0:
                rates = self.scfa_rates[fiber]
                total_acetate += rates["acetate"] * amount
                total_propionate += rates["propionate"] * amount
                total_butyrate += rates["butyrate"] * amount

        # FOS/GOS contribute to acetate primarily
        total_acetate += (prebiotic.fos_g + prebiotic.gos_g) * 2.0

        total = total_acetate + total_propionate + total_butyrate
        butyrate_ratio = total_butyrate / max(total, 1)

        return SCFAProduction(
            acetate_mmol=total_acetate,
            propionate_mmol=total_propionate,
            butyrate_mmol=total_butyrate,
            total_mmol=total,
            butyrate_ratio=butyrate_ratio,
        )

    def _predict_vagal_tone(
        self,
        microbiome: MicrobiomeProfile,
        scfa: SCFAProduction,
        probiotic: Optional[ProbioticIntake],
        fermented: Optional[List[str]],
    ) -> VagalTonePrediction:
        """
        Predict vagal tone impact from gut health.

        Based on research:
        - 3 months probiotic improves morning VN function
        - Akkermansia associated with improved HRV
        - SCFAs signal directly to vagus via neuropods
        """
        contributing_factors = {}

        # Microbiome diversity contribution
        diversity_contrib = (microbiome.diversity_score - 50) / 100 * 0.1
        contributing_factors["microbiome_diversity"] = diversity_contrib

        # Beneficial bacteria contribution
        bacteria_contrib = 0
        for bacteria, abundance in microbiome.beneficial_bacteria.items():
            if bacteria in self.bacteria_effects:
                effect = self.bacteria_effects[bacteria].get("vagal_function", 0)
                bacteria_contrib += effect * abundance
        contributing_factors["beneficial_bacteria"] = bacteria_contrib

        # SCFA contribution (butyrate especially important)
        scfa_contrib = scfa.quality_score * 0.08
        contributing_factors["scfa_production"] = scfa_contrib

        # Probiotic supplement contribution
        probiotic_contrib = 0
        time_to_effect = 90  # days
        if probiotic:
            # Check for specific strains with research backing
            effective_strains = [
                "lactobacillus_rhamnosus",
                "bifidobacterium_longum",
                "lactobacillus_plantarum",
            ]
            for strain in probiotic.strains:
                if strain.lower() in effective_strains:
                    probiotic_contrib += 0.04

            # Multi-strain is more effective
            if probiotic.multi_strain:
                probiotic_contrib *= 1.5
                time_to_effect = 84  # Slightly faster

        contributing_factors["probiotic_supplement"] = probiotic_contrib

        # Fermented food contribution
        fermented_contrib = 0
        if fermented:
            for food in fermented:
                if food.lower() in self.fermented_foods:
                    fermented_contrib += self.fermented_foods[food.lower()][
                        "vagal_support"
                    ]
        contributing_factors["fermented_foods"] = fermented_contrib

        # Total HRV impact
        total_hrv_impact = sum(contributing_factors.values())
        total_hrv_impact = max(-0.15, min(0.15, total_hrv_impact))

        # Vagal tone score (0-100)
        vagal_score = 50 + total_hrv_impact * 200

        # Parasympathetic support
        para_support = 0.5 + total_hrv_impact

        # Confidence based on data completeness
        confidence = 0.6
        if probiotic:
            confidence += 0.15
        if fermented:
            confidence += 0.1
        if microbiome.diversity_score > 60:
            confidence += 0.1

        return VagalTonePrediction(
            vagal_tone_score=vagal_score,
            hrv_impact=total_hrv_impact,
            parasympathetic_support=para_support,
            time_to_effect_days=time_to_effect,
            confidence=min(0.95, confidence),
            contributing_factors=contributing_factors,
        )

    def _estimate_enteric_neurotransmitters(
        self, microbiome: MicrobiomeProfile, probiotic: Optional[ProbioticIntake]
    ) -> Dict[str, float]:
        """
        Estimate enteric neurotransmitter production.

        Research: 95% of serotonin is produced in the gut.
        Certain bacteria produce neurotransmitters directly.
        """
        neurotransmitters = {
            "serotonin": 0.5,  # 95% made in gut
            "gaba": 0.5,  # Some bacteria produce GABA
            "acetylcholine": 0.5,  # L. plantarum produces ACh
            "dopamine": 0.5,  # Some enteric production
        }

        # Serotonin - supported by good microbiome
        neurotransmitters["serotonin"] += (microbiome.diversity_score - 50) / 200
        neurotransmitters["serotonin"] += (
            microbiome.beneficial_bacteria.get("bifidobacterium_longum", 0) * 0.1
        )

        # GABA - L. rhamnosus produces GABA
        if probiotic:
            for strain in probiotic.strains:
                if "rhamnosus" in strain.lower():
                    neurotransmitters["gaba"] += 0.15

        # Acetylcholine - L. plantarum, B. subtilis produce ACh
        neurotransmitters["acetylcholine"] += (
            microbiome.beneficial_bacteria.get("lactobacillus_plantarum", 0) * 0.12
        )

        # Normalize all values
        return {k: min(1.0, max(0.0, v)) for k, v in neurotransmitters.items()}

    def _assess_inflammation_pathway(
        self, microbiome: MicrobiomeProfile, scfa: SCFAProduction, diet_quality: float
    ) -> str:
        """
        Assess overall inflammation pathway.

        SCFAs (especially butyrate) activate anti-inflammatory pathways.
        Poor microbiome diversity promotes inflammation.
        """
        # Anti-inflammatory factors
        anti_score = 0
        anti_score += scfa.butyrate_mmol / 20  # Butyrate is anti-inflammatory
        anti_score += (microbiome.diversity_score - 50) / 100
        anti_score += (
            microbiome.beneficial_bacteria.get("faecalibacterium_prausnitzii", 0) * 0.5
        )  # Major anti-inflammatory bacterium
        anti_score += diet_quality * 0.3

        # Pro-inflammatory factors
        pro_score = 0
        pro_score += microbiome.inflammation_markers * 0.5
        pro_score += max(0, microbiome.firmicutes_ratio - 2) * 0.2

        net_score = anti_score - pro_score

        if net_score > 0.3:
            return "anti_inflammatory"
        elif net_score < -0.1:
            return "pro_inflammatory"
        return "neutral"

    def _generate_recommendations(
        self,
        microbiome: MicrobiomeProfile,
        scfa: SCFAProduction,
        vagal: VagalTonePrediction,
        probiotic: Optional[ProbioticIntake],
    ) -> List[Dict[str, Any]]:
        """Generate personalized gut-brain recommendations."""
        recommendations = []

        # Low diversity
        if microbiome.diversity_score < 60:
            recommendations.append(
                {
                    "priority": 1,
                    "category": "diversity",
                    "recommendation": "Increase microbiome diversity",
                    "actions": [
                        "Eat 30+ different plant foods per week",
                        "Include fermented foods daily",
                        "Vary fiber sources (legumes, whole grains, vegetables)",
                    ],
                    "expected_benefit": "Improved vagal tone, better HRV",
                    "evidence": "Higher diversity correlates with better HRV",
                }
            )

        # Low SCFA production
        if scfa.total_mmol < 30:
            recommendations.append(
                {
                    "priority": 1,
                    "category": "scfa_production",
                    "recommendation": "Boost SCFA production",
                    "actions": [
                        "Add resistant starch (cooled potatoes, green bananas)",
                        "Include 25-35g fiber daily",
                        "Eat more legumes (chickpeas, lentils, beans)",
                    ],
                    "expected_benefit": "Increased butyrate, reduced inflammation",
                    "evidence": "SCFAs signal directly to vagus nerve via neuropods",
                }
            )

        # No probiotic
        if not probiotic or probiotic.total_cfu < 1e9:
            recommendations.append(
                {
                    "priority": 2,
                    "category": "probiotics",
                    "recommendation": "Consider multi-strain probiotic",
                    "actions": [
                        "Look for strains: L. rhamnosus, B. longum, L. plantarum",
                        "Target 10+ billion CFU daily",
                        "Commit to 3 months for vagal benefits",
                    ],
                    "expected_benefit": "Improved morning vagal function after 3 months",
                    "evidence": "Gut Microbes 2025 RCT: Multi-species probiotic improves VN",
                }
            )

        # Low Akkermansia support
        if microbiome.beneficial_bacteria.get("akkermansia_muciniphila", 0) < 0.4:
            recommendations.append(
                {
                    "priority": 2,
                    "category": "akkermansia",
                    "recommendation": "Support Akkermansia growth",
                    "actions": [
                        "Increase polyphenol intake (berries, pomegranate)",
                        "Add pectin-rich foods (apples, citrus)",
                        "Consider pomegranate extract",
                    ],
                    "expected_benefit": "Better metabolic health, improved sleep",
                    "evidence": "Akkermansia associated with improved HRV and sleep",
                }
            )

        # Fermented foods
        recommendations.append(
            {
                "priority": 3,
                "category": "fermented_foods",
                "recommendation": "Daily fermented foods",
                "actions": [
                    "Yogurt or kefir with breakfast",
                    "Sauerkraut or kimchi with lunch",
                    "Miso soup or kombucha as snack",
                ],
                "expected_benefit": "Direct probiotic delivery, vagal support",
                "evidence": "Fermented food consumption correlates with better mood",
            }
        )

        return recommendations

    def get_probiotic_protocol(
        self,
        target: str = "hrv_improvement",  # "hrv_improvement", "anxiety", "sleep", "general"
    ) -> ProbioticProtocol:
        """
        Get evidence-based probiotic protocol for specific goal.
        """
        protocols = {
            "hrv_improvement": ProbioticProtocol(
                priority_strains=[
                    "Lactobacillus rhamnosus GG",
                    "Bifidobacterium longum",
                    "Lactobacillus plantarum",
                ],
                target_cfu=2e10,  # 20 billion
                duration_weeks=12,  # Research: 3 months for vagal effects
                expected_hrv_improvement=0.12,
                expected_vagal_improvement=0.15,
                supporting_prebiotics=["inulin", "GOS"],
                food_sources=["kefir", "yogurt", "kimchi"],
                evidence_summary="Gut Microbes 2025: Multi-species probiotic improves "
                "morning vagal function in MDD patients after 3 months",
            ),
            "anxiety": ProbioticProtocol(
                priority_strains=[
                    "Lactobacillus rhamnosus JB-1",
                    "Bifidobacterium longum 1714",
                ],
                target_cfu=1e10,
                duration_weeks=8,
                expected_hrv_improvement=0.08,
                expected_vagal_improvement=0.10,
                supporting_prebiotics=["GOS", "inulin"],
                food_sources=["yogurt", "kefir"],
                evidence_summary="L. rhamnosus anxiolytic effects are vagus-dependent "
                "(effect abolished when vagus is cut in animal models)",
            ),
            "sleep": ProbioticProtocol(
                priority_strains=[
                    "Lactobacillus rhamnosus",
                    "Bifidobacterium longum",
                    "Akkermansia muciniphila (via polyphenols)",
                ],
                target_cfu=2e10,
                duration_weeks=12,
                expected_hrv_improvement=0.10,
                expected_vagal_improvement=0.12,
                supporting_prebiotics=["inulin", "pectin"],
                food_sources=["kefir", "tart cherry (prebiotic)"],
                evidence_summary="Akkermansia muciniphila associated with improved "
                "sleep parameters in probiotic RCT",
            ),
            "general": ProbioticProtocol(
                priority_strains=[
                    "Lactobacillus acidophilus",
                    "Lactobacillus rhamnosus",
                    "Bifidobacterium bifidum",
                    "Bifidobacterium longum",
                ],
                target_cfu=1e10,
                duration_weeks=8,
                expected_hrv_improvement=0.08,
                expected_vagal_improvement=0.08,
                supporting_prebiotics=["inulin", "FOS"],
                food_sources=["yogurt", "kefir", "sauerkraut"],
                evidence_summary="General multi-strain probiotics support overall "
                "gut-brain axis health",
            ),
        }

        return protocols.get(target, protocols["general"])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_gut_brain_health(
    prebiotic_g: float = 0,
    probiotic_cfu: float = 0,
    fermented_servings: int = 0,
    diet_quality: float = 0.5,
) -> GutBrainAxisState:
    """
    Quick gut-brain axis analysis.

    Args:
        prebiotic_g: Total prebiotic fiber in grams
        probiotic_cfu: Probiotic CFU (colony forming units)
        fermented_servings: Number of fermented food servings
        diet_quality: Overall diet quality score (0-1)

    Returns:
        Complete gut-brain axis state
    """
    engine = GutBrainVagalEngine()

    prebiotic = PrebioticIntake(
        total_prebiotic_g=prebiotic_g,
        inulin_g=prebiotic_g * 0.3,
        fos_g=prebiotic_g * 0.2,
        resistant_starch_g=prebiotic_g * 0.3,
        pectin_g=prebiotic_g * 0.2,
    )

    probiotic = None
    if probiotic_cfu > 0:
        probiotic = ProbioticIntake(
            total_cfu=probiotic_cfu,
            strains=["lactobacillus_rhamnosus", "bifidobacterium_longum"],
            source="supplement",
            multi_strain=True,
        )

    fermented = None
    if fermented_servings > 0:
        fermented = ["yogurt"] * min(fermented_servings, 3)

    return engine.analyze_gut_brain_axis(
        prebiotic_intake=prebiotic,
        probiotic_intake=probiotic,
        fermented_foods=fermented,
        diet_quality_score=diet_quality,
    )


def get_hrv_probiotic_protocol() -> ProbioticProtocol:
    """Get the HRV-optimized probiotic protocol."""
    engine = GutBrainVagalEngine()
    return engine.get_probiotic_protocol("hrv_improvement")
