"""
Amino Acid Metabolism Tracker and Health Impact Predictor

Implements research-backed models for:
1. Tryptophan-Serotonin-HRV pathway modeling
2. BCAA competition and brain amino acid transport
3. Tyrosine-Dopamine-Catecholamine cascade
4. Glycine cardiovascular protection mechanisms
5. Temporal effects of amino acid intake on health metrics

Research Sources:
- ScienceDirect: Tryptophan depletion reduces HF-HRV in remitted patients
- Frontiers 2025: Blood metabolome signatures predict HRV in obesity
- MDPI 2024: Glycine cardiovascular health narrative review
- Sports Medicine Open 2024: BCAA meta-analysis with meta-regression
- Cambridge Nutrition Reviews 2024: BCAA-MPS molecular signaling update
"""
# mypy: ignore-errors

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import math
from collections import deque


# =============================================================================
# METABOLIC PATHWAY CONSTANTS
# =============================================================================


class MetabolicPathway(str, Enum):
    """Major amino acid metabolic pathways"""

    TRYPTOPHAN_SEROTONIN = "tryptophan_serotonin"
    TRYPTOPHAN_KYNURENINE = "tryptophan_kynurenine"
    TYROSINE_CATECHOLAMINE = "tyrosine_catecholamine"
    PHENYLALANINE_TYROSINE = "phenylalanine_tyrosine"
    GLUTAMATE_GABA = "glutamate_gaba"
    GLYCINE_COLLAGEN = "glycine_collagen"
    GLYCINE_GLUTATHIONE = "glycine_glutathione"
    HISTIDINE_HISTAMINE = "histidine_histamine"
    ARGININE_NITRIC_OXIDE = "arginine_nitric_oxide"
    METHIONINE_SAM = "methionine_sam"
    BCAA_MUSCLE = "bcaa_muscle"


# Enzyme kinetics parameters (Km in μM, Vmax relative)
ENZYME_KINETICS = {
    "tryptophan_hydroxylase": {
        "km": 50,  # μM
        "vmax_relative": 1.0,
        "cofactors": ["bh4", "fe2+", "o2"],
        "inhibitors": ["phenylalanine_excess"],
    },
    "aromatic_l_amino_acid_decarboxylase": {
        "km": 40,
        "vmax_relative": 1.2,
        "cofactors": ["plp"],  # Vitamin B6
        "inhibitors": [],
    },
    "tyrosine_hydroxylase": {
        "km": 40,
        "vmax_relative": 1.0,
        "cofactors": ["bh4", "fe2+", "o2"],
        "inhibitors": ["dopamine"],  # Product inhibition
    },
    "glutamate_decarboxylase": {
        "km": 800,
        "vmax_relative": 0.8,
        "cofactors": ["plp"],  # B6 critical
        "inhibitors": [],
    },
    "diamine_oxidase": {
        "km": 20,
        "vmax_relative": 1.0,
        "cofactors": ["cu2+"],
        "inhibitors": ["alcohol", "certain_drugs"],
    },
}

# Large Neutral Amino Acid Transporter (LAT1) competition
# Ki values for brain uptake competition (lower = stronger competitor)
LAT1_COMPETITION = {
    "tryptophan": {"ki": 15, "brain_uptake_priority": 0.7},
    "tyrosine": {"ki": 40, "brain_uptake_priority": 0.85},
    "phenylalanine": {"ki": 20, "brain_uptake_priority": 0.9},
    "leucine": {"ki": 25, "brain_uptake_priority": 0.95},
    "isoleucine": {"ki": 60, "brain_uptake_priority": 0.8},
    "valine": {"ki": 200, "brain_uptake_priority": 0.75},
    "methionine": {"ki": 40, "brain_uptake_priority": 0.8},
    "histidine": {"ki": 100, "brain_uptake_priority": 0.6},
}

# Amino acid half-lives and absorption kinetics (hours)
PHARMACOKINETICS = {
    "tryptophan": {"t_max": 1.5, "t_half": 2.5, "bioavailability": 0.9},
    "tyrosine": {"t_max": 1.0, "t_half": 2.0, "bioavailability": 0.85},
    "leucine": {"t_max": 0.5, "t_half": 1.5, "bioavailability": 0.95},
    "isoleucine": {"t_max": 0.5, "t_half": 1.5, "bioavailability": 0.95},
    "valine": {"t_max": 0.5, "t_half": 2.0, "bioavailability": 0.95},
    "glycine": {"t_max": 0.75, "t_half": 3.0, "bioavailability": 0.85},
    "arginine": {"t_max": 1.0, "t_half": 1.5, "bioavailability": 0.7},
    "glutamine": {"t_max": 0.5, "t_half": 1.0, "bioavailability": 0.65},
}

# Health outcome correlation coefficients (from research)
HEALTH_CORRELATIONS = {
    "tryptophan_serotonin": {
        "hrv_hf_power": 0.35,  # Higher tryptophan → higher HF-HRV
        "mood_score": 0.45,
        "sleep_quality": 0.40,
        "anxiety_inverse": -0.30,
    },
    "tyrosine_dopamine": {
        "cognitive_performance": 0.35,
        "stress_resilience": 0.30,
        "motivation": 0.40,
        "hrv_under_stress": 0.25,
    },
    "glycine_cardiovascular": {
        "oxidative_stress_inverse": -0.40,
        "inflammation_inverse": -0.35,
        "hrv_sdnn": 0.30,
        "blood_pressure_inverse": -0.25,
    },
    "bcaa_recovery": {
        "muscle_damage_inverse": -0.44,  # From meta-analysis
        "doms_inverse": -0.55,
        "mps_rate": 0.40,
        "recovery_hrv": 0.20,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class AminoAcidIntake:
    """Single amino acid intake event"""

    amino_acid: str
    amount_mg: float
    timestamp: datetime
    source: str  # "food", "supplement", "protein_shake"
    with_carbs: bool = False  # Carbs + insulin enhance uptake
    fasted: bool = False  # Fasted state affects absorption


@dataclass
class PlasmaConcentration:
    """Modeled plasma amino acid concentration"""

    amino_acid: str
    concentration_umol: float
    timestamp: datetime
    brain_availability: float  # 0-1, accounting for competition


@dataclass
class PathwayActivity:
    """Metabolic pathway activity level"""

    pathway: MetabolicPathway
    activity_level: float  # 0-1 normalized
    rate_limiting_factor: Optional[str] = None
    cofactor_status: Dict[str, float] = field(default_factory=dict)
    product_accumulation: float = 0.0


@dataclass
class NeurotransmitterSynthesis:
    """Predicted neurotransmitter synthesis rate"""

    neurotransmitter: str
    synthesis_rate: float  # Relative to baseline
    precursor_availability: float
    enzyme_activity: float
    cofactor_adequacy: float
    predicted_brain_level: str  # "low", "normal", "elevated"


@dataclass
class HealthOutcomePrediction:
    """Predicted health outcome from amino acid status"""

    outcome: str
    predicted_change: float  # -1 to 1 relative change
    confidence: float
    time_to_effect_hours: float
    mechanism: str
    supporting_evidence: str


@dataclass
class MetabolicState:
    """Complete metabolic state snapshot"""

    timestamp: datetime
    plasma_amino_acids: Dict[str, float]  # μmol/L
    pathway_activities: Dict[str, float]
    neurotransmitter_estimates: Dict[str, float]
    hrv_prediction: float
    mood_prediction: float
    energy_prediction: float
    sleep_prediction: float


# =============================================================================
# PHARMACOKINETIC MODEL
# =============================================================================


class AminoAcidPharmacokinetics:
    """
    Models amino acid absorption, distribution, and metabolism.

    Uses one-compartment model with first-order absorption and elimination.
    """

    def __init__(self):
        self.kinetics = PHARMACOKINETICS
        self.lat1_competition = LAT1_COMPETITION

    def calculate_plasma_concentration(
        self, intakes: List[AminoAcidIntake], current_time: datetime
    ) -> Dict[str, PlasmaConcentration]:
        """
        Calculate current plasma concentrations from intake history.

        Uses: C(t) = (F * D * ka / (V * (ka - ke))) * (e^(-ke*t) - e^(-ka*t))

        Where:
            F = bioavailability
            D = dose
            ka = absorption rate constant
            ke = elimination rate constant
            V = volume of distribution (assumed)
        """
        concentrations = {}

        for aa_name in self.kinetics.keys():
            total_conc = 0.0
            params = self.kinetics[aa_name]

            # Calculate contribution from each intake
            for intake in intakes:
                if intake.amino_acid.lower() != aa_name:
                    continue

                # Time since intake (hours)
                dt = (current_time - intake.timestamp).total_seconds() / 3600

                if dt < 0 or dt > 24:  # Only consider last 24h
                    continue

                # Pharmacokinetic parameters
                t_max = params["t_max"]
                t_half = params["t_half"]
                bioavail = params["bioavailability"]

                # Modify for fed/fasted state
                if intake.with_carbs:
                    bioavail *= 1.15  # Insulin enhances uptake
                if intake.fasted:
                    t_max *= 0.7  # Faster absorption when fasted

                # Rate constants
                ke = 0.693 / t_half  # Elimination
                ka = 2.5 / t_max  # Absorption (empirical)

                # One-compartment model
                # Simplified: peak at t_max, exponential decay
                if dt <= t_max:
                    # Rising phase
                    conc_fraction = 1 - math.exp(-ka * dt)
                else:
                    # Decay phase
                    peak_fraction = 1 - math.exp(-ka * t_max)
                    decay = math.exp(-ke * (dt - t_max))
                    conc_fraction = peak_fraction * decay

                # Convert mg to μmol (rough MW estimates)
                mw_estimates = {
                    "tryptophan": 204,
                    "tyrosine": 181,
                    "leucine": 131,
                    "isoleucine": 131,
                    "valine": 117,
                    "glycine": 75,
                    "arginine": 174,
                    "glutamine": 146,
                }
                mw = mw_estimates.get(aa_name, 150)

                # Dose contribution (μmol/L, assume 5L distribution volume)
                dose_umol = (intake.amount_mg * 1000 / mw) * bioavail
                contribution = dose_umol * conc_fraction / 5000  # Per liter

                total_conc += contribution

            # Calculate brain availability (LAT1 competition)
            brain_avail = self._calculate_brain_availability(aa_name, concentrations)

            if total_conc > 0:
                concentrations[aa_name] = PlasmaConcentration(
                    amino_acid=aa_name,
                    concentration_umol=total_conc,
                    timestamp=current_time,
                    brain_availability=brain_avail,
                )

        return concentrations

    def _calculate_brain_availability(
        self, amino_acid: str, current_concentrations: Dict[str, PlasmaConcentration]
    ) -> float:
        """
        Calculate brain availability accounting for LAT1 competition.

        Research: BCAAs compete with tryptophan for brain uptake.
        High BCAA:tryptophan ratio reduces brain tryptophan.
        """
        if amino_acid not in self.lat1_competition:
            return 0.8  # Default

        target_params = self.lat1_competition[amino_acid]
        base_priority = target_params["brain_uptake_priority"]

        # Calculate competition from other amino acids
        competition_factor = 1.0
        for other_aa, conc in current_concentrations.items():
            if other_aa == amino_acid:
                continue
            if other_aa in self.lat1_competition:
                other_params = self.lat1_competition[other_aa]
                # Higher concentration and lower Ki = stronger competition
                if conc.concentration_umol > 0:
                    competition = conc.concentration_umol / other_params["ki"]
                    competition_factor += competition * 0.1

        # Brain availability decreases with competition
        brain_availability = base_priority / competition_factor

        return min(1.0, max(0.1, brain_availability))


# =============================================================================
# PATHWAY MODELING
# =============================================================================


class MetabolicPathwayModel:
    """
    Models amino acid metabolic pathways and enzyme kinetics.

    Implements Michaelis-Menten kinetics for key enzymes with
    cofactor dependencies and product inhibition.
    """

    def __init__(self):
        self.enzyme_kinetics = ENZYME_KINETICS

    def calculate_pathway_activity(
        self,
        pathway: MetabolicPathway,
        substrate_concentration: float,
        cofactor_status: Dict[str, float],
    ) -> PathwayActivity:
        """
        Calculate metabolic pathway activity using enzyme kinetics.

        Uses Michaelis-Menten: v = Vmax * [S] / (Km + [S])
        Modified for cofactor dependencies.
        """
        # Map pathways to rate-limiting enzymes
        pathway_enzymes = {
            MetabolicPathway.TRYPTOPHAN_SEROTONIN: "tryptophan_hydroxylase",
            MetabolicPathway.TYROSINE_CATECHOLAMINE: "tyrosine_hydroxylase",
            MetabolicPathway.GLUTAMATE_GABA: "glutamate_decarboxylase",
            MetabolicPathway.HISTIDINE_HISTAMINE: "diamine_oxidase",
        }

        enzyme_name = pathway_enzymes.get(pathway)

        if enzyme_name and enzyme_name in self.enzyme_kinetics:
            enzyme = self.enzyme_kinetics[enzyme_name]

            # Base Michaelis-Menten
            km = enzyme["km"]
            vmax = enzyme["vmax_relative"]
            base_activity = (
                vmax * substrate_concentration / (km + substrate_concentration)
            )

            # Cofactor modification
            cofactor_modifier = 1.0
            rate_limiting_factor = None
            cofactor_details = {}

            for cofactor in enzyme["cofactors"]:
                cofactor_level = cofactor_status.get(cofactor, 0.5)
                cofactor_details[cofactor] = cofactor_level

                if cofactor_level < 0.5:
                    cofactor_modifier *= cofactor_level * 2  # Linear penalty
                    if rate_limiting_factor is None:
                        rate_limiting_factor = f"low_{cofactor}"

            # Apply cofactor modifier
            final_activity = base_activity * cofactor_modifier

            return PathwayActivity(
                pathway=pathway,
                activity_level=min(1.0, final_activity),
                rate_limiting_factor=rate_limiting_factor,
                cofactor_status=cofactor_details,
                product_accumulation=0.0,
            )

        # Default for unmapped pathways
        return PathwayActivity(
            pathway=pathway,
            activity_level=0.5,
            rate_limiting_factor=None,
            cofactor_status=cofactor_status,
        )

    def predict_neurotransmitter_synthesis(
        self,
        plasma_concentrations: Dict[str, PlasmaConcentration],
        cofactor_status: Dict[str, float],
    ) -> List[NeurotransmitterSynthesis]:
        """
        Predict neurotransmitter synthesis rates from precursor availability.
        """
        predictions = []

        # Serotonin from tryptophan
        if "tryptophan" in plasma_concentrations:
            trp = plasma_concentrations["tryptophan"]
            pathway_activity = self.calculate_pathway_activity(
                MetabolicPathway.TRYPTOPHAN_SEROTONIN,
                trp.concentration_umol,
                cofactor_status,
            )

            # Account for brain availability
            effective_precursor = trp.concentration_umol * trp.brain_availability

            # B6 is critical for final decarboxylation step
            b6_status = cofactor_status.get("plp", cofactor_status.get("b6", 0.8))

            synthesis_rate = (
                pathway_activity.activity_level * b6_status * trp.brain_availability
            )

            # Classify brain level
            if synthesis_rate < 0.4:
                brain_level = "low"
            elif synthesis_rate < 0.7:
                brain_level = "normal"
            else:
                brain_level = "elevated"

            predictions.append(
                NeurotransmitterSynthesis(
                    neurotransmitter="serotonin",
                    synthesis_rate=synthesis_rate,
                    precursor_availability=effective_precursor / 100,
                    enzyme_activity=pathway_activity.activity_level,
                    cofactor_adequacy=b6_status,
                    predicted_brain_level=brain_level,
                )
            )

        # Dopamine from tyrosine
        if "tyrosine" in plasma_concentrations:
            tyr = plasma_concentrations["tyrosine"]
            pathway_activity = self.calculate_pathway_activity(
                MetabolicPathway.TYROSINE_CATECHOLAMINE,
                tyr.concentration_umol,
                cofactor_status,
            )

            iron_status = cofactor_status.get("fe2+", cofactor_status.get("iron", 0.8))
            synthesis_rate = (
                pathway_activity.activity_level * iron_status * tyr.brain_availability
            )

            if synthesis_rate < 0.4:
                brain_level = "low"
            elif synthesis_rate < 0.8:
                brain_level = "normal"
            else:
                brain_level = "elevated"

            predictions.append(
                NeurotransmitterSynthesis(
                    neurotransmitter="dopamine",
                    synthesis_rate=synthesis_rate,
                    precursor_availability=tyr.concentration_umol / 100,
                    enzyme_activity=pathway_activity.activity_level,
                    cofactor_adequacy=iron_status,
                    predicted_brain_level=brain_level,
                )
            )

        # GABA from glutamate/glutamine
        if "glutamine" in plasma_concentrations:
            gln = plasma_concentrations["glutamine"]
            pathway_activity = self.calculate_pathway_activity(
                MetabolicPathway.GLUTAMATE_GABA,
                gln.concentration_umol * 0.8,  # Conversion to glutamate
                cofactor_status,
            )

            b6_status = cofactor_status.get("plp", cofactor_status.get("b6", 0.8))
            # B6 is CRITICAL for GAD enzyme
            synthesis_rate = pathway_activity.activity_level * (b6_status**1.5)

            if synthesis_rate < 0.3:
                brain_level = "low"
            elif synthesis_rate < 0.7:
                brain_level = "normal"
            else:
                brain_level = "elevated"

            predictions.append(
                NeurotransmitterSynthesis(
                    neurotransmitter="gaba",
                    synthesis_rate=synthesis_rate,
                    precursor_availability=gln.concentration_umol / 500,
                    enzyme_activity=pathway_activity.activity_level,
                    cofactor_adequacy=b6_status,
                    predicted_brain_level=brain_level,
                )
            )

        return predictions


# =============================================================================
# DEEP LEARNING MODEL
# =============================================================================


class AminoAcidHealthPredictor(nn.Module):
    """
    Neural network for predicting health outcomes from amino acid profiles.

    Architecture:
    - Input: 20+ amino acid concentrations + cofactors + temporal features
    - Multi-head attention for pathway interactions
    - Residual connections for stable training
    - Multiple output heads for different health outcomes
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dims: Dict[str, int] = None,
    ):
        super().__init__()

        if output_dims is None:
            output_dims = {
                "hrv_change": 1,
                "mood_change": 1,
                "energy_change": 1,
                "sleep_quality": 1,
                "cognitive_performance": 1,
            }

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Pathway interaction attention
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # Processing layers with residual connections
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Output heads for different predictions
        self.output_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, dim),
                )
                for name, dim in output_dims.items()
            }
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, len(output_dims)),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            predictions: Dict of output name to prediction tensor
            uncertainty: Uncertainty estimates for each prediction
        """
        # Input projection
        h = self.input_proj(x)

        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # [batch, 1, hidden]

        # Self-attention for feature interactions
        attn_out, attn_weights = self.pathway_attention(h, h, h)
        h = h + attn_out  # Residual

        # Remove sequence dimension
        h = h.squeeze(1)

        # Processing layers with residuals
        for layer in self.layers:
            h = h + layer(h)

        # Generate predictions from each head
        predictions = {name: head(h) for name, head in self.output_heads.items()}

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(h)

        if return_attention:
            return predictions, uncertainty, attn_weights

        return predictions, uncertainty


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================


class AminoAcidMetabolismTracker:
    """
    Comprehensive amino acid metabolism tracking and health prediction.

    Features:
    - Real-time plasma concentration modeling
    - Metabolic pathway activity tracking
    - Neurotransmitter synthesis prediction
    - Health outcome forecasting
    - Personalized recommendations
    """

    def __init__(self, use_neural_model: bool = True):
        self.pk_model = AminoAcidPharmacokinetics()
        self.pathway_model = MetabolicPathwayModel()
        self.intake_history: deque = deque(maxlen=1000)
        self.metabolic_history: deque = deque(maxlen=500)

        # Neural model for predictions
        self.use_neural_model = use_neural_model
        if use_neural_model:
            self.neural_model = AminoAcidHealthPredictor()
            self.neural_model.eval()

        # Default cofactor status (can be updated)
        self.cofactor_status = {
            "plp": 0.8,  # Vitamin B6
            "bh4": 0.85,  # Tetrahydrobiopterin
            "fe2+": 0.8,  # Iron
            "o2": 1.0,  # Oxygen
            "cu2+": 0.9,  # Copper
            "mg2+": 0.75,  # Magnesium
        }

    def log_intake(self, intake: AminoAcidIntake) -> None:
        """Log an amino acid intake event."""
        self.intake_history.append(intake)

    def log_food_intake(
        self,
        food_name: str,
        amount_g: float,
        amino_acid_content: Dict[str, float],  # mg per 100g
        timestamp: Optional[datetime] = None,
        with_carbs: bool = True,
        fasted: bool = False,
    ) -> None:
        """
        Log food intake and create amino acid intake events.

        Args:
            food_name: Name of food
            amount_g: Amount consumed in grams
            amino_acid_content: Dict of amino acid name to mg per 100g
            timestamp: When consumed (default: now)
            with_carbs: Whether consumed with carbohydrates
            fasted: Whether in fasted state
        """
        if timestamp is None:
            timestamp = datetime.now()

        for aa_name, content_per_100g in amino_acid_content.items():
            amount_mg = content_per_100g * (amount_g / 100)

            if amount_mg > 0:
                intake = AminoAcidIntake(
                    amino_acid=aa_name.lower(),
                    amount_mg=amount_mg,
                    timestamp=timestamp,
                    source=f"food:{food_name}",
                    with_carbs=with_carbs,
                    fasted=fasted,
                )
                self.log_intake(intake)

    def update_cofactor_status(self, cofactors: Dict[str, float]) -> None:
        """Update cofactor status (0-1 scale for each)."""
        self.cofactor_status.update(cofactors)

    def get_current_state(self, timestamp: Optional[datetime] = None) -> MetabolicState:
        """
        Get complete current metabolic state.

        Returns comprehensive snapshot including:
        - Plasma amino acid concentrations
        - Pathway activities
        - Neurotransmitter estimates
        - Health predictions
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate plasma concentrations
        intakes = list(self.intake_history)
        plasma = self.pk_model.calculate_plasma_concentration(intakes, timestamp)

        # Get pathway activities
        pathway_activities = {}
        for pathway in MetabolicPathway:
            # Map pathway to primary substrate
            substrate_map = {
                MetabolicPathway.TRYPTOPHAN_SEROTONIN: "tryptophan",
                MetabolicPathway.TYROSINE_CATECHOLAMINE: "tyrosine",
                MetabolicPathway.GLUTAMATE_GABA: "glutamine",
                MetabolicPathway.GLYCINE_GLUTATHIONE: "glycine",
                MetabolicPathway.BCAA_MUSCLE: "leucine",
            }

            substrate = substrate_map.get(pathway)
            if substrate and substrate in plasma:
                activity = self.pathway_model.calculate_pathway_activity(
                    pathway, plasma[substrate].concentration_umol, self.cofactor_status
                )
                pathway_activities[pathway.value] = activity.activity_level
            else:
                pathway_activities[pathway.value] = 0.5  # Default

        # Predict neurotransmitter synthesis
        nt_predictions = self.pathway_model.predict_neurotransmitter_synthesis(
            plasma, self.cofactor_status
        )
        neurotransmitter_estimates = {
            nt.neurotransmitter: nt.synthesis_rate for nt in nt_predictions
        }

        # Health predictions
        hrv_pred, mood_pred, energy_pred, sleep_pred = self._predict_health_outcomes(
            plasma, pathway_activities, neurotransmitter_estimates
        )

        state = MetabolicState(
            timestamp=timestamp,
            plasma_amino_acids={
                aa: conc.concentration_umol for aa, conc in plasma.items()
            },
            pathway_activities=pathway_activities,
            neurotransmitter_estimates=neurotransmitter_estimates,
            hrv_prediction=hrv_pred,
            mood_prediction=mood_pred,
            energy_prediction=energy_pred,
            sleep_prediction=sleep_pred,
        )

        self.metabolic_history.append(state)
        return state

    def _predict_health_outcomes(
        self,
        plasma: Dict[str, PlasmaConcentration],
        pathway_activities: Dict[str, float],
        neurotransmitters: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        """
        Predict health outcomes from current metabolic state.

        Returns: (hrv_change, mood_change, energy_change, sleep_change)
        All values are -1 to 1 relative changes.
        """
        # Use research-based correlations
        correlations = HEALTH_CORRELATIONS

        # HRV prediction
        hrv_change = 0.0
        if "tryptophan" in plasma:
            trp_effect = correlations["tryptophan_serotonin"]["hrv_hf_power"]
            hrv_change += trp_effect * (plasma["tryptophan"].brain_availability - 0.5)

        if "glycine" in plasma:
            gly_effect = correlations["glycine_cardiovascular"]["hrv_sdnn"]
            hrv_change += gly_effect * min(
                1, plasma["glycine"].concentration_umol / 500
            )

        # BCAA recovery effect (post-exercise)
        bcaa_level = sum(
            plasma.get(
                aa, PlasmaConcentration(aa, 0, datetime.now(), 0)
            ).concentration_umol
            for aa in ["leucine", "isoleucine", "valine"]
        )
        if bcaa_level > 500:
            hrv_change += correlations["bcaa_recovery"]["recovery_hrv"] * 0.5

        # Mood prediction
        mood_change = 0.0
        serotonin = neurotransmitters.get("serotonin", 0.5)
        dopamine = neurotransmitters.get("dopamine", 0.5)

        mood_change += correlations["tryptophan_serotonin"]["mood_score"] * (
            serotonin - 0.5
        )
        mood_change += correlations["tyrosine_dopamine"]["motivation"] * (
            dopamine - 0.5
        )

        # Energy prediction
        energy_change = 0.0
        energy_change += correlations["tyrosine_dopamine"]["cognitive_performance"] * (
            dopamine - 0.5
        )
        if bcaa_level > 300:
            energy_change += 0.1  # BCAAs support energy

        # Sleep prediction
        sleep_change = 0.0
        sleep_change += correlations["tryptophan_serotonin"]["sleep_quality"] * (
            serotonin - 0.5
        )
        gaba = neurotransmitters.get("gaba", 0.5)
        sleep_change += 0.3 * (gaba - 0.5)  # GABA promotes sleep

        # Clamp all predictions
        return (
            max(-1, min(1, hrv_change)),
            max(-1, min(1, mood_change)),
            max(-1, min(1, energy_change)),
            max(-1, min(1, sleep_change)),
        )

    def predict_outcome_timeline(
        self, hours_ahead: int = 12, resolution_minutes: int = 30
    ) -> List[MetabolicState]:
        """
        Predict metabolic state timeline for upcoming hours.

        Args:
            hours_ahead: How many hours to predict
            resolution_minutes: Time resolution in minutes

        Returns:
            List of predicted metabolic states
        """
        timeline = []
        current = datetime.now()

        for minutes in range(0, hours_ahead * 60, resolution_minutes):
            future_time = current + timedelta(minutes=minutes)
            state = self.get_current_state(future_time)
            timeline.append(state)

        return timeline

    def get_optimization_recommendations(
        self, target: str = "balanced"  # "hrv", "mood", "energy", "sleep", "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations to optimize amino acid intake.

        Args:
            target: Optimization target

        Returns:
            List of actionable recommendations
        """
        current_state = self.get_current_state()
        recommendations = []

        # Analyze current deficiencies
        plasma = current_state.plasma_amino_acids

        # Tryptophan for serotonin/sleep
        if target in ["sleep", "mood", "balanced"]:
            trp_level = plasma.get("tryptophan", 0)
            if trp_level < 50:
                recommendations.append(
                    {
                        "priority": 1,
                        "amino_acid": "tryptophan",
                        "current_level_umol": trp_level,
                        "target_level_umol": 80,
                        "recommendation": "Increase tryptophan intake",
                        "food_sources": ["turkey", "chicken", "eggs", "cheese", "nuts"],
                        "timing": "2-3 hours before bed for sleep, with carbs to reduce BCAA competition",
                        "mechanism": "Tryptophan → 5-HTP → Serotonin → Melatonin pathway",
                        "evidence": "Tryptophan depletion reduces HF-HRV (Biol Psychiatry 2006)",
                    }
                )

        # Tyrosine for energy/cognition
        if target in ["energy", "cognitive", "balanced"]:
            tyr_level = plasma.get("tyrosine", 0)
            if tyr_level < 60:
                recommendations.append(
                    {
                        "priority": 2,
                        "amino_acid": "tyrosine",
                        "current_level_umol": tyr_level,
                        "target_level_umol": 100,
                        "recommendation": "Increase tyrosine intake",
                        "food_sources": [
                            "chicken",
                            "fish",
                            "dairy",
                            "almonds",
                            "avocado",
                        ],
                        "timing": "Morning or before demanding tasks",
                        "mechanism": "Tyrosine → L-DOPA → Dopamine → Norepinephrine",
                        "evidence": "Tyrosine improves cognitive performance under stress",
                    }
                )

        # Glycine for cardiovascular/sleep
        if target in ["hrv", "sleep", "balanced"]:
            gly_level = plasma.get("glycine", 0)
            if gly_level < 200:
                recommendations.append(
                    {
                        "priority": 2,
                        "amino_acid": "glycine",
                        "current_level_umol": gly_level,
                        "target_level_umol": 400,
                        "recommendation": "Increase glycine intake (3g before bed)",
                        "food_sources": [
                            "bone broth",
                            "gelatin",
                            "collagen",
                            "pork skin",
                        ],
                        "timing": "Before bed - improves sleep quality",
                        "mechanism": "Glycine is inhibitory neurotransmitter, reduces core body temp",
                        "evidence": "MDPI 2024 narrative review on glycine cardiovascular benefits",
                    }
                )

        # BCAAs for recovery
        if target in ["recovery", "balanced"]:
            bcaa_total = sum(
                plasma.get(aa, 0) for aa in ["leucine", "isoleucine", "valine"]
            )
            if bcaa_total < 300:
                recommendations.append(
                    {
                        "priority": 3,
                        "amino_acid": "bcaa",
                        "current_level_umol": bcaa_total,
                        "target_level_umol": 600,
                        "recommendation": "Add BCAAs around exercise (2:1:1 ratio)",
                        "food_sources": ["whey protein", "eggs", "chicken", "beef"],
                        "timing": "30min pre-workout and immediately post-workout",
                        "mechanism": "Reduces muscle damage markers (CK), DOMS",
                        "evidence": "Sports Med Open 2024: BCAA reduces CK at 72h (g=-0.99)",
                    }
                )

        # Cofactor recommendations
        if self.cofactor_status.get("plp", 1) < 0.7:
            recommendations.append(
                {
                    "priority": 1,
                    "type": "cofactor",
                    "nutrient": "vitamin_b6",
                    "recommendation": "Increase B6 intake - critical for neurotransmitter synthesis",
                    "food_sources": ["chicken", "fish", "potatoes", "bananas"],
                    "mechanism": "B6 (PLP) is cofactor for aromatic amino acid decarboxylase",
                    "evidence": "B6 deficiency impairs serotonin and GABA synthesis",
                }
            )

        return sorted(recommendations, key=lambda x: x.get("priority", 99))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_tracker() -> AminoAcidMetabolismTracker:
    """Create and return a tracker instance."""
    return AminoAcidMetabolismTracker()


def quick_analysis(foods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quick amino acid analysis from food list.

    Args:
        foods: List of {name, amount_g, amino_acids: {aa: mg_per_100g}}

    Returns:
        Analysis summary
    """
    tracker = create_tracker()

    for food in foods:
        tracker.log_food_intake(
            food_name=food.get("name", "unknown"),
            amount_g=food.get("amount_g", 100),
            amino_acid_content=food.get("amino_acids", {}),
        )

    state = tracker.get_current_state()
    recommendations = tracker.get_optimization_recommendations()

    return {
        "plasma_levels": state.plasma_amino_acids,
        "pathway_activities": state.pathway_activities,
        "neurotransmitter_status": state.neurotransmitter_estimates,
        "predictions": {
            "hrv_change": state.hrv_prediction,
            "mood_change": state.mood_prediction,
            "energy_change": state.energy_prediction,
            "sleep_change": state.sleep_prediction,
        },
        "recommendations": recommendations,
    }
