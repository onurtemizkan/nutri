"""
Business logic services for ML operations.

This module provides:
- Core ML services (feature engineering, correlations)
- Food sensitivity detection (ingredient extraction, compound quantification)
- Optimized sensitivity pipeline with advanced components
- Advanced nutritional biomarker engines (v3)

Optimized Components (v2):
- AdvancedIngredientMatcher: Trie + BK-Tree for O(log n) fuzzy matching
- BayesianSensitivityEngine: Probabilistic inference with Beta-Binomial model
- SensitivityCacheService: Multi-tier Redis caching with Bloom filters
- NLPIngredientExtractor: spaCy-based NER for ingredient extraction
- LSTMTemporalPatternAnalyzer: BiLSTM + Attention for temporal patterns
- OptimizedSensitivityPipeline: Unified pipeline integrating all components

Nutritional Biomarker Engines (v3):
- NutritionalBiomarkerEngine: Comprehensive HRV prediction from nutrition
- AminoAcidMetabolismTracker: Pharmacokinetic amino acid modeling
- DietaryInflammatoryIndexCalculator: DII scoring from 45 parameters
- GlycemicResponseCalculator: GL/GI prediction with curve modeling
- GutBrainVagalEngine: Microbiome-vagus nerve axis analysis

Research basis:
- Nutrition & Metabolism 2024: Tryptophan-CVD meta-analysis
- Frontiers 2025: Blood metabolome-HRV correlations
- Gut Microbes 2025: Probiotic-HRV clinical trial
- Sports Medicine Open 2024: BCAA meta-analysis
- Atherosclerosis 2020: DII meta-analysis (15 cohort studies)
"""

from .feature_engineering import FeatureEngineeringService
from .correlation_engine import CorrelationEngineService
from .ingredient_extraction_service import (
    IngredientExtractionService,
    ingredient_extraction_service,
)
from .compound_quantification_service import (
    CompoundQuantificationService,
    compound_quantification_service,
)
from .hrv_sensitivity_analyzer import (
    HRVSensitivityAnalyzer,
    hrv_sensitivity_analyzer,
)
from .sensitivity_ml_model import (
    SensitivityMLModel,
    sensitivity_ml_model,
)

# Optimized components (v2)
from .advanced_ingredient_matcher import (
    AdvancedIngredientMatcher,
    MatchResult,
)
from .bayesian_sensitivity_engine import (
    BayesianSensitivityEngine,
    BayesianBelief,
    SensitivityInference,
    ExposureEvidence,
)
from .sensitivity_cache_service import (
    SensitivityCacheService,
    get_cache_service,
    cached,
)
from .nlp_ingredient_extractor import (
    NLPIngredientExtractor,
    ExtractedIngredient,
    ExtractionResult,
    get_nlp_extractor,
    extract_ingredients,
)
from .lstm_temporal_analyzer import (
    LSTMTemporalPatternAnalyzer,
    TemporalPattern,
    PatternType,
    get_lstm_analyzer,
    analyze_hrv_patterns,
)
from .optimized_sensitivity_pipeline import (
    OptimizedSensitivityPipeline,
    SensitivityAnalysisResult,
    AnalysisMode,
    get_pipeline,
    analyze_food_sensitivity,
)

# Nutritional Biomarker Engines (v3)
from .nutritional_biomarkers import (
    NutritionalBiomarkerEngine,
    AminoAcidProfile,
    MicronutrientProfile,
    InflammatoryProfile,
    BiogenicAmineProfile,
    GlycemicProfile,
    GutBrainProfile,
    NeurotransmitterPrediction,
    HRVPrediction,
    create_engine as create_biomarker_engine,
    quick_hrv_assessment,
)
from .amino_acid_metabolism import (
    AminoAcidMetabolismTracker,
    AminoAcidIntake,
    MetabolicState,
    NeurotransmitterSynthesis,
    AminoAcidPharmacokinetics,
    MetabolicPathwayModel,
    AminoAcidHealthPredictor,
    create_tracker as create_amino_tracker,
    quick_analysis as quick_amino_analysis,
)
from .inflammatory_glycemic_engine import (
    DietaryInflammatoryIndexCalculator,
    GlycemicResponseCalculator,
    CombinedInflammatoryGlycemicEngine,
    DIICalculation,
    GlycemicMeal,
    GlycemicDay,
    CombinedHealthPrediction,
    InflammatoryResponsePredictor,
    GlycemicResponsePredictor,
    calculate_dii,
    analyze_meal_glycemic,
    full_dietary_analysis,
)
from .gut_brain_vagal_engine import (
    GutBrainVagalEngine,
    MicrobiomeProfile,
    PrebioticIntake,
    ProbioticIntake,
    SCFAProduction,
    VagalTonePrediction,
    GutBrainAxisState,
    ProbioticProtocol,
    GutBrainPredictor,
    TemporalMicrobiomeModel,
    analyze_gut_brain_health,
    get_hrv_probiotic_protocol,
)

__all__ = [
    # Core services
    "FeatureEngineeringService",
    "CorrelationEngineService",
    # Original sensitivity detection services
    "IngredientExtractionService",
    "ingredient_extraction_service",
    "CompoundQuantificationService",
    "compound_quantification_service",
    "HRVSensitivityAnalyzer",
    "hrv_sensitivity_analyzer",
    "SensitivityMLModel",
    "sensitivity_ml_model",
    # Optimized components (v2)
    "AdvancedIngredientMatcher",
    "MatchResult",
    "BayesianSensitivityEngine",
    "BayesianBelief",
    "SensitivityInference",
    "ExposureEvidence",
    "SensitivityCacheService",
    "get_cache_service",
    "cached",
    "NLPIngredientExtractor",
    "ExtractedIngredient",
    "ExtractionResult",
    "get_nlp_extractor",
    "extract_ingredients",
    "LSTMTemporalPatternAnalyzer",
    "TemporalPattern",
    "PatternType",
    "get_lstm_analyzer",
    "analyze_hrv_patterns",
    "OptimizedSensitivityPipeline",
    "SensitivityAnalysisResult",
    "AnalysisMode",
    "get_pipeline",
    "analyze_food_sensitivity",
    # Nutritional Biomarker Engines (v3)
    "NutritionalBiomarkerEngine",
    "AminoAcidProfile",
    "MicronutrientProfile",
    "InflammatoryProfile",
    "BiogenicAmineProfile",
    "GlycemicProfile",
    "GutBrainProfile",
    "NeurotransmitterPrediction",
    "HRVPrediction",
    "create_biomarker_engine",
    "quick_hrv_assessment",
    # Amino acid metabolism
    "AminoAcidMetabolismTracker",
    "AminoAcidIntake",
    "MetabolicState",
    "NeurotransmitterSynthesis",
    "AminoAcidPharmacokinetics",
    "MetabolicPathwayModel",
    "AminoAcidHealthPredictor",
    "create_amino_tracker",
    "quick_amino_analysis",
    # Inflammatory and glycemic engines
    "DietaryInflammatoryIndexCalculator",
    "GlycemicResponseCalculator",
    "CombinedInflammatoryGlycemicEngine",
    "DIICalculation",
    "GlycemicMeal",
    "GlycemicDay",
    "CombinedHealthPrediction",
    "InflammatoryResponsePredictor",
    "GlycemicResponsePredictor",
    "calculate_dii",
    "analyze_meal_glycemic",
    "full_dietary_analysis",
    # Gut-brain-vagal engine
    "GutBrainVagalEngine",
    "MicrobiomeProfile",
    "PrebioticIntake",
    "ProbioticIntake",
    "SCFAProduction",
    "VagalTonePrediction",
    "GutBrainAxisState",
    "ProbioticProtocol",
    "GutBrainPredictor",
    "TemporalMicrobiomeModel",
    "analyze_gut_brain_health",
    "get_hrv_probiotic_protocol",
]
