"""
Pydantic schemas for Food Sensitivity Detection System.

These schemas define the API request/response models for:
- Ingredient extraction and allergen detection
- Sensitivity exposure tracking
- HRV-based sensitivity analysis
- Sensitivity insights and recommendations
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# ENUMS (matching SQLAlchemy models)
# =============================================================================


class AllergenTypeSchema(str, Enum):
    """FDA Big 9 + EU additions"""
    MILK = "milk"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH_CRUSTACEAN = "shellfish_crustacean"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT = "wheat"
    SOY = "soy"
    SESAME = "sesame"
    GLUTEN_CEREALS = "gluten_cereals"
    SHELLFISH_MOLLUSCAN = "shellfish_molluscan"
    MUSTARD = "mustard"
    CELERY = "celery"
    LUPIN = "lupin"
    SULFITES = "sulfites"


class SensitivityTypeSchema(str, Enum):
    """Types of food sensitivities"""
    ALLERGY = "allergy"
    INTOLERANCE = "intolerance"
    SENSITIVITY = "sensitivity"
    FODMAP = "fodmap"
    HISTAMINE = "histamine"
    TYRAMINE = "tyramine"
    SALICYLATE = "salicylate"
    OXALATE = "oxalate"
    LECTIN = "lectin"
    NIGHTSHADE = "nightshade"
    SULFITE = "sulfite"


class SeveritySchema(str, Enum):
    """Severity levels"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    EMERGENCY = "emergency"


class CompoundLevelSchema(str, Enum):
    """Compound concentration levels"""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class DerivationTypeSchema(str, Enum):
    """How an ingredient relates to an allergen"""
    DIRECTLY_CONTAINS = "directly_contains"
    DERIVED_FROM = "derived_from"
    MAY_CONTAIN = "may_contain"
    PROCESSED_WITH = "processed_with"
    FREE_FROM = "free_from"
    LIKELY_CONTAINS = "likely_contains"


# =============================================================================
# INGREDIENT EXTRACTION SCHEMAS
# =============================================================================


class IngredientExtractionRequest(BaseModel):
    """Request to extract ingredients from text"""
    text: str = Field(..., description="Meal name, description, or ingredient list")
    include_hidden_allergens: bool = Field(
        True, description="Check for hidden allergen keywords"
    )
    fuzzy_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Fuzzy matching threshold (0-1)"
    )
    max_results: int = Field(10, ge=1, le=50, description="Maximum ingredients to return")


class ExtractedIngredient(BaseModel):
    """An ingredient extracted from text"""
    matched_text: str = Field(..., description="Original text that matched")
    ingredient_id: Optional[str] = Field(None, description="Database ingredient ID if found")
    ingredient_name: str = Field(..., description="Normalized ingredient name")
    display_name: str = Field(..., description="Human-readable name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence")
    match_type: str = Field(..., description="exact, fuzzy, hidden_keyword, inference")
    category: Optional[str] = Field(None, description="Ingredient category")


class AllergenWarning(BaseModel):
    """Warning about a detected allergen"""
    allergen_type: AllergenTypeSchema
    display_name: str = Field(..., description="Human-readable allergen name")
    derivation: DerivationTypeSchema
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_ingredient: str = Field(..., description="Ingredient containing the allergen")
    is_hidden: bool = Field(False, description="Was this a hidden allergen?")
    warning_level: SeveritySchema = Field(
        SeveritySchema.MODERATE, description="Risk level"
    )


class CompoundWarning(BaseModel):
    """Warning about a bioactive compound"""
    compound_type: str = Field(..., description="histamine, tyramine, fodmap, etc.")
    level: CompoundLevelSchema
    amount_mg: Optional[float] = Field(None, description="Estimated mg per 100g")
    source_ingredient: str
    warning_message: str


class IngredientExtractionResponse(BaseModel):
    """Response from ingredient extraction"""
    success: bool = True
    ingredients: List[ExtractedIngredient] = Field(default_factory=list)
    allergen_warnings: List[AllergenWarning] = Field(default_factory=list)
    compound_warnings: List[CompoundWarning] = Field(default_factory=list)
    total_histamine_mg: Optional[float] = Field(None, description="Total histamine estimate")
    total_tyramine_mg: Optional[float] = Field(None, description="Total tyramine estimate")
    fodmap_summary: Optional[Dict[str, str]] = Field(None, description="FODMAP type levels")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    suggestions: List[str] = Field(default_factory=list, description="Helpful suggestions")


# =============================================================================
# MEAL SENSITIVITY CHECK SCHEMAS
# =============================================================================


class MealSensitivityCheckRequest(BaseModel):
    """Request to check a meal for user's known sensitivities"""
    user_id: str = Field(..., description="User ID")
    meal_text: Optional[str] = Field(None, description="Meal name/description")
    ingredients: Optional[List[str]] = Field(None, description="List of ingredients")
    meal_id: Optional[str] = Field(None, description="Existing meal ID to check")


class SensitivityMatch(BaseModel):
    """A match between meal content and user sensitivity"""
    sensitivity_id: str
    sensitivity_type: SensitivityTypeSchema
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    severity: SeveritySchema
    matched_ingredient: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    historical_reaction_rate: Optional[float] = Field(
        None, description="Past reaction rate 0-1"
    )
    avg_hrv_impact: Optional[float] = Field(
        None, description="Average HRV drop percentage"
    )


class MealSensitivityCheckResponse(BaseModel):
    """Response from meal sensitivity check"""
    success: bool = True
    is_safe: bool = Field(..., description="True if no sensitivity matches found")
    risk_level: SeveritySchema = Field(..., description="Overall risk level")
    sensitivity_matches: List[SensitivityMatch] = Field(default_factory=list)
    allergen_warnings: List[AllergenWarning] = Field(default_factory=list)
    compound_warnings: List[CompoundWarning] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    safe_alternatives: List[str] = Field(
        default_factory=list, description="Suggested safe alternatives"
    )


# =============================================================================
# EXPOSURE TRACKING SCHEMAS
# =============================================================================


class RecordExposureRequest(BaseModel):
    """Request to record a sensitivity exposure event"""
    user_id: str
    meal_id: Optional[str] = None
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    estimated_amount_mg: Optional[float] = None
    exposed_at: datetime = Field(default_factory=datetime.utcnow)

    # Reaction info (can be added later)
    had_reaction: bool = False
    reaction_severity: Optional[SeveritySchema] = None
    symptoms: Optional[List[str]] = None
    onset_minutes: Optional[int] = None
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None


class RecordExposureResponse(BaseModel):
    """Response from recording exposure"""
    success: bool = True
    exposure_id: str
    message: str


class UpdateReactionRequest(BaseModel):
    """Request to update reaction info for an exposure"""
    exposure_id: str
    had_reaction: bool
    reaction_severity: Optional[SeveritySchema] = None
    symptoms: Optional[List[str]] = None
    onset_minutes: Optional[int] = None
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None


# =============================================================================
# HRV SENSITIVITY ANALYSIS SCHEMAS
# =============================================================================


class HRVWindow(BaseModel):
    """HRV data for a specific time window"""
    window_name: str = Field(..., description="immediate, short_term, medium_term, next_day")
    start_minutes: int
    end_minutes: int
    baseline_hrv: float = Field(..., description="RMSSD baseline")
    post_exposure_hrv: float = Field(..., description="RMSSD after exposure")
    hrv_change_ms: float = Field(..., description="Absolute change in ms")
    hrv_change_pct: float = Field(..., description="Percentage change")
    is_significant: bool = Field(..., description="Exceeds threshold")


class HRVSensitivityAnalysisRequest(BaseModel):
    """Request HRV-based sensitivity analysis"""
    user_id: str
    exposure_id: Optional[str] = Field(None, description="Specific exposure to analyze")
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    days_back: int = Field(30, ge=7, le=365, description="Days of history to analyze")
    min_data_points: int = Field(5, ge=3, description="Minimum exposures required")


class HRVSensitivityResult(BaseModel):
    """Result of HRV sensitivity analysis for one trigger"""
    trigger_type: str = Field(..., description="allergen or compound type")
    trigger_name: str = Field(..., description="Human-readable name")
    exposure_count: int
    reaction_count: int
    reaction_rate: float = Field(..., ge=0.0, le=1.0)

    # HRV statistics
    avg_hrv_drop_ms: float
    avg_hrv_drop_pct: float
    max_hrv_drop_pct: float
    hrv_by_window: Dict[str, float] = Field(
        ..., description="Average HRV change per time window"
    )

    # Statistical significance
    correlation_coefficient: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: bool = False
    confidence_level: float = Field(..., ge=0.0, le=1.0)

    # Recommendation
    suggested_severity: SeveritySchema
    recommendation: str


class HRVSensitivityAnalysisResponse(BaseModel):
    """Response from HRV sensitivity analysis"""
    success: bool = True
    user_id: str
    analysis_period_days: int
    total_exposures_analyzed: int

    results: List[HRVSensitivityResult] = Field(default_factory=list)

    # Newly discovered sensitivities
    new_discoveries: List[str] = Field(
        default_factory=list, description="Potential new sensitivities"
    )

    # Overall health pattern
    overall_hrv_trend: str = Field(..., description="improving, stable, declining")
    baseline_hrv: float
    current_hrv: float

    generated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# SENSITIVITY INSIGHTS SCHEMAS
# =============================================================================


class SensitivityInsightSchema(BaseModel):
    """A generated insight about user's food sensitivities"""
    id: str
    insight_type: str = Field(..., description="pattern, correlation, warning, recommendation")
    priority: str = Field(..., description="low, medium, high, critical")
    title: str
    description: str
    recommendation: Optional[str] = None

    # Supporting data
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    correlation_coefficient: Optional[float] = None
    confidence: float
    data_points: int
    p_value: Optional[float] = None
    is_significant: bool = False

    # Chart data (for visualization)
    chart_data: Optional[Dict[str, Any]] = None

    created_at: datetime
    expires_at: Optional[datetime] = None


class GetInsightsRequest(BaseModel):
    """Request to get sensitivity insights"""
    user_id: str
    include_dismissed: bool = False
    include_viewed: bool = True
    priority_filter: Optional[List[str]] = None
    insight_type_filter: Optional[List[str]] = None
    limit: int = Field(20, ge=1, le=100)


class GetInsightsResponse(BaseModel):
    """Response with user's sensitivity insights"""
    success: bool = True
    insights: List[SensitivityInsightSchema] = Field(default_factory=list)
    total_count: int
    unread_count: int


class MarkInsightRequest(BaseModel):
    """Request to mark insight as viewed/dismissed/helpful"""
    insight_id: str
    action: str = Field(..., description="view, dismiss, helpful, not_helpful")


# =============================================================================
# USER SENSITIVITY MANAGEMENT SCHEMAS
# =============================================================================


class AddUserSensitivityRequest(BaseModel):
    """Request to add a user sensitivity"""
    user_id: str
    sensitivity_type: SensitivityTypeSchema
    severity: SeveritySchema
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    confirmed_by_test: bool = False
    notes: Optional[str] = None


class UserSensitivitySchema(BaseModel):
    """User sensitivity record"""
    id: str
    user_id: str
    sensitivity_type: SensitivityTypeSchema
    severity: SeveritySchema
    allergen_type: Optional[AllergenTypeSchema] = None
    compound_type: Optional[str] = None
    confirmed_by_test: bool
    notes: Optional[str] = None
    active: bool

    # Correlation data
    avg_hrv_drop: Optional[float] = None
    correlation_score: Optional[float] = None
    exposure_count: int = 0
    reaction_count: int = 0
    reaction_rate: Optional[float] = None

    created_at: datetime
    updated_at: datetime


class GetUserSensitivitiesResponse(BaseModel):
    """Response with user's sensitivities"""
    success: bool = True
    sensitivities: List[UserSensitivitySchema] = Field(default_factory=list)
    total_count: int
