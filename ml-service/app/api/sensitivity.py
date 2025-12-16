"""
Food Sensitivity Detection API Endpoints

Comprehensive API for:
- Ingredient extraction and allergen detection
- Meal sensitivity checking against user profiles
- Compound quantification (histamine, tyramine, FODMAP)
- HRV-based sensitivity analysis
- Exposure tracking and reaction recording
- ML-powered sensitivity prediction
"""
# mypy: ignore-errors

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from app.schemas.sensitivity import (
    IngredientExtractionRequest,
    IngredientExtractionResponse,
    MealSensitivityCheckRequest,
    MealSensitivityCheckResponse,
    RecordExposureRequest,
    RecordExposureResponse,
    UpdateReactionRequest,
    HRVSensitivityAnalysisRequest,
    AddUserSensitivityRequest,
    GetUserSensitivitiesResponse,
    SeveritySchema,
)

from app.services.ingredient_extraction_service import (
    ingredient_extraction_service,
)
from app.services.compound_quantification_service import (
    compound_quantification_service,
)
from app.services.hrv_sensitivity_analyzer import (
    hrv_sensitivity_analyzer,
)
from app.services.sensitivity_ml_model import (
    sensitivity_ml_model,
    TrainingDataPoint,
)
from app.data.allergen_database import (
    INGREDIENT_DATABASE,
    AllergenType,
    CROSS_REACTIVITY,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# INGREDIENT EXTRACTION ENDPOINTS
# =============================================================================


@router.post("/extract-ingredients", response_model=IngredientExtractionResponse)
async def extract_ingredients(
    request: IngredientExtractionRequest = Body(...),
):
    """
    Extract ingredients from meal text and detect allergens/compounds.

    This endpoint parses meal names, descriptions, or ingredient lists to:
    - Identify individual ingredients using fuzzy matching
    - Detect allergens (FDA Big 9 + EU Big 14)
    - Flag hidden allergens in ingredient names
    - Quantify compounds (histamine, tyramine, FODMAP levels)

    **Example request:**
    ```json
    {
        "text": "Grilled salmon with aged parmesan and spinach",
        "include_hidden_allergens": true,
        "fuzzy_threshold": 0.75
    }
    ```

    **Returns:**
    - List of extracted ingredients with confidence scores
    - Allergen warnings with severity levels
    - Compound warnings for histamine/tyramine/FODMAP
    """
    try:
        response = await ingredient_extraction_service.extract_ingredients(request)
        return response
    except Exception as e:
        logger.error(f"Ingredient extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingredients/search")
async def search_ingredients(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Maximum results"),
):
    """
    Search the ingredient database by name.

    Useful for autocomplete and ingredient lookup.

    **Parameters:**
    - **q**: Search query (minimum 2 characters)
    - **limit**: Maximum number of results (default 20)

    **Returns:**
    - List of matching ingredients with allergen and compound info
    """
    try:
        results = ingredient_extraction_service.search_ingredients(q, limit)
        return {
            "query": q,
            "results": results,
            "total": len(results),
        }
    except Exception as e:
        logger.error(f"Ingredient search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/allergens")
async def list_allergens():
    """
    Get list of all supported allergen types.

    Returns FDA Big 9 and EU Big 14 allergens with display names.
    """
    allergens = ingredient_extraction_service.get_all_allergen_types()
    return {
        "allergens": allergens,
        "total": len(allergens),
        "note": "Includes FDA Big 9 (USA) and EU Big 14 allergens",
    }


@router.get("/allergens/{allergen_type}/cross-reactivity")
async def get_cross_reactivity(allergen_type: str):
    """
    Get cross-reactive allergens for a specific allergen.

    Cross-reactivity means the immune system may react to multiple
    allergens due to similar protein structures.

    **Example:** Tree nuts cross-react with peanuts in some individuals.
    """
    try:
        allergen = AllergenType(allergen_type)
        cross_reactive = ingredient_extraction_service.get_cross_reactivity(allergen)
        return {
            "allergen": allergen_type,
            "cross_reactive": cross_reactive,
            "count": len(cross_reactive),
        }
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Unknown allergen type: {allergen_type}"
        )


# =============================================================================
# COMPOUND QUANTIFICATION ENDPOINTS
# =============================================================================


@router.post("/compounds/quantify")
async def quantify_meal_compounds(
    ingredients: List[str] = Body(..., description="List of ingredient names"),
    portion_weights: Optional[dict] = Body(
        None, description="Ingredient weights in grams"
    ),
    user_profile: Optional[dict] = Body(None, description="User sensitivity profile"),
):
    """
    Quantify bioactive compounds in a meal.

    Calculates total histamine, tyramine, and FODMAP load with
    risk assessment based on clinical thresholds.

    **User profile options:**
    ```json
    {
        "is_maoi_user": false,
        "is_histamine_sensitive": true,
        "is_migraine_prone": false,
        "sensitivities": ["fodmap", "histamine"]
    }
    ```

    **Returns:**
    - Compound levels (mg) with risk assessment
    - Threshold comparisons for user profile
    - Interaction warnings
    - Timing recommendations
    """
    try:
        profile = user_profile or {}

        result = compound_quantification_service.quantify_meal_compounds(
            ingredients=ingredients,
            portion_weights=portion_weights,
            user_sensitivities=profile.get("sensitivities"),
            is_maoi_user=profile.get("is_maoi_user", False),
            is_histamine_sensitive=profile.get("is_histamine_sensitive", False),
            is_migraine_prone=profile.get("is_migraine_prone", False),
        )

        # Convert to JSON-serializable format
        return {
            "histamine": {
                "total_mg": result.histamine.total_mg,
                "level": result.histamine.level.value,
                "risk_level": result.histamine.risk_level.value,
                "threshold": result.histamine.threshold_value,
                "percentage_of_threshold": result.histamine.percentage_of_threshold,
                "sources": result.histamine.sources,
                "warnings": result.histamine.warnings,
                "recommendations": result.histamine.recommendations,
            },
            "tyramine": {
                "total_mg": result.tyramine.total_mg,
                "level": result.tyramine.level.value,
                "risk_level": result.tyramine.risk_level.value,
                "threshold": result.tyramine.threshold_value,
                "percentage_of_threshold": result.tyramine.percentage_of_threshold,
                "sources": result.tyramine.sources,
                "warnings": result.tyramine.warnings,
                "recommendations": result.tyramine.recommendations,
            },
            "fodmap": {
                "total_types": result.fodmap.total_fodmap_types,
                "high_count": result.fodmap.high_fodmap_count,
                "risk_level": result.fodmap.risk_level.value,
                "by_type": result.fodmap.by_type,
                "stacking_warning": result.fodmap.stacking_warning,
                "warnings": result.fodmap.warnings,
                "recommendations": result.fodmap.recommendations,
            },
            "overall_risk": result.overall_risk.value,
            "interaction_warnings": result.interaction_warnings,
            "timing_recommendation": result.meal_timing_recommendation,
        }

    except Exception as e:
        logger.error(f"Compound quantification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compounds/thresholds")
async def get_compound_thresholds():
    """
    Get clinical thresholds for all tracked compounds.

    Returns safe limits for:
    - Histamine (general and sensitive)
    - Tyramine (general, migraine, MAOI)
    - FODMAP (per type)
    """
    thresholds = compound_quantification_service.get_compound_thresholds()
    return {
        "thresholds": thresholds,
        "notes": {
            "histamine": "Values in mg per meal. Sensitive individuals may react to <10mg.",
            "tyramine": "MAOI users must stay below 6mg per meal (hypertensive crisis risk).",
            "fodmap": "Values in grams. Stacking multiple types increases reaction risk.",
        },
    }


@router.get("/compounds/dao-inhibitors")
async def get_dao_inhibitors():
    """
    Get list of foods that inhibit DAO enzyme.

    DAO (diamine oxidase) breaks down histamine. These foods
    slow histamine metabolism, increasing reaction risk.
    """
    inhibitors = compound_quantification_service.get_dao_inhibitors()
    return {
        "dao_inhibitors": inhibitors,
        "count": len(inhibitors),
        "warning": "Consuming these with high-histamine foods increases reaction risk",
    }


@router.get("/compounds/histamine-liberators")
async def get_histamine_liberators():
    """
    Get list of histamine liberator foods.

    These foods trigger the body to release histamine,
    even if they don't contain much histamine themselves.
    """
    liberators = compound_quantification_service.get_histamine_liberators()
    return {
        "histamine_liberators": liberators,
        "count": len(liberators),
        "warning": "May trigger histamine symptoms even in low-histamine diets",
    }


# =============================================================================
# MEAL SENSITIVITY CHECK ENDPOINTS
# =============================================================================


@router.post("/check-meal", response_model=MealSensitivityCheckResponse)
async def check_meal_sensitivity(
    request: MealSensitivityCheckRequest = Body(...),
):
    """
    Check a meal against user's known sensitivities.

    Combines ingredient extraction, allergen detection, and
    user sensitivity profile to assess meal safety.

    **Request:**
    ```json
    {
        "user_id": "user123",
        "meal_text": "Caesar salad with parmesan",
        "ingredients": ["romaine", "parmesan", "anchovy"]
    }
    ```

    **Returns:**
    - Safety assessment (is_safe)
    - Risk level
    - Matched sensitivities with historical reaction data
    - Recommendations and safe alternatives
    """
    try:
        # Extract ingredients from text if provided
        ingredients = request.ingredients or []

        if request.meal_text:
            extraction = await ingredient_extraction_service.extract_ingredients(
                IngredientExtractionRequest(text=request.meal_text)
            )
            extracted_names = [i.ingredient_name for i in extraction.ingredients]
            ingredients = list(set(ingredients + extracted_names))

        if not ingredients:
            raise HTTPException(
                status_code=400, detail="Either meal_text or ingredients list required"
            )

        # TODO: Fetch user sensitivities from database
        # For now, return a mock response demonstrating the structure
        # In production, this would query UserSensitivity table

        # Extract again for allergens
        extraction = await ingredient_extraction_service.extract_ingredients(
            IngredientExtractionRequest(text=" ".join(ingredients))
        )

        # Build response
        is_safe = len(extraction.allergen_warnings) == 0
        risk_level = SeveritySchema.NONE

        if extraction.allergen_warnings:
            max_severity = max(w.warning_level for w in extraction.allergen_warnings)
            risk_level = max_severity
            is_safe = False

        recommendations = []
        if not is_safe:
            recommendations.append("This meal contains potential triggers")
            recommendations.extend(extraction.suggestions)

        return MealSensitivityCheckResponse(
            success=True,
            is_safe=is_safe,
            risk_level=risk_level,
            sensitivity_matches=[],  # Would be populated from user sensitivity lookup
            allergen_warnings=extraction.allergen_warnings,
            compound_warnings=extraction.compound_warnings,
            recommendations=recommendations,
            safe_alternatives=[],  # Would suggest alternatives
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meal sensitivity check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HRV SENSITIVITY ANALYSIS ENDPOINTS
# =============================================================================


@router.post("/hrv/analyze")
async def analyze_hrv_sensitivity(
    request: HRVSensitivityAnalysisRequest = Body(...),
):
    """
    Analyze HRV data to detect food sensitivity patterns.

    Uses multi-window temporal analysis to correlate food exposures
    with HRV changes. Requires historical HRV and exposure data.

    **Analysis windows:**
    - Immediate: 0-30 minutes (acute reactions)
    - Short-term: 30min-2hr (IgE allergies peak)
    - Medium-term: 2-6hr (intolerances, FODMAP)
    - Extended: 6-24hr (cumulative effects)
    - Next-day: 24-48hr (delayed reactions)

    **Returns:**
    - Sensitivity patterns with statistical significance
    - HRV impact by time window
    - Confidence scores and p-values
    - Severity recommendations
    """
    try:
        # TODO: Fetch HRV readings and exposure data from database
        # For now, return mock analysis demonstrating structure

        return {
            "success": True,
            "user_id": request.user_id,
            "analysis_period_days": request.days_back,
            "message": "HRV sensitivity analysis requires historical data",
            "instructions": {
                "step_1": "Record HRV readings via wearable integration",
                "step_2": "Log meals with ingredients via /extract-ingredients",
                "step_3": "Track reactions via /exposure/record",
                "step_4": "Run analysis after 30+ exposures for statistical validity",
            },
            "minimum_requirements": {
                "hrv_readings": "3+ per day for baseline",
                "exposure_events": "5+ per trigger type",
                "reaction_rate": "Data on whether reactions occurred",
            },
        }

    except Exception as e:
        logger.error(f"HRV analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hrv/reaction-patterns/{sensitivity_type}")
async def get_reaction_pattern(sensitivity_type: str):
    """
    Get typical HRV reaction pattern for a sensitivity type.

    Different sensitivities show different temporal patterns:
    - Allergies: Peak at 30min-2hr
    - Intolerances: Peak at 2-6hr
    - FODMAP: Peak at 2-6hr with extended effects
    - Histamine: Variable, 30min-6hr
    """
    from app.models.sensitivity import SensitivityType

    try:
        sens_type = SensitivityType(sensitivity_type)
        pattern = hrv_sensitivity_analyzer.get_reaction_pattern_info(sens_type)
        return {
            "sensitivity_type": sensitivity_type,
            "pattern": pattern,
        }
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Unknown sensitivity type: {sensitivity_type}"
        )


# =============================================================================
# EXPOSURE TRACKING ENDPOINTS
# =============================================================================


@router.post("/exposure/record", response_model=RecordExposureResponse)
async def record_exposure(
    request: RecordExposureRequest = Body(...),
):
    """
    Record a sensitivity exposure event.

    Track when a user consumed a potential trigger food.
    This data is used for HRV correlation analysis.

    **Request:**
    ```json
    {
        "user_id": "user123",
        "allergen_type": "milk",
        "meal_id": "meal456",
        "exposed_at": "2024-01-15T12:30:00Z",
        "had_reaction": false
    }
    ```

    Reactions can be updated later via /exposure/reaction endpoint.
    """
    try:
        # TODO: Save to database
        # For now, return success demonstrating the flow

        import uuid

        exposure_id = str(uuid.uuid4())

        return RecordExposureResponse(
            success=True,
            exposure_id=exposure_id,
            message="Exposure recorded. Update with reaction data if symptoms occur.",
        )

    except Exception as e:
        logger.error(f"Record exposure failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/exposure/reaction")
async def update_exposure_reaction(
    request: UpdateReactionRequest = Body(...),
):
    """
    Update reaction information for a recorded exposure.

    Call this endpoint after symptoms appear (or confirm no reaction).

    **Request:**
    ```json
    {
        "exposure_id": "exp123",
        "had_reaction": true,
        "reaction_severity": "moderate",
        "symptoms": ["hives", "stomach pain"],
        "onset_minutes": 45,
        "duration_minutes": 180
    }
    ```
    """
    try:
        # TODO: Update in database
        return {
            "success": True,
            "exposure_id": request.exposure_id,
            "message": "Reaction data updated. This improves sensitivity detection accuracy.",
        }

    except Exception as e:
        logger.error(f"Update reaction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ML PREDICTION ENDPOINTS
# =============================================================================


@router.post("/predict")
async def predict_reaction(
    trigger_type: str = Body(..., description="Allergen or compound type"),
    baseline_rmssd: float = Body(..., description="User's baseline HRV (RMSSD)"),
    baseline_std: float = Body(15.0, description="Baseline HRV standard deviation"),
    hrv_drops: Optional[dict] = Body(None, description="Current HRV drops by window"),
    user_history: Optional[dict] = Body(
        None, description="User's history with this trigger"
    ),
    compound_info: Optional[dict] = Body(None, description="Compound amounts in meal"),
):
    """
    Predict reaction probability for a potential exposure.

    Uses ML model trained on HRV patterns to predict:
    - Likelihood of reaction (0-1)
    - Expected severity
    - Risk factors
    - Recommendations

    **Request:**
    ```json
    {
        "trigger_type": "milk",
        "baseline_rmssd": 42.5,
        "baseline_std": 12.0,
        "hrv_drops": {
            "immediate": -5.2,
            "short_term": -8.1
        },
        "user_history": {
            "prior_reaction_rate": 0.6,
            "exposure_count_last_30d": 3
        },
        "compound_info": {
            "total_histamine_mg": 15.0,
            "has_dao_inhibitor": false
        }
    }
    ```
    """
    try:
        from app.services.hrv_sensitivity_analyzer import TimeWindow

        # Build data point from request
        hrv_drop_dict = {}
        if hrv_drops:
            window_map = {
                "immediate": TimeWindow.IMMEDIATE,
                "short_term": TimeWindow.SHORT_TERM,
                "medium_term": TimeWindow.MEDIUM_TERM,
                "extended": TimeWindow.EXTENDED,
                "next_day": TimeWindow.NEXT_DAY,
            }
            for key, value in hrv_drops.items():
                if key in window_map:
                    hrv_drop_dict[window_map[key]] = value

        history = user_history or {}
        compounds = compound_info or {}

        data_point = TrainingDataPoint(
            hrv_drops=hrv_drop_dict,
            baseline_rmssd=baseline_rmssd,
            baseline_std=baseline_std,
            trigger_type=trigger_type,
            prior_reaction_rate=history.get("prior_reaction_rate", 0.0),
            days_since_last_exposure=history.get("days_since_last_exposure", 30),
            exposure_count_last_30d=history.get("exposure_count_last_30d", 0),
            has_dao_inhibitor=compounds.get("has_dao_inhibitor", False),
            has_histamine_liberator=compounds.get("has_histamine_liberator", False),
            total_histamine_mg=compounds.get("total_histamine_mg", 0.0),
            total_tyramine_mg=compounds.get("total_tyramine_mg", 0.0),
        )

        result = sensitivity_ml_model.predict(data_point)

        return {
            "reaction_probability": result.reaction_probability,
            "predicted_severity": result.predicted_severity.value,
            "severity_probabilities": result.severity_probabilities,
            "confidence": result.confidence,
            "risk_factors": result.risk_factors,
            "recommendations": result.recommendations,
            "model_trained": sensitivity_ml_model.is_trained,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """
    Get information about the sensitivity prediction model.

    Returns model status, performance metrics, and feature importance.
    """
    info = sensitivity_ml_model.get_model_info()
    return info


# =============================================================================
# USER SENSITIVITY MANAGEMENT
# =============================================================================


@router.post("/user/sensitivity")
async def add_user_sensitivity(
    request: AddUserSensitivityRequest = Body(...),
):
    """
    Add a new sensitivity to user's profile.

    **Request:**
    ```json
    {
        "user_id": "user123",
        "sensitivity_type": "allergy",
        "severity": "moderate",
        "allergen_type": "peanuts",
        "confirmed_by_test": true,
        "notes": "Diagnosed by allergist in 2020"
    }
    ```
    """
    try:
        # TODO: Save to database
        import uuid

        sensitivity_id = str(uuid.uuid4())

        return {
            "success": True,
            "sensitivity_id": sensitivity_id,
            "message": "Sensitivity added to profile",
        }

    except Exception as e:
        logger.error(f"Add sensitivity failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/user/{user_id}/sensitivities", response_model=GetUserSensitivitiesResponse
)
async def get_user_sensitivities(user_id: str):
    """
    Get all sensitivities for a user.

    Returns active and inactive sensitivities with correlation data.
    """
    try:
        # TODO: Fetch from database
        return GetUserSensitivitiesResponse(
            success=True,
            sensitivities=[],
            total_count=0,
        )

    except Exception as e:
        logger.error(f"Get sensitivities failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH CHECK
# =============================================================================


@router.get("/health")
async def sensitivity_health():
    """
    Health check for sensitivity detection service.

    Returns service status and component health.
    """
    try:
        # Check services
        ingredient_db_size = len(INGREDIENT_DATABASE)
        model_info = sensitivity_ml_model.get_model_info()

        return {
            "status": "healthy",
            "service": "sensitivity-detection",
            "components": {
                "ingredient_database": {
                    "status": "healthy",
                    "ingredients": ingredient_db_size,
                },
                "ml_model": {
                    "status": "healthy" if model_info["is_trained"] else "untrained",
                    "is_trained": model_info["is_trained"],
                },
                "allergen_types": len(AllergenType),
                "cross_reactivity_mappings": len(CROSS_REACTIVITY),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "sensitivity-detection",
                "error": str(e),
            },
        )
