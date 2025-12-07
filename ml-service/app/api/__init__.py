"""
API routes for ML service endpoints.
"""

from fastapi import APIRouter

from .features import router as features_router
from .correlations import router as correlations_router
from .predictions import router as predictions_router
from .interpretability import router as interpretability_router
from .food_analysis import router as food_analysis_router

# Main API router
api_router = APIRouter(prefix="/api")

# Register sub-routers
api_router.include_router(features_router, prefix="/features", tags=["features"])
api_router.include_router(
    correlations_router, prefix="/correlations", tags=["correlations"]
)
api_router.include_router(
    predictions_router, prefix="/predictions", tags=["predictions"]
)
api_router.include_router(
    interpretability_router, prefix="/interpretability", tags=["interpretability"]
)
api_router.include_router(food_analysis_router, prefix="/food", tags=["food-analysis"])

__all__ = ["api_router"]
