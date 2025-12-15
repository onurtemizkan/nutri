"""
Food Classification Feedback API

Endpoints for:
1. Submitting corrections when classifier is wrong
2. Viewing feedback statistics
3. Managing learned prompts
4. Applying improvements to the classifier
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.food_feedback import (
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    FeedbackStatsResponse,
    FeedbackListResponse,
    FeedbackItem,
    PromptSuggestionsResponse,
    PromptSuggestion,
    ApplyPromptsRequest,
    ApplyPromptsResponse,
    AnalyticsResponse,
    CorrectionPattern,
)
from app.services.feedback_service import feedback_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackSubmitResponse)
async def submit_feedback(
    request: FeedbackSubmitRequest, db: AsyncSession = Depends(get_db)
):
    """
    Submit feedback when food classification is incorrect.

    **Use this endpoint when:**
    - The classifier predicted the wrong food
    - You want to help improve accuracy

    **Parameters:**
    - **original_prediction**: What the classifier predicted
    - **original_confidence**: Confidence score (0-1)
    - **corrected_label**: The correct food name
    - **image_hash**: SHA-256 hash of the image
    - **alternatives** (optional): Other predictions shown
    - **user_description** (optional): Description to help learn better prompts

    **Example:**
    ```json
    {
        "original_prediction": "apple",
        "original_confidence": 0.85,
        "corrected_label": "tomato",
        "image_hash": "abc123...",
        "user_description": "a red ripe tomato on the vine"
    }
    ```
    """
    try:
        feedback_id, suggestions = await feedback_service.submit_feedback(
            db=db,
            image_hash=request.image_hash,
            original_prediction=request.original_prediction,
            original_confidence=request.original_confidence,
            corrected_label=request.corrected_label,
            alternatives=request.alternatives,
            user_description=request.user_description,
        )

        if feedback_id == -1:
            return FeedbackSubmitResponse(
                success=True,
                feedback_id=0,
                message="Feedback already recorded for this image",
                prompt_suggestions=None,
            )

        return FeedbackSubmitResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thank you! Your feedback helps improve food recognition.",
            prompt_suggestions=suggestions if suggestions else None,
        )

    except Exception as e:
        logger.error(f"Feedback submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(db: AsyncSession = Depends(get_db)):
    """
    Get overall feedback statistics.

    **Returns:**
    - Total feedback counts by status
    - Top misclassification pairs
    - Foods needing the most improvement
    - Learned prompt statistics
    """
    try:
        stats = await feedback_service.get_stats(db)
        return FeedbackStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/feedback/list", response_model=FeedbackListResponse)
async def list_feedback(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    food_key: Optional[str] = Query(None, description="Filter by food"),
    db: AsyncSession = Depends(get_db),
):
    """
    List feedback items with pagination.

    **Filters:**
    - **status**: pending, approved, rejected
    - **food_key**: Filter by original prediction or corrected label
    """
    try:
        items, total = await feedback_service.get_feedback_list(
            db=db, page=page, page_size=page_size, status=status, food_key=food_key
        )

        return FeedbackListResponse(
            items=[FeedbackItem(**item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.error(f"List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list feedback")


@router.get(
    "/feedback/suggestions/{food_key}", response_model=PromptSuggestionsResponse
)
async def get_prompt_suggestions(food_key: str, db: AsyncSession = Depends(get_db)):
    """
    Get prompt suggestions for a specific food category.

    Based on user feedback, this endpoint suggests new CLIP prompts
    that could improve classification accuracy for the specified food.

    **Returns:**
    - Current prompts in use
    - Suggested new prompts
    - Common confusion pairs
    - Feedback count
    """
    try:
        result = await feedback_service.get_prompt_suggestions(db, food_key)

        return PromptSuggestionsResponse(
            food_key=result["food_key"],
            current_prompts=result["current_prompts"],
            suggested_prompts=[
                PromptSuggestion(
                    prompt=s["prompt"],
                    source=s["source"],
                    confidence=s["confidence"],
                    feedback_count=s["feedback_count"],
                )
                for s in result["suggested_prompts"]
            ],
            feedback_count=result["feedback_count"],
            common_corrections=result["common_corrections"],
        )
    except Exception as e:
        logger.error(f"Suggestions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get suggestions")


@router.post("/feedback/apply", response_model=ApplyPromptsResponse)
async def apply_learned_prompts(
    request: Optional[ApplyPromptsRequest] = None, db: AsyncSession = Depends(get_db)
):
    """
    Apply learned prompts to the CLIP classifier.

    This updates the classifier's text embeddings with new prompts
    learned from user feedback.

    **Note:** This is an admin endpoint that modifies the classifier.

    **Parameters:**
    - **food_keys** (optional): Specific foods to update
    - **min_feedback_count**: Minimum corrections before applying (default: 3)
    """
    try:
        if request is None:
            request = ApplyPromptsRequest()

        prompts_applied, foods_updated = await feedback_service.apply_learned_prompts(
            db=db,
            food_keys=request.food_keys,
            min_feedback_count=request.min_feedback_count,
        )

        return ApplyPromptsResponse(
            success=True,
            prompts_applied=prompts_applied,
            foods_updated=foods_updated,
            message=f"Applied {prompts_applied} new prompts to {len(foods_updated)} food categories",
        )
    except Exception as e:
        logger.error(f"Apply prompts error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to apply prompts")


@router.get("/feedback/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = Query(30, ge=1, le=365, description="Days to analyze"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get detailed analytics for feedback-driven learning.

    **Returns:**
    - Correction patterns over time
    - Category-by-category breakdown
    - Prioritized improvement opportunities
    """
    try:
        analytics = await feedback_service.get_analytics(db, days)

        return AnalyticsResponse(
            time_period=analytics["time_period"],
            total_predictions=analytics["total_predictions"],
            total_corrections=analytics["total_corrections"],
            accuracy_rate=analytics["accuracy_rate"],
            by_category=analytics["by_category"],
            correction_patterns=[
                CorrectionPattern(**p) for p in analytics["correction_patterns"]
            ],
            improvement_opportunities=analytics["improvement_opportunities"],
        )
    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get analytics")


@router.patch("/feedback/{feedback_id}/status")
async def update_feedback_status(
    feedback_id: int,
    status: str = Query(..., regex="^(pending|approved|rejected)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Update the status of a feedback item.

    **Statuses:**
    - **pending**: Awaiting review
    - **approved**: Confirmed correct
    - **rejected**: Marked as incorrect/spam
    """
    try:
        success = await feedback_service.update_feedback_status(db, feedback_id, status)

        if not success:
            raise HTTPException(status_code=404, detail="Feedback not found")

        return {"success": True, "message": f"Status updated to {status}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update status")


@router.get("/feedback/export")
async def export_feedback(
    format: str = Query("json", regex="^(json|csv)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Export all feedback data for analysis.

    **Formats:**
    - **json**: Full JSON export
    - **csv**: CSV for spreadsheet analysis
    """
    try:
        items, total = await feedback_service.get_feedback_list(
            db=db, page=1, page_size=10000  # Get all
        )

        if format == "csv":
            import csv
            import io
            from fastapi.responses import StreamingResponse

            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "id",
                    "image_hash",
                    "original_prediction",
                    "original_confidence",
                    "corrected_label",
                    "user_description",
                    "status",
                    "created_at",
                ],
            )
            writer.writeheader()
            writer.writerows(items)

            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=feedback_export.csv"
                },
            )

        return {"items": items, "total": total}

    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to export data")
