"""
Food Feedback Schemas

Pydantic models for feedback API requests and responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# === Request Schemas ===


class FeedbackSubmitRequest(BaseModel):
    """Request to submit a classification correction."""

    original_prediction: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="The original prediction from the classifier",
    )
    original_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of original prediction"
    )
    corrected_label: str = Field(
        ..., min_length=1, max_length=100, description="The correct food label"
    )
    image_hash: str = Field(
        ..., min_length=64, max_length=64, description="SHA-256 hash of the image"
    )
    alternatives: Optional[List[Dict[str, Any]]] = Field(
        None, description="Top alternative predictions at time of classification"
    )
    user_description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description of the food for prompt learning",
    )

    @field_validator("corrected_label")
    @classmethod
    def normalize_label(cls, v: str) -> str:
        """Normalize the corrected label."""
        return v.lower().strip().replace(" ", "_")


class PromptSuggestionRequest(BaseModel):
    """Request to get prompt suggestions for a food."""

    food_key: str = Field(..., description="The food category key")


class ApplyPromptsRequest(BaseModel):
    """Request to apply learned prompts to the classifier."""

    food_keys: Optional[List[str]] = Field(
        None,
        description="Specific food keys to apply. If None, applies all pending prompts.",
    )
    min_feedback_count: int = Field(
        3,
        ge=1,
        description="Minimum number of feedback entries before applying prompts",
    )


# === Response Schemas ===


class FeedbackSubmitResponse(BaseModel):
    """Response after submitting feedback."""

    success: bool
    feedback_id: int
    message: str
    prompt_suggestions: Optional[List[str]] = None


class FeedbackStatsResponse(BaseModel):
    """Overall feedback statistics."""

    total_feedback: int
    pending_feedback: int
    approved_feedback: int
    rejected_feedback: int

    # Accuracy metrics
    correction_rate: float = Field(
        ..., description="Percentage of predictions that were corrected"
    )

    # Top misclassifications
    top_misclassifications: List[Dict[str, Any]] = Field(
        default_factory=list, description="Most common prediction -> correction pairs"
    )

    # Foods needing attention
    problem_foods: List[Dict[str, Any]] = Field(
        default_factory=list, description="Foods with highest correction rates"
    )

    # Learned prompts
    learned_prompts_count: int
    active_prompts_count: int


class FeedbackItem(BaseModel):
    """Individual feedback item."""

    id: int
    image_hash: str
    original_prediction: str
    original_confidence: float
    corrected_label: str
    user_description: Optional[str]
    status: str
    created_at: datetime


class FeedbackListResponse(BaseModel):
    """List of feedback items."""

    items: List[FeedbackItem]
    total: int
    page: int
    page_size: int


class PromptSuggestion(BaseModel):
    """A suggested prompt for a food category."""

    prompt: str
    source: str  # 'user_description', 'auto_generated', 'pattern'
    confidence: float
    feedback_count: int


class PromptSuggestionsResponse(BaseModel):
    """Prompt suggestions for improving a food category."""

    food_key: str
    current_prompts: List[str]
    suggested_prompts: List[PromptSuggestion]
    feedback_count: int
    common_corrections: List[Dict[str, Any]]


class ApplyPromptsResponse(BaseModel):
    """Response after applying prompts."""

    success: bool
    prompts_applied: int
    foods_updated: List[str]
    message: str


class CorrectionPattern(BaseModel):
    """Pattern of corrections for analytics."""

    original: str
    corrected: str
    count: int
    percentage: float
    suggested_action: str


class AnalyticsResponse(BaseModel):
    """Detailed analytics for feedback-driven learning."""

    time_period: str
    total_predictions: int
    total_corrections: int
    accuracy_rate: float

    by_category: Dict[str, Dict[str, Any]]
    correction_patterns: List[CorrectionPattern]

    improvement_opportunities: List[Dict[str, Any]]
