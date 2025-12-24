"""
Food Classification Feedback Model

Stores user corrections for misclassified food items.
Used to improve CLIP prompts and track classification accuracy over time.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Index
from sqlalchemy import func
from app.database import Base
import hashlib


class FoodFeedback(Base):
    """
    User feedback for food classification corrections.

    When a user disagrees with a classification, they can submit
    the correct label. This data is used to:
    1. Track which foods are frequently misclassified
    2. Generate new CLIP prompts for problem foods
    3. Measure classifier accuracy over time
    """
    __tablename__ = "FoodFeedback"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Image identification (hash for deduplication)
    image_hash = Column("imageHash", String(64), nullable=False, index=True)

    # Original prediction from the classifier
    original_prediction = Column("originalPrediction", String(100), nullable=False, index=True)
    original_confidence = Column("originalConfidence", Float, nullable=False)
    original_category = Column("originalCategory", String(100), nullable=True)

    # Classification reference
    classification_id = Column("classificationId", String, nullable=True)

    # User's correction (selected food from FDC)
    selected_fdc_id = Column("selectedFdcId", Integer, nullable=False)
    selected_food_name = Column("selectedFoodName", String(200), nullable=False)
    was_correct = Column("wasCorrect", Integer, nullable=False)  # SQLite boolean

    # Classification hints for future improvements (JSON)
    classification_hints = Column("classificationHints", Text, nullable=True)

    # Optional: user-provided description for prompt learning
    user_description = Column("userDescription", Text, nullable=True)

    # Optional: user ID if authenticated
    user_id = Column("userId", String, nullable=True, index=True)

    # Feedback status
    status = Column(String(20), default="pending")  # pending, approved, rejected

    # Timestamps
    created_at = Column("createdAt", DateTime, server_default=func.now())

    # Create composite index for common queries (matching Prisma schema)
    __table_args__ = (
        Index('idx_feedback_prediction_food', 'originalPrediction', 'selectedFoodName'),
        Index('idx_feedback_status_created', 'status', 'createdAt'),
    )

    def __repr__(self):
        return f"<FoodFeedback(id={self.id}, {self.original_prediction} -> {self.selected_food_name})>"


class LearnedPrompt(Base):
    """
    Prompts learned from user feedback.

    When the same correction is received multiple times,
    the system generates new CLIP prompts to improve accuracy.
    """
    __tablename__ = "LearnedPrompt"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Target food category
    food_key = Column(String(100), nullable=False, index=True)

    # The learned prompt text
    prompt = Column(Text, nullable=False)

    # Source: how this prompt was generated
    source = Column(String(50), nullable=False)  # user_description, auto_generated, admin

    # Effectiveness tracking
    times_used = Column(Integer, default=0)
    success_count = Column(Integer, default=0)

    # Status
    is_active = Column(Integer, default=1)  # SQLite doesn't have boolean

    # Timestamps
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column("updatedAt", DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_learned_prompt_food_active', 'food_key', 'is_active'),
    )

    def __repr__(self):
        return f"<LearnedPrompt(food={self.food_key}, prompt='{self.prompt[:30]}...')>"


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image for deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()
