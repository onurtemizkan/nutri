"""
Feedback Service for Food Classification Learning

Handles:
1. Storing user corrections
2. Analyzing misclassification patterns
3. Generating new CLIP prompts from feedback
4. Applying learned improvements to the classifier
"""
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_

from app.models.food_feedback import FoodFeedback, LearnedPrompt
from app.ml_models.clip_food_classifier import FOOD_PROMPTS, get_clip_classifier

logger = logging.getLogger(__name__)


class FeedbackService:
    """
    Service for managing food classification feedback and learning.

    The feedback loop:
    1. User submits correction when classifier is wrong
    2. System aggregates corrections to find patterns
    3. New prompts are generated for problem foods
    4. CLIP classifier is updated with new prompts
    5. Accuracy improves over time
    """

    def __init__(self):
        self._prompt_cache: Dict[str, List[str]] = {}
        self._stats_cache: Optional[Dict] = None
        self._stats_cache_time: Optional[datetime] = None

    async def submit_feedback(
        self,
        db: AsyncSession,
        image_hash: str,
        original_prediction: str,
        original_confidence: float,
        corrected_label: str,
        alternatives: Optional[List[Dict]] = None,
        user_description: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[int, List[str]]:
        """
        Submit user feedback for a misclassification.

        Returns:
            Tuple of (feedback_id, suggested_prompts)
        """
        # Normalize labels
        original_prediction = original_prediction.lower().strip().replace(" ", "_")
        corrected_label = corrected_label.lower().strip().replace(" ", "_")

        # Check if this exact feedback already exists (same image, same correction)
        existing = await db.execute(
            select(FoodFeedback).where(
                and_(
                    FoodFeedback.image_hash == image_hash,
                    FoodFeedback.corrected_label == corrected_label
                )
            )
        )
        if existing.scalar_one_or_none():
            logger.info(f"Duplicate feedback for image {image_hash[:8]}...")
            return -1, []

        # Create feedback record
        feedback = FoodFeedback(
            image_hash=image_hash,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            corrected_label=corrected_label,
            alternatives=json.dumps(alternatives) if alternatives else None,
            user_description=user_description,
            user_id=user_id,
            status="pending"
        )

        db.add(feedback)
        await db.flush()

        # Generate prompt suggestions based on user description
        suggestions = []
        if user_description:
            suggestions = self._generate_prompts_from_description(
                corrected_label, user_description
            )
            # Store as learned prompts
            for prompt in suggestions:
                learned = LearnedPrompt(
                    food_key=corrected_label,
                    prompt=prompt,
                    source="user_description"
                )
                db.add(learned)

        logger.info(
            f"Feedback submitted: {original_prediction} -> {corrected_label} "
            f"(confidence: {original_confidence:.2f})"
        )

        # Invalidate stats cache
        self._stats_cache = None

        return feedback.id, suggestions

    def _generate_prompts_from_description(
        self,
        food_key: str,
        description: str
    ) -> List[str]:
        """Generate CLIP prompts from a user description."""
        prompts = []

        # Clean up the description
        desc_clean = description.strip().lower()

        # Generate variations
        food_name = food_key.replace("_", " ")

        # Direct description
        if len(desc_clean) > 10:
            prompts.append(f"a photo of {desc_clean}")

        # With food name
        if food_name not in desc_clean:
            prompts.append(f"{desc_clean} {food_name}")
            prompts.append(f"a {food_name} that is {desc_clean}")

        return prompts[:3]  # Limit to 3 prompts

    async def get_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get overall feedback statistics."""
        # Check cache (5 minute TTL)
        if (
            self._stats_cache
            and self._stats_cache_time
            and datetime.now() - self._stats_cache_time < timedelta(minutes=5)
        ):
            return self._stats_cache

        # Total counts by status
        status_counts = await db.execute(
            select(FoodFeedback.status, func.count(FoodFeedback.id))
            .group_by(FoodFeedback.status)
        )
        counts = dict(status_counts.fetchall())

        total = sum(counts.values())
        pending = counts.get("pending", 0)
        approved = counts.get("approved", 0)
        rejected = counts.get("rejected", 0)

        # Top misclassifications
        misclass_query = await db.execute(
            select(
                FoodFeedback.original_prediction,
                FoodFeedback.corrected_label,
                func.count(FoodFeedback.id).label("count")
            )
            .group_by(FoodFeedback.original_prediction, FoodFeedback.corrected_label)
            .order_by(desc("count"))
            .limit(10)
        )
        top_misclassifications = [
            {
                "original": row[0],
                "corrected": row[1],
                "count": row[2]
            }
            for row in misclass_query.fetchall()
        ]

        # Problem foods (most corrections needed)
        problem_query = await db.execute(
            select(
                FoodFeedback.original_prediction,
                func.count(FoodFeedback.id).label("correction_count"),
                func.avg(FoodFeedback.original_confidence).label("avg_confidence")
            )
            .group_by(FoodFeedback.original_prediction)
            .order_by(desc("correction_count"))
            .limit(10)
        )
        problem_foods = [
            {
                "food": row[0],
                "correction_count": row[1],
                "avg_confidence": round(float(row[2]) if row[2] else 0, 2)
            }
            for row in problem_query.fetchall()
        ]

        # Learned prompts count
        prompts_query = await db.execute(
            select(
                func.count(LearnedPrompt.id).label("total"),
                func.sum(LearnedPrompt.is_active).label("active")
            )
        )
        prompts_row = prompts_query.fetchone()
        learned_count = prompts_row[0] if prompts_row else 0
        active_count = prompts_row[1] if prompts_row else 0

        stats = {
            "total_feedback": total,
            "pending_feedback": pending,
            "approved_feedback": approved,
            "rejected_feedback": rejected,
            "correction_rate": 0.0,  # Would need total predictions to calculate
            "top_misclassifications": top_misclassifications,
            "problem_foods": problem_foods,
            "learned_prompts_count": learned_count,
            "active_prompts_count": active_count or 0
        }

        # Cache results
        self._stats_cache = stats
        self._stats_cache_time = datetime.now()

        return stats

    async def get_feedback_list(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        food_key: Optional[str] = None
    ) -> Tuple[List[Dict], int]:
        """Get paginated list of feedback items."""
        query = select(FoodFeedback)

        if status:
            query = query.where(FoodFeedback.status == status)
        if food_key:
            query = query.where(
                or_(
                    FoodFeedback.original_prediction == food_key,
                    FoodFeedback.corrected_label == food_key
                )
            )

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar()

        # Get page
        query = query.order_by(desc(FoodFeedback.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        items = [
            {
                "id": fb.id,
                "image_hash": fb.image_hash,
                "original_prediction": fb.original_prediction,
                "original_confidence": fb.original_confidence,
                "corrected_label": fb.corrected_label,
                "user_description": fb.user_description,
                "status": fb.status,
                "created_at": fb.created_at.isoformat() if fb.created_at else None
            }
            for fb in result.scalars()
        ]

        return items, total

    async def get_prompt_suggestions(
        self,
        db: AsyncSession,
        food_key: str
    ) -> Dict[str, Any]:
        """
        Get prompt suggestions for a food category based on feedback.
        """
        food_key = food_key.lower().strip().replace(" ", "_")

        # Current prompts
        current_prompts = FOOD_PROMPTS.get(food_key, [f"a photo of {food_key}"])

        # Get learned prompts
        learned_query = await db.execute(
            select(LearnedPrompt)
            .where(LearnedPrompt.food_key == food_key)
            .order_by(desc(LearnedPrompt.success_count))
        )
        learned_prompts = [
            {
                "prompt": lp.prompt,
                "source": lp.source,
                "confidence": (lp.success_count / max(lp.times_used, 1)) if lp.times_used > 0 else 0.5,
                "feedback_count": lp.times_used
            }
            for lp in learned_query.scalars()
        ]

        # Get correction patterns (what this food is confused with)
        corrections_query = await db.execute(
            select(
                FoodFeedback.original_prediction,
                func.count(FoodFeedback.id).label("count")
            )
            .where(FoodFeedback.corrected_label == food_key)
            .group_by(FoodFeedback.original_prediction)
            .order_by(desc("count"))
            .limit(5)
        )
        common_corrections = [
            {"confused_with": row[0], "count": row[1]}
            for row in corrections_query.fetchall()
        ]

        # Auto-generate prompts based on confusion patterns
        auto_suggestions = self._generate_disambiguation_prompts(
            food_key, common_corrections
        )

        # Count total feedback for this food
        feedback_count = await db.execute(
            select(func.count(FoodFeedback.id))
            .where(FoodFeedback.corrected_label == food_key)
        )
        count = feedback_count.scalar() or 0

        return {
            "food_key": food_key,
            "current_prompts": current_prompts,
            "suggested_prompts": learned_prompts + auto_suggestions,
            "feedback_count": count,
            "common_corrections": common_corrections
        }

    def _generate_disambiguation_prompts(
        self,
        food_key: str,
        confusion_pairs: List[Dict]
    ) -> List[Dict]:
        """
        Generate prompts that help distinguish this food from confused items.
        """
        suggestions = []
        food_name = food_key.replace("_", " ")

        for pair in confusion_pairs[:3]:
            confused_with = pair["confused_with"].replace("_", " ")
            count = pair["count"]

            # Generate distinguishing prompts
            prompts = [
                f"a {food_name}, not {confused_with}",
                f"clearly a {food_name}",
                f"a photo showing {food_name} specifically"
            ]

            for prompt in prompts:
                suggestions.append({
                    "prompt": prompt,
                    "source": "auto_generated",
                    "confidence": min(0.7, count / 10),
                    "feedback_count": count
                })

        return suggestions

    async def apply_learned_prompts(
        self,
        db: AsyncSession,
        food_keys: Optional[List[str]] = None,
        min_feedback_count: int = 3
    ) -> Tuple[int, List[str]]:
        """
        Apply learned prompts to the CLIP classifier.

        Returns:
            Tuple of (number of prompts applied, list of updated foods)
        """
        # Get active learned prompts
        query = select(LearnedPrompt).where(LearnedPrompt.is_active == 1)

        if food_keys:
            query = query.where(LearnedPrompt.food_key.in_(food_keys))

        result = await db.execute(query)
        learned = list(result.scalars())

        # Group by food key
        prompts_by_food: Dict[str, List[str]] = defaultdict(list)
        for lp in learned:
            prompts_by_food[lp.food_key].append(lp.prompt)

        # Only apply if we have enough feedback
        updated_foods = []
        total_prompts = 0

        for food_key, prompts in prompts_by_food.items():
            # Check feedback count
            feedback_count = await db.execute(
                select(func.count(FoodFeedback.id))
                .where(FoodFeedback.corrected_label == food_key)
            )
            count = feedback_count.scalar() or 0

            if count >= min_feedback_count:
                # Add to CLIP classifier
                if food_key in FOOD_PROMPTS:
                    existing = set(FOOD_PROMPTS[food_key])
                    new_prompts = [p for p in prompts if p not in existing]
                    if new_prompts:
                        FOOD_PROMPTS[food_key].extend(new_prompts[:3])  # Limit additions
                        updated_foods.append(food_key)
                        total_prompts += len(new_prompts[:3])
                else:
                    # New food category
                    FOOD_PROMPTS[food_key] = prompts[:5]
                    updated_foods.append(food_key)
                    total_prompts += len(prompts[:5])

        # Rebuild CLIP text features if any updates
        if updated_foods:
            try:
                classifier = get_clip_classifier()
                if classifier._loaded:
                    classifier._precompute_text_features()
                    logger.info(f"Rebuilt CLIP text features for {len(updated_foods)} foods")
            except Exception as e:
                logger.error(f"Failed to rebuild CLIP features: {e}")

        return total_prompts, updated_foods

    async def get_analytics(
        self,
        db: AsyncSession,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get detailed analytics for feedback-driven learning."""
        cutoff = datetime.now() - timedelta(days=days)

        # Get all feedback in period
        feedback_query = await db.execute(
            select(FoodFeedback)
            .where(FoodFeedback.created_at >= cutoff)
        )
        feedbacks = list(feedback_query.scalars())

        total_corrections = len(feedbacks)

        # Analyze by category
        by_category: Dict[str, Dict] = defaultdict(
            lambda: {"corrections": 0, "examples": []}
        )

        for fb in feedbacks:
            by_category[fb.corrected_label]["corrections"] += 1
            if len(by_category[fb.corrected_label]["examples"]) < 3:
                by_category[fb.corrected_label]["examples"].append({
                    "from": fb.original_prediction,
                    "confidence": fb.original_confidence
                })

        # Correction patterns
        pattern_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for fb in feedbacks:
            pattern_counts[(fb.original_prediction, fb.corrected_label)] += 1

        patterns = [
            {
                "original": orig,
                "corrected": corr,
                "count": count,
                "percentage": count / total_corrections * 100 if total_corrections > 0 else 0,
                "suggested_action": self._suggest_action(orig, corr, count)
            }
            for (orig, corr), count in sorted(
                pattern_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        ]

        # Improvement opportunities
        opportunities = []
        for (orig, corr), count in pattern_counts.items():
            if count >= 3:
                opportunities.append({
                    "type": "add_prompt",
                    "food_key": corr,
                    "reason": f"Confused with {orig} {count} times",
                    "priority": count,
                    "suggested_prompts": [
                        f"a {corr.replace('_', ' ')}, not {orig.replace('_', ' ')}",
                        f"clearly {corr.replace('_', ' ')} food"
                    ]
                })

        opportunities.sort(key=lambda x: x["priority"], reverse=True)

        return {
            "time_period": f"last {days} days",
            "total_predictions": 0,  # Would need tracking
            "total_corrections": total_corrections,
            "accuracy_rate": 0.0,  # Would need total predictions
            "by_category": dict(by_category),
            "correction_patterns": patterns,
            "improvement_opportunities": opportunities[:10]
        }

    def _suggest_action(self, original: str, corrected: str, count: int) -> str:
        """Suggest an action based on correction pattern."""
        if count >= 10:
            return f"HIGH PRIORITY: Add disambiguation prompts for {corrected}"
        elif count >= 5:
            return f"Add new prompts to distinguish {corrected} from {original}"
        else:
            return "Monitor - may need prompt adjustments"

    async def update_feedback_status(
        self,
        db: AsyncSession,
        feedback_id: int,
        status: str
    ) -> bool:
        """Update the status of a feedback item."""
        result = await db.execute(
            select(FoodFeedback).where(FoodFeedback.id == feedback_id)
        )
        feedback = result.scalar_one_or_none()

        if not feedback:
            return False

        feedback.status = status
        return True


# Singleton instance
feedback_service = FeedbackService()
