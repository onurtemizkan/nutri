"""Meal model matching Prisma schema"""
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, func, Index
from sqlalchemy.orm import relationship
from app.database import Base


class Meal(Base):
    __tablename__ = "Meal"
    __table_args__ = (
        Index("Meal_userId_consumedAt_idx", "userId", "consumedAt"),
        Index("Meal_userId_mealType_idx", "userId", "mealType"),
    )

    id = Column(String, primary_key=True)
    user_id = Column("userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False)

    name = Column(String, nullable=False)
    meal_type = Column("mealType", String, nullable=False)  # breakfast, lunch, dinner, snack
    calories = Column(Float, nullable=False)
    protein = Column(Float, nullable=False)
    carbs = Column(Float, nullable=False)
    fat = Column(Float, nullable=False)
    fiber = Column(Float, nullable=True)
    sugar = Column(Float, nullable=True)

    serving_size = Column("servingSize", String, nullable=True)
    notes = Column(String, nullable=True)
    image_url = Column("imageUrl", String, nullable=True)

    consumed_at = Column("consumedAt", DateTime, server_default=func.now())
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column("updatedAt", DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="meals")

    def __repr__(self):
        return f"<Meal(id={self.id}, name={self.name}, calories={self.calories})>"
