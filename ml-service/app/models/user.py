"""User model matching Prisma schema"""
from sqlalchemy import Boolean, Column, String, Integer, Float, DateTime, func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column(
        "updatedAt", DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Account status
    is_active = Column("isActive", Boolean, default=True, nullable=False)

    # Password reset
    reset_token = Column("resetToken", String, nullable=True, index=True)
    reset_token_expires_at = Column("resetTokenExpiresAt", DateTime, nullable=True)

    # User profile data
    goal_calories = Column("goalCalories", Integer, default=2000)
    goal_protein = Column("goalProtein", Float, default=150.0)
    goal_carbs = Column("goalCarbs", Float, default=200.0)
    goal_fat = Column("goalFat", Float, default=65.0)
    current_weight = Column("currentWeight", Float, nullable=True)
    goal_weight = Column("goalWeight", Float, nullable=True)
    height = Column("height", Float, nullable=True)
    activity_level = Column("activityLevel", String, default="moderate")

    # Relationships (for eager loading if needed)
    meals = relationship("Meal", back_populates="user", lazy="select")
    health_metrics = relationship("HealthMetric", back_populates="user", lazy="select")
    activities = relationship("Activity", back_populates="user", lazy="select")

    # Sensitivity tracking relationships
    sensitivities = relationship(
        "UserSensitivity",
        back_populates="user",
        lazy="select",
        cascade="all, delete-orphan",
    )
    sensitivity_exposures = relationship(
        "SensitivityExposure",
        back_populates="user",
        lazy="select",
        cascade="all, delete-orphan",
    )
    sensitivity_insights = relationship(
        "SensitivityInsight",
        back_populates="user",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
