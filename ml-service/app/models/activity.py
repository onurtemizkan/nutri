"""Activity model matching Prisma schema"""
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, func, Index
from sqlalchemy.orm import relationship
from app.database import Base


class Activity(Base):
    __tablename__ = "Activity"
    __table_args__ = (
        Index("Activity_userId_startedAt_idx", "userId", "startedAt"),
        Index("Activity_userId_activityType_startedAt_idx", "userId", "activityType", "startedAt"),
    )

    id = Column(String, primary_key=True)
    user_id = Column("userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False)

    # Timing
    started_at = Column("startedAt", DateTime, nullable=False)
    ended_at = Column("endedAt", DateTime, nullable=False)
    duration = Column(Integer, nullable=False)  # Duration in minutes

    # Type & Intensity
    activity_type = Column("activityType", String, nullable=False)
    intensity = Column(String, nullable=False)

    # Metrics
    calories_burned = Column("caloriesBurned", Float, nullable=True)
    average_heart_rate = Column("averageHeartRate", Float, nullable=True)
    max_heart_rate = Column("maxHeartRate", Float, nullable=True)
    distance = Column(Float, nullable=True)  # In meters
    steps = Column(Integer, nullable=True)

    # Source
    source = Column(String, nullable=False)  # "apple_health", "strava", "garmin", "manual"
    source_id = Column("sourceId", String, nullable=True)

    # Notes
    notes = Column(String, nullable=True)

    # Audit
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column("updatedAt", DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="activities")

    def __repr__(self):
        return f"<Activity(id={self.id}, type={self.activity_type}, duration={self.duration}min)>"
