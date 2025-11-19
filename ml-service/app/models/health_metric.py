"""HealthMetric model matching Prisma schema"""
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, func, Index
from sqlalchemy.orm import relationship
from app.database import Base


class HealthMetric(Base):
    __tablename__ = "HealthMetric"
    __table_args__ = (
        Index("HealthMetric_userId_recordedAt_idx", "userId", "recordedAt"),
        Index("HealthMetric_userId_metricType_recordedAt_idx", "userId", "metricType", "recordedAt"),
    )

    id = Column(String, primary_key=True)
    user_id = Column("userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False)

    # Timestamp (UTC, precise to second)
    recorded_at = Column("recordedAt", DateTime, nullable=False)

    # Metric type (enum for type safety)
    metric_type = Column("metricType", String, nullable=False)

    # Value (flexible for different units)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)  # "bpm", "ms", "%", "steps", "kcal", etc.

    # Source (for data provenance)
    source = Column(String, nullable=False)  # "apple_health", "fitbit", "garmin", "oura", "whoop", "manual"
    source_id = Column("sourceId", String, nullable=True)  # Original ID from source system

    # Metadata (JSON for flexibility)
    metric_metadata = Column("metadata", JSON, nullable=True)  # {quality: "high", confidence: 0.95, device: "Apple Watch Series 9"}

    # Audit
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column("updatedAt", DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="health_metrics")

    def __repr__(self):
        return f"<HealthMetric(id={self.id}, type={self.metric_type}, value={self.value} {self.unit})>"
