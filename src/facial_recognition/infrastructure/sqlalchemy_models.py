from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PersonModel(Base):
    __tablename__ = "persons"

    person_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sex: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class FaceEmbeddingModel(Base):
    __tablename__ = "face_embeddings"

    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("persons.person_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class FaceSampleModel(Base):
    __tablename__ = "face_samples"

    sample_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("persons.person_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    capture_type: Mapped[str] = mapped_column(String(32), nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    pose_yaw: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_pitch: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_roll: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class RecognitionEventModel(Base):
    __tablename__ = "recognition_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    track_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision: Mapped[str] = mapped_column(String(64), nullable=False)
    top1_person_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    top1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    top2_person_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    top2_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
