from sqlalchemy import Column, Text, Integer, Float, ForeignKey, Index
from webapp.database import Base
from webapp.config import settings
from pathlib import Path


class User(Base):
    __tablename__ = "users"

    id = Column(Text, primary_key=True)
    email = Column(Text, unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    display_name = Column(Text)
    tier = Column(Text, nullable=False, default="free")
    stripe_customer_id = Column(Text)
    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)

    @property
    def is_pro(self) -> bool:
        return self.tier == "pro"


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Text, primary_key=True)
    user_id = Column(Text, ForeignKey("users.id"), nullable=False, index=True)
    stripe_subscription_id = Column(Text, unique=True, nullable=False)
    stripe_price_id = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
    current_period_start = Column(Text)
    current_period_end = Column(Text)
    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Text, primary_key=True)
    user_id = Column(Text, ForeignKey("users.id"), nullable=True)
    status = Column(Text, nullable=False, default="pending", index=True)
    progress = Column(Integer, default=0)
    progress_message = Column(Text)
    original_filename = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    duration_s = Column(Float)
    tier_at_creation = Column(Text, nullable=False)
    midi_full_path = Column(Text)
    midi_piano_path = Column(Text)
    midi_bass_path = Column(Text)
    midi_drums_path = Column(Text)
    wav_path = Column(Text)
    error_message = Column(Text)
    created_at = Column(Text, nullable=False)
    completed_at = Column(Text)

    @property
    def output_dir(self) -> Path:
        return Path(settings.UPLOAD_DIR) / "jobs" / self.id
