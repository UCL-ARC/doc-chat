"""User model module."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional, List, Dict

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .document import Document
    from .conversation import Conversation


class User(Base):
    """
    User model for authentication and user management.

    Attributes:
        id: Primary key.
        email: User's email address.
        hashed_password: Hashed password.
        full_name: User's full name.
        is_active: Whether the user is active.
        is_superuser: Whether the user is a superuser.
        created_at: When the user was created.
        updated_at: When the user was last updated.
        documents: List of documents associated with the user.
        settings: Relationship to UserSettings.
        conversations: List of conversations associated with the user.

    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )
    settings: Mapped[Optional["UserSettings"]] = relationship(
        "UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class UserSettings(Base):
    """
    UserSettings model for storing per-user settings.

    Attributes:
        id: Primary key.
        user_id: Foreign key to User.
        pdf_parser: Selected PDF parser (e.g., 'tesseract', 'llm').
        model_name: Selected LLM model (e.g., 'gpt-4o').
        api_keys: Dict of API keys per provider.
        prompts: Dict of prompts for summarize, qa, etc.
        created_at: When the settings were created.
        updated_at: When the settings were last updated.
        user: Relationship to User.

    """

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), unique=True)
    pdf_parser: Mapped[str] = mapped_column(String(50), default="tesseract")
    model_name: Mapped[Optional[str]] = mapped_column(String(100), default="azure/gpt-4o-mini")
    api_keys: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict)
    prompts: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    user: Mapped["User"] = relationship("User", back_populates="settings")
