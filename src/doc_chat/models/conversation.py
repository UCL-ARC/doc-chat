"""Conversation history models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, Enum as SQLAEnum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationType(str, Enum):
    """Type of conversation."""

    SUMMARIZE = "summarize"
    QA = "qa"


class Conversation(Base):
    """Model for storing conversation metadata."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    type: Mapped[ConversationType] = mapped_column(SQLAEnum(ConversationType), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    user: Mapped["User"] = relationship("User", back_populates="conversations")


class Message(Base):
    """Model for storing individual messages in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer, ForeignKey("conversations.id"), nullable=False)
    role: Mapped[MessageRole] = mapped_column(SQLAEnum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    meta_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages") 