"""Conversation schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.conversation import ConversationType, MessageRole


class MessageBase(BaseModel):
    """Base schema for message."""

    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = Field(None, alias="meta_data")


class MessageCreate(MessageBase):
    """Schema for creating a message."""

    pass


class Message(MessageBase):
    """Schema for message response."""

    id: int
    conversation_id: int
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True
        populate_by_name = True


class ConversationBase(BaseModel):
    """Base schema for conversation."""

    type: ConversationType
    title: str
    metadata: Optional[Dict[str, Any]] = Field(None, alias="meta_data")


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""

    pass


class Conversation(ConversationBase):
    """Schema for conversation response."""

    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    messages: List[Message]

    class Config:
        """Pydantic config."""

        from_attributes = True
        populate_by_name = True


class ConversationList(BaseModel):
    """Schema for list of conversations."""

    conversations: List[Conversation]

    class Config:
        """Pydantic config."""

        from_attributes = True
        populate_by_name = True 