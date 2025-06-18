"""Conversation router module."""

from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models.conversation import Conversation, ConversationType, Message, MessageRole
from ..models.user import User
from ..routers.documents import get_current_user
from ..schemas.conversation import (
    Conversation as ConversationSchema,
    ConversationCreate,
    ConversationList,
    Message as MessageSchema,
    MessageCreate,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/", response_model=ConversationSchema)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Create a new conversation."""
    db_conversation = Conversation(
        user_id=current_user.id,
        type=conversation.type,
        title=conversation.title,
        meta_data=conversation.metadata,
    )
    db.add(db_conversation)
    await db.commit()
    await db.refresh(db_conversation)
    return db_conversation


@router.get("/", response_model=ConversationList)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """List all conversations for the current user."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()
    return ConversationList(conversations=conversations)


@router.get("/{conversation_id}", response_model=ConversationSchema)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Get a specific conversation."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


@router.post("/{conversation_id}/messages", response_model=MessageSchema)
async def add_message(
    conversation_id: int,
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Add a message to a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    db_message = Message(
        conversation_id=conversation_id,
        role=message.role,
        content=message.content,
        meta_data=message.metadata,
    )
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    await db.delete(conversation)
    await db.commit() 