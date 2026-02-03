"""Authentication router module."""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth.utils import create_access_token, get_password_hash, verify_password
from ..config import settings
from ..database import get_db
from ..llm_manager import ensure_default_model, list_local_models
from ..models.user import User, UserSettings
from ..schemas.auth import (
    Token,
    UserCreate,
    UserResponse,
    UserSettingsBase,
    UserSettingsResponse,
)
from .documents import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


@router.get("/ollama/models")
async def get_ollama_models(
    current_user: User = Depends(get_current_user),
) -> dict[str, list[str]]:
    """
    Return list of Ollama model names available on the server (with 'ollama/' prefix).
    """
    names = await list_local_models()
    return {"models": [f"ollama/{n}" for n in names]}


@router.get("/ollama/ensure-default")
async def ensure_ollama_default(
    current_user: User = Depends(get_current_user),
) -> dict[str, str | bool]:
    """
    Ensure the default model (gemma3:1b) is available; pull if not.
    Frontend can show 'Downloading default model (one time)...' while this request is in flight.
    """
    result = await ensure_default_model()
    return {"status": "ready", "pulled": result["pulled"]}


@router.get("/status")
async def get_auth_status() -> dict[str, bool]:
    """
    Get authentication status.
    
    Returns:
        dict: Whether authentication is disabled.
    """
    return {"auth_disabled": settings.DISABLE_AUTH}


@router.post("/signup", response_model=UserResponse)
async def signup(user_data: UserCreate, db: AsyncSession = Depends(get_db)) -> Any:
    """
    Create a new user.

    Args:
        user_data: User registration data.
        db: Database session.

    Returns:
        UserResponse: Created user data.

    Raises:
        HTTPException: If email already exists.

    """
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Login user and return access token.

    Args:
        form_data: Login form data.
        db: Database session.

    Returns:
        Token: Access token.

    Raises:
        HTTPException: If credentials are invalid.

    """
    # Get user
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/settings", response_model=UserSettingsResponse)
async def get_user_settings(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get the current user's settings.

    Args:
        current_user: The authenticated user.
        db: Database session.

    Returns:
        UserSettingsResponse: The user's settings.

    """
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    settings = result.scalar_one_or_none()
    if not settings:
        # Create default settings if not exist
        settings = UserSettings(
            user_id=current_user.id,
            model_name="ollama/gemma3:1b",
            pdf_parser="tesseract",
            api_keys={},
            prompts={
                "summarize": "Summarize the following text for efficacy and clarity.",
                "qa": "Given the following text, answer the question as accurately as possible."
            }
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


@router.post("/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    settings_in: UserSettingsBase,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Update the current user's settings.

    Args:
        settings_in: The new settings data.
        current_user: The authenticated user.
        db: Database session.

    Returns:
        UserSettingsResponse: The updated settings.

    """
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    settings = result.scalar_one_or_none()
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)
    settings.pdf_parser = settings_in.pdf_parser
    settings.model_name = settings_in.model_name
    settings.api_keys = settings_in.api_keys
    settings.prompts = settings_in.prompts
    await db.commit()
    await db.refresh(settings)
    return settings
