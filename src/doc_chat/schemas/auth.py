"""Authentication schemas module."""

from pydantic import BaseModel, EmailStr, Field, constr


class Token(BaseModel):
    """
    Token response schema.

    Attributes:
        access_token: JWT access token.
        token_type: Type of token (bearer).

    """

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Token data schema.

    Attributes:
        email: User's email address.

    """

    email: str | None = None


class UserBase(BaseModel):
    """
    Base user schema.

    Attributes:
        email: User's email address.
        full_name: User's full name.

    """

    email: EmailStr
    full_name: str | None = None


class UserCreate(UserBase):
    """
    User creation schema.

    Attributes:
        password: User's password.

    """

    password: constr(min_length=8)


class UserResponse(UserBase):
    """
    User response schema.

    Attributes:
        id: User's ID.
        is_active: Whether the user is active.

    """

    id: int
    is_active: bool

    class Config:
        """Pydantic model configuration."""

        from_attributes = True


class UserSettingsBase(BaseModel):
    """
    Base schema for user settings.

    Attributes:
        pdf_parser: Selected PDF parser.
        model_name: Selected LLM model name.
        api_keys: Dict of API keys per provider/model.
        prompts: Dict of prompts for summarize, qa, etc.

    """

    pdf_parser: str = Field(default="tesseract")
    model_name: str = Field(default="ollama/gemma3:1b")
    api_keys: dict[str, str] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)


class UserSettingsResponse(UserSettingsBase):
    """Response schema for user settings, including id and user_id."""

    id: int
    user_id: int

    class Config:
        from_attributes = True
