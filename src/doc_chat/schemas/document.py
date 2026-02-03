"""Document schemas module."""

from datetime import datetime

from pydantic import BaseModel


class DocumentBase(BaseModel):
    """
    Base document schema.

    Attributes:
        filename: Original filename.
        file_type: Type of the file.

    """

    filename: str
    file_type: str


class DocumentCreate(DocumentBase):
    """Document creation schema."""


class DocumentResponse(DocumentBase):
    """
    Document response schema.

    Attributes:
        id: Document ID.
        user_id: ID of the user who uploaded the document.
        file_path: Path where the file is stored.
        summary: Generated summary of the document.
        faqs: Generated FAQs from the document.
        created_at: When the document was uploaded.
        updated_at: When the document was last updated.
        parsing_status: Aggregate parsing status: not_started, pending, in_progress, done, error.

    """

    id: int
    user_id: int
    file_path: str
    summary: str | None = None
    faqs: str | None = None
    created_at: datetime
    updated_at: datetime
    parsing_status: str = "not_started"

    class Config:
        """Pydantic model configuration."""

        from_attributes = True
