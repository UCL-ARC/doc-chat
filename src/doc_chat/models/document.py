"""Document model module."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class Document(Base):
    """
    Document model for file uploads and analysis.

    Attributes:
        id: Primary key.
        user_id: Foreign key to the user who uploaded the document.
        filename: Original filename.
        file_path: Path where the file is stored.
        file_type: Type of the file (PDF, image, etc.).
        summary: Generated summary of the document.
        faqs: Generated FAQs from the document.
        created_at: When the document was uploaded.
        updated_at: When the document was last updated.

    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    faqs: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    user = relationship("User", back_populates="documents")


class ParsedDocument(Base):
    """
    ParsedDocument model for caching parsed text per document, user, and method.

    Attributes:
        id: Primary key.
        document_id: Foreign key to Document.
        user_id: Foreign key to User.
        method: Parsing method (e.g., 'tesseract', 'llm').
        parsed_text: Parsed text (markdown or plain text).
        parsing_status: Status ('pending', 'in_progress', 'done', 'error').
        error_message: Error message if parsing failed.
        created_at: When the record was created.
        updated_at: When the record was last updated.

    """

    __tablename__ = "parsed_document"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    method: Mapped[str] = mapped_column(String(32))
    parsed_text: Mapped[str] = mapped_column(Text, nullable=True)
    parsing_status: Mapped[str] = mapped_column(String(16), default="pending")
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    # model_name is kept for backward compatibility but not used for cache key
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # __table_args__ = (UniqueConstraint('document_id', 'user_id', 'method'),)  # enforce uniqueness
