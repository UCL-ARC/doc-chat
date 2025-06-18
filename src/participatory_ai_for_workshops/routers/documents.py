"""Document router module."""

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import services
from ..auth.utils import verify_token
from ..database import async_session_factory, get_db
from ..models.conversation import Conversation, ConversationType, Message, MessageRole
from ..models.document import Document, ParsedDocument
from ..models.user import User, UserSettings
from ..schemas.document import DocumentResponse
from ..services.llm import (
    answer_question_with_llm,
    answer_question_with_llm_stream,
    summarize_text_with_llm_stream,
)
from ..services.pdf_parsers import (
    parse_image_with_llm_vision,
    parse_image_with_tesseract,
    parse_pdf_with_docling,
    parse_pdf_with_llm_vision,
    parse_pdf_with_tesseract,
)

router = APIRouter(prefix="/documents", tags=["documents"])

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user.

    Args:
        token: JWT token from Authorization header.
        db: Database session.

    Returns:
        User: Current user.

    Raises:
        HTTPException: If user not found or inactive.

    """
    payload = verify_token(token)
    result = await db.execute(select(User).where(User.email == payload["sub"]))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def parse_pdf_backend(
    file_path: str, method: str, model_name: str = None, api_key: str = None
) -> list[str]:
    """Unified PDF parsing dispatcher."""
    if method == "tesseract":
        return parse_pdf_with_tesseract(file_path)
    if method == "llm":
        return parse_pdf_with_llm_vision(
            file_path, model_name=model_name, api_key=api_key
        )
    if method == "docling":
        return parse_pdf_with_docling(file_path)
    raise ValueError(f"Unknown parsing method: {method}")


async def parse_document_background(
    document_id: int,
    user_id: int,
    method: str,
    file_path: str,
    api_key: str,
    model_name: str = None,
) -> None:
    """Background task to parse a document and update ParsedDocument row."""
    async with async_session_factory() as db:
        result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == document_id,
                ParsedDocument.user_id == user_id,
                ParsedDocument.method == method,
            )
        )
        parsed_doc = result.scalar_one_or_none()
        if not parsed_doc:
            return
        parsed_doc.parsing_status = "in_progress"
        await db.commit()
        try:
            if file_path.lower().endswith(".pdf"):
                parsed = parse_pdf_backend(
                    file_path, method, model_name=model_name, api_key=api_key
                )
            # For images, keep existing logic
            elif method == "tesseract":
                parsed = [parse_image_with_tesseract(file_path)]
            elif method == "llm":
                parsed = [
                    parse_image_with_llm_vision(
                        file_path, model_name=model_name, api_key=api_key
                    )
                ]
            elif method == "docling":
                # Docling is for PDFs, fallback to tesseract for images
                parsed = [parse_image_with_tesseract(file_path)]
            else:
                raise ValueError("Unknown parsing method")
            print(f"[parse_document_background] Parsed {file_path}")
            parsed_text = "\n".join(parsed)
            # print(f"[parse_document_background] Parsed text: {parsed_text}")
            parsed_doc.parsed_text = parsed_text
            parsed_doc.parsing_status = "done"
            parsed_doc.error_message = None
        except Exception as e:
            parsed_doc.parsing_status = "error"
            print(f"[parse_document_background] Error in parsing {file_path}: {e}")
            parsed_doc.error_message = str(e)
        await db.commit()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Upload a document.

    Args:
        file: File to upload.
        current_user: Current authenticated user.
        db: Database session.

    Returns:
        DocumentResponse: Uploaded document data.

    Raises:
        HTTPException: If file type not supported.

    """
    # Validate file type
    allowed_types = {"application/pdf", "image/jpeg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File type not supported"
        )

    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create document record
    document = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=file_path,
        file_type=file.content_type,
    )

    db.add(document)
    await db.commit()
    await db.refresh(document)

    # Get user settings for default parse method
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()
    method = user_settings.pdf_parser if user_settings else "tesseract"
    api_key = (
        user_settings.api_keys.get(user_settings.model_name) if user_settings else None
    )

    # Create ParsedDocument row (status pending)
    parsed_doc = ParsedDocument(
        document_id=document.id,
        user_id=current_user.id,
        method=method,
        parsing_status="pending",
    )
    db.add(parsed_doc)
    await db.commit()
    await db.refresh(parsed_doc)

    # Trigger background parsing
    background_tasks.add_task(
        parse_document_background,
        document.id,
        current_user.id,
        method,
        document.file_path,
        api_key,
        user_settings.model_name if user_settings else None,
    )

    return document


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Any:
    """List user's documents, including aggregate parsing status across all methods."""
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id)
    )
    documents = result.scalars().all()
    docs_with_status = []
    for doc in documents:
        parsed_results = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == doc.id,
                ParsedDocument.user_id == current_user.id,
            )
        )
        parsed_docs = parsed_results.scalars().all()
        if not parsed_docs:
            parsing_status = "not_started"
        elif any(pd.parsing_status == "done" for pd in parsed_docs):
            parsing_status = "done"
        elif any(pd.parsing_status == "in_progress" for pd in parsed_docs):
            parsing_status = "in_progress"
        elif any(pd.parsing_status == "pending" for pd in parsed_docs):
            parsing_status = "pending"
        elif all(pd.parsing_status == "error" for pd in parsed_docs):
            parsing_status = "error"
        else:
            parsing_status = "not_started"
        doc_dict = doc.__dict__.copy()
        doc_dict["parsing_status"] = parsing_status
        print(f"[list_documents] Document ID {doc.id} parsing_status: {parsing_status}")
        docs_with_status.append(doc_dict)
    return docs_with_status


@router.get("/status")
async def document_status(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Any:
    """Return parsing status for each document for the current user."""
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id)
    )
    documents = result.scalars().all()
    status_list = []
    for doc in documents:
        parsed_result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == doc.id,
                ParsedDocument.user_id == current_user.id,
            )
        )
        parsed_doc = parsed_result.scalar_one_or_none()
        if parsed_doc:
            status_list.append(
                {
                    "document_id": doc.id,
                    "parsing_status": parsed_doc.parsing_status,
                    "error_message": parsed_doc.error_message,
                    "method": parsed_doc.method,
                }
            )
        else:
            status_list.append(
                {
                    "document_id": doc.id,
                    "parsing_status": "not_started",
                    "error_message": None,
                    "method": None,
                }
            )
    return status_list


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get document details.

    Args:
        document_id: Document ID.
        current_user: Current authenticated user.
        db: Database session.

    Returns:
        DocumentResponse: Document data.

    Raises:
        HTTPException: If document not found or not owned by user.

    """
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    return document


async def wait_for_parsed_document(
    document_id: int, user_id: int, method: str, db: AsyncSession, timeout: int = 30
) -> ParsedDocument:
    """Wait for a ParsedDocument to be done or error, with timeout."""
    waited = 0
    poll_interval = 1
    while waited < timeout:
        result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == document_id,
                ParsedDocument.user_id == user_id,
                ParsedDocument.method == method,
            )
        )
        parsed_doc = result.scalar_one_or_none()
        if parsed_doc and parsed_doc.parsing_status in ("done", "error"):
            return parsed_doc
        await asyncio.sleep(poll_interval)
        waited += poll_interval
    return None


@router.post("/parse/pdf")
async def parse_pdf(
    document_id: int = Query(..., description="ID of the uploaded document"),
    method: str = Query(
        None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"
    ),
    model_name: str = Query(
        None, description="LLM model name (if method is 'llm', default: user setting)"
    ),
    provider: str = Query(
        None, description="LLM provider (if method is 'llm', default: user setting)"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Parse a PDF using Tesseract OCR or a multimodal LLM (OpenAI, Gemini, etc.), using user settings by default."""
    # Fetch user settings
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)
    # Use settings unless overridden
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    # Determine per-user API key for the selected model
    api_key = user_settings.api_keys.get(model_name)
    # Get document
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.file_type.startswith("application/pdf"):
        raise HTTPException(status_code=400, detail="Not a PDF document")
    # Check cache
    result = await db.execute(
        select(ParsedDocument).where(
            ParsedDocument.document_id == document_id,
            ParsedDocument.user_id == current_user.id,
            ParsedDocument.method == method,
        )
    )
    parsed_doc = result.scalar_one_or_none()
    if parsed_doc:
        if parsed_doc.parsing_status == "done":
            return {
                "document_id": document_id,
                "method": method,
                "content": parsed_doc.parsed_text,
            }
        if parsed_doc.parsing_status in ("pending", "in_progress"):
            parsed_doc = await wait_for_parsed_document(
                document_id, current_user.id, method, db
            )
            if parsed_doc and parsed_doc.parsing_status == "done":
                return {
                    "document_id": document_id,
                    "method": method,
                    "content": parsed_doc.parsed_text,
                }
            if parsed_doc and parsed_doc.parsing_status == "error":
                raise HTTPException(
                    status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
                )
            raise HTTPException(status_code=504, detail="Parsing timed out.")
        if parsed_doc.parsing_status == "error":
            raise HTTPException(
                status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
            )
    # If not found, parse now (fallback)
    if method == "tesseract":
        content = parse_pdf_with_tesseract(document.file_path)
    elif method == "llm":
        content = parse_pdf_with_llm_vision(
            document.file_path, model_name=model_name, api_key=api_key
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown parsing method")
    return {"document_id": document_id, "method": method, "content": content}


@router.post("/parse/image")
async def parse_image(
    document_id: int = Query(..., description="ID of the uploaded image"),
    method: str = Query(
        None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"
    ),
    model_name: str = Query(
        None, description="LLM model name (if method is 'llm', default: user setting)"
    ),
    provider: str = Query(
        None, description="LLM provider (if method is 'llm', default: user setting)"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Parse an image using Tesseract OCR or a multimodal LLM (OpenAI, Gemini, etc.), using user settings by default."""
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    # Determine per-user API key for the selected model
    api_key = user_settings.api_keys.get(model_name)
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if not document.file_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image document")
    # Check cache
    result = await db.execute(
        select(ParsedDocument).where(
            ParsedDocument.document_id == document_id,
            ParsedDocument.user_id == current_user.id,
            ParsedDocument.method == method,
        )
    )
    parsed_doc = result.scalar_one_or_none()
    if parsed_doc:
        if parsed_doc.parsing_status == "done":
            return {
                "document_id": document_id,
                "method": method,
                "content": parsed_doc.parsed_text,
            }
        if parsed_doc.parsing_status in ("pending", "in_progress"):
            parsed_doc = await wait_for_parsed_document(
                document_id, current_user.id, method, db
            )
            if parsed_doc and parsed_doc.parsing_status == "done":
                return {
                    "document_id": document_id,
                    "method": method,
                    "content": parsed_doc.parsed_text,
                }
            if parsed_doc and parsed_doc.parsing_status == "error":
                raise HTTPException(
                    status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
                )
            raise HTTPException(status_code=504, detail="Parsing timed out.")
        if parsed_doc.parsing_status == "error":
            raise HTTPException(
                status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
            )
    # If not found, parse now (fallback)
    if method == "tesseract":
        content = parse_image_with_tesseract(document.file_path)
    elif method == "llm":
        content = parse_image_with_llm_vision(
            document.file_path, model_name=model_name, api_key=api_key
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown parsing method")
    return {"document_id": document_id, "method": method, "content": content}


class DocumentIdsRequest(BaseModel):
    document_ids: list[int]


class QARequest(DocumentIdsRequest):
    question: str


@router.post("/llm/summarize")
async def summarize_documents(
    request: DocumentIdsRequest,
    method: str = Query(
        None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"
    ),
    model_name: str = Query(None, description="LLM model name (default: user setting)"),
    provider: str = Query(None, description="LLM provider (default: user setting)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    api_key = user_settings.api_keys.get(model_name)
    prompt = user_settings.prompts.get("summarize", None)
    document_ids = request.document_ids
    texts = []
    for doc_id in document_ids:
        result = await db.execute(
            select(Document).where(
                Document.id == doc_id, Document.user_id == current_user.id
            )
        )
        document = result.scalar_one_or_none()
        if not document:
            continue
        # Check parsed cache (independent of model)
        result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == doc_id,
                ParsedDocument.user_id == current_user.id,
                ParsedDocument.method == method,
            )
        )
        parsed_doc = result.scalar_one_or_none()
        if parsed_doc:
            if parsed_doc.parsing_status == "done":
                texts.append(parsed_doc.parsed_text)
                continue
            if parsed_doc.parsing_status in ("pending", "in_progress"):
                parsed_doc = await wait_for_parsed_document(
                    doc_id, current_user.id, method, db
                )
                if parsed_doc and parsed_doc.parsing_status == "done":
                    texts.append(parsed_doc.parsed_text)
                    continue
                if parsed_doc and parsed_doc.parsing_status == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Parsing error: {parsed_doc.error_message}",
                    )
                raise HTTPException(status_code=504, detail="Parsing timed out.")
            if parsed_doc.parsing_status == "error":
                raise HTTPException(
                    status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
                )
        continue
    full_text = "\n".join(texts)
    if not full_text:
        raise HTTPException(
            status_code=404, detail="No parsed text available for selected documents."
        )
    summary = (
        services.summarize_text_with_llm(
            full_text, model_name=model_name, api_key=api_key
        )
        if not prompt
        else services.summarize_text_with_llm(
            full_text, model_name=model_name, api_key=api_key
        )
    )
    return {"summary": summary}


@router.post("/llm/qa")
async def qa_documents(
    request: QARequest,
    method: str = Query(
        None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"
    ),
    model_name: str = Query(None, description="LLM model name (default: user setting)"),
    provider: str = Query(None, description="LLM provider (default: user setting)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    result = await db.execute(
        select(UserSettings).where(UserSettings.user_id == current_user.id)
    )
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=current_user.id)
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    api_key = user_settings.api_keys.get(model_name)
    prompt = user_settings.prompts.get("qa", None)
    document_ids = request.document_ids
    question = request.question
    texts = []
    for doc_id in document_ids:
        result = await db.execute(
            select(Document).where(
                Document.id == doc_id, Document.user_id == current_user.id
            )
        )
        document = result.scalar_one_or_none()
        if not document:
            continue
        # Check parsed cache (independent of model)
        result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == doc_id,
                ParsedDocument.user_id == current_user.id,
                ParsedDocument.method == method,
            )
        )
        parsed_doc = result.scalar_one_or_none()
        if parsed_doc:
            if parsed_doc.parsing_status == "done":
                texts.append(parsed_doc.parsed_text)
                continue
            if parsed_doc.parsing_status in ("pending", "in_progress"):
                parsed_doc = await wait_for_parsed_document(
                    doc_id, current_user.id, method, db
                )
                if parsed_doc and parsed_doc.parsing_status == "done":
                    texts.append(parsed_doc.parsed_text)
                    continue
                if parsed_doc and parsed_doc.parsing_status == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Parsing error: {parsed_doc.error_message}",
                    )
                raise HTTPException(status_code=504, detail="Parsing timed out.")
            if parsed_doc.parsing_status == "error":
                raise HTTPException(
                    status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
                )
        continue
    full_text = "\n".join(texts)
    if not full_text:
        raise HTTPException(
            status_code=404, detail="No parsed text available for selected documents."
        )
    answer = (
        answer_question_with_llm(
            full_text, question, model_name=model_name, api_key=api_key
        )
        if not prompt
        else answer_question_with_llm(
            full_text, question, model_name=model_name, api_key=api_key
        )
    )
    return {"answer": answer}


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and its parsed cache."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    # Delete parsed cache
    await db.execute(
        ParsedDocument.__table__.delete().where(
            ParsedDocument.document_id == document_id,
            ParsedDocument.user_id == current_user.id,
        )
    )
    await db.delete(document)
    await db.commit()


async def get_or_create_conversation(
    db: AsyncSession, user_id: int, conversation_id: int, conv_type: ConversationType, title: str, metadata: dict
) -> Conversation:
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.type == conv_type,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            user_id=user_id,
            type=conv_type,
            title=title,
            meta_data=metadata,
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
    return conversation


async def get_or_create_user_settings(db: AsyncSession, user_id: int) -> UserSettings:
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == user_id))
    user_settings = result.scalar_one_or_none()
    if not user_settings:
        user_settings = UserSettings(user_id=user_id)
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)
    return user_settings


async def get_concatenated_document_texts(
    db: AsyncSession, user_id: int, document_ids: list[int], method: str
) -> str:
    texts = []
    for doc_id in document_ids:
        result = await db.execute(
            select(Document).where(Document.id == doc_id, Document.user_id == user_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            continue
        result = await db.execute(
            select(ParsedDocument).where(
                ParsedDocument.document_id == doc_id,
                ParsedDocument.user_id == user_id,
                ParsedDocument.method == method,
            )
        )
        parsed_doc = result.scalar_one_or_none()
        if parsed_doc:
            if parsed_doc.parsing_status == "done":
                texts.append(parsed_doc.parsed_text)
                continue
            if parsed_doc.parsing_status in ("pending", "in_progress"):
                parsed_doc = await wait_for_parsed_document(
                    doc_id, user_id, method, db
                )
                if parsed_doc and parsed_doc.parsing_status == "done":
                    texts.append(parsed_doc.parsed_text)
                    continue
                if parsed_doc and parsed_doc.parsing_status == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Parsing error: {parsed_doc.error_message}",
                    )
                raise HTTPException(status_code=504, detail="Parsing timed out.")
            if parsed_doc.parsing_status == "error":
                raise HTTPException(
                    status_code=500, detail=f"Parsing error: {parsed_doc.error_message}"
                )
    return "\n".join(texts)


async def add_user_message_to_conversation(
    db: AsyncSession, conversation_id: int, role: MessageRole, content: str, metadata: dict
):
    user_message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        meta_data=metadata,
    )
    db.add(user_message)
    await db.commit()


@router.post("/llm/summarize/stream")
async def summarize_documents_stream(
    request: DocumentIdsRequest,
    conversation_id: int = Query(None, description="Optional conversation ID to continue"),
    method: str = Query(None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"),
    model_name: str = Query(None, description="LLM model name (default: user setting)"),
    provider: str = Query(None, description="LLM provider (default: user setting)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream a summary of the selected documents using an LLM."""
    conversation = await get_or_create_conversation(
        db,
        current_user.id,
        conversation_id,
        ConversationType.SUMMARIZE,
        f"Summary of {len(request.document_ids)} documents",
        {"document_ids": request.document_ids},
    )
    user_settings = await get_or_create_user_settings(db, current_user.id)
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    api_key = user_settings.api_keys.get(model_name)
    prompt = user_settings.prompts.get("summarize", None)
    full_text = await get_concatenated_document_texts(db, current_user.id, request.document_ids, method)
    if not full_text:
        raise HTTPException(
            status_code=404, detail="No parsed text available for selected documents."
        )
    await add_user_message_to_conversation(
        db,
        conversation.id,
        MessageRole.SYSTEM,
        f"Please summarize the following documents: {', '.join([str(id) for id in request.document_ids])}",
        {"document_ids": request.document_ids},
    )
    async def stream_response():
        accumulated_text = ""
        async for chunk in summarize_text_with_llm_stream(
            full_text, model_name=model_name, api_key=api_key
        ):
            if chunk:
                accumulated_text += chunk
                yield f"data: {{\"text\": {json.dumps(chunk)}, \"conversation_id\": {conversation.id}}}\n\n"
        
        # Add the assistant's summary to the conversation history
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=accumulated_text,
            meta_data={"document_ids": request.document_ids},
        )
        async with async_session_factory() as session:
            session.add(assistant_message)
            await session.commit()
    return StreamingResponse(stream_response(), media_type="text/event-stream")


@router.post("/llm/qa/stream")
async def qa_documents_stream(
    request: QARequest,
    conversation_id: int = Query(None, description="Optional conversation ID to continue"),
    method: str = Query(None, description="Parsing method: 'tesseract' or 'llm' (default: user setting)"),
    model_name: str = Query(None, description="LLM model name (default: user setting)"),
    provider: str = Query(None, description="LLM provider (default: user setting)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream an answer to a question about the selected documents using an LLM."""
    conversation = await get_or_create_conversation(
        db,
        current_user.id,
        conversation_id,
        ConversationType.QA,
        f"Q&A about {len(request.document_ids)} documents",
        {"document_ids": request.document_ids},
    )
    user_settings = await get_or_create_user_settings(db, current_user.id)
    method = method or user_settings.pdf_parser
    model_name = model_name or user_settings.model_name or "gpt-4o"
    api_key = user_settings.api_keys.get(model_name)
    prompt = user_settings.prompts.get("qa", None)
    full_text = await get_concatenated_document_texts(db, current_user.id, request.document_ids, method)
    if not full_text:
        raise HTTPException(
            status_code=404, detail="No parsed text available for selected documents."
        )
    await add_user_message_to_conversation(
        db,
        conversation.id,
        MessageRole.USER,
        request.question,
        {"document_ids": request.document_ids},
    )
    async def stream_response():
        accumulated_text = ""
        async for chunk in answer_question_with_llm_stream(
            full_text, request.question, model_name=model_name, api_key=api_key
        ):
            if chunk:
                accumulated_text += chunk
                yield f"data: {{\"text\": {json.dumps(chunk)}, \"conversation_id\": {conversation.id}}}\n\n"
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=accumulated_text,
            meta_data={"document_ids": request.document_ids},
        )
        async with async_session_factory() as session:
            session.add(assistant_message)
            await session.commit()
    return StreamingResponse(stream_response(), media_type="text/event-stream")
