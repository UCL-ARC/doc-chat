import base64

import pytesseract
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from PIL import Image

from ..config import settings
from .llm import _infer_provider_from_model

# Import your existing tesseract and llm vision parsers here, or define stubs for now


def parse_pdf_with_tesseract(file_path: str) -> list[str]:
    """Parse a PDF file using Tesseract OCR and return a list of text chunks."""
    print(f"[parse_pdf_with_tesseract] Parsing file {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        pages = []
        for page in loader.load():
            pages.append(page.page_content)
        # docs = loader.load()
        # return [doc.page_content for doc in docs]
        return pages
    except Exception as e:
        print(f"[parse_pdf_with_tesseract] Skipping file {file_path} due to error: {e}")
        return []


async def parse_pdf_with_llm_vision(
    file_path: str, model_name: str = "gpt-4o", api_key: str = None
) -> list[str]:
    """Parse a PDF file using a multimodal LLM and return a list of text chunks."""
    print(f"[parse_pdf_with_llm_vision] Parsing file {file_path}")
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = (
            settings.OPENAI_API_KEY if provider == "openai" else settings.GOOGLE_API_KEY
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name, api_key=api_key)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    loader = PyPDFLoader(
        file_path,
        mode="page",
        images_inner_format="markdown-img",
        images_parser=LLMImageBlobParser(model=model),
    )
    docs = loader.load()
    return [doc.page_content for doc in docs]


def parse_image_with_tesseract(image_path: str) -> str:
    """Parse an image file using Tesseract OCR."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def parse_image_with_llm_vision(
    image_path: str, model_name: str = "gpt-4o", api_key: str = None
) -> str:
    """Parse an image file using a multimodal LLM (OpenAI, Gemini, etc.)."""
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = (
            settings.OPENAI_API_KEY if provider == "openai" else settings.GOOGLE_API_KEY
        )
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name, api_key=api_key)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    from langchain_core.messages import HumanMessage

    message = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
        )
    ]
    response = model.invoke(message)
    return getattr(response, "content", str(response))


def parse_pdf_with_docling(file_path: str) -> list[str]:
    """Parse a PDF file using Docling and return a list of text chunks (markdown)."""
    print(f"[parse_pdf_with_docling] Parsing file {file_path}")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()
    return [markdown]
