import logging
from contextlib import suppress
from collections.abc import AsyncIterator
from typing import AsyncGenerator

from litellm import completion
import litellm

from ..config import settings
from dotenv import load_dotenv

load_dotenv(override=True)

# Disable LiteLLM cost calculation and reduce its logging (no repeated "cost calculation" lines)
litellm.disable_end_user_cost_tracking = True
# Prevent LiteLLM from fetching model_prices_and_context_window.json from GitHub
litellm.model_cost_map_url = ""
# Return empty cost map so no HTTP request is made for model prices/context window
def _get_model_cost_map_no_fetch(*args: object, **kwargs: object) -> dict:
    return {}

litellm.get_model_cost_map = _get_model_cost_map_no_fetch
with suppress(AttributeError):
    litellm.litellm_core_utils.get_model_cost_map = _get_model_cost_map_no_fetch
_litellm_log = logging.getLogger("LiteLLM")
_litellm_log.setLevel(getattr(logging, settings.LITELLM_LOG_LEVEL.upper(), logging.ERROR))


def _infer_provider_from_model(model_name: str) -> str:
    """Infer provider from model name."""
    if model_name.lower().startswith("openai/"):
        return "openai"
    elif model_name.lower().startswith("azure/"):
        return "azure"
    elif model_name.lower().startswith("google/"):
        return "google"
    elif model_name.lower().startswith("ollama/"):
        return "ollama"
    raise ValueError(f"Unknown provider for model: {model_name}")


def _get_api_key(provider: str) -> str:
    if provider.lower() == "openai":
        return settings.OPENAI_API_KEY
    elif provider.lower().startswith("azure"):
        return settings.AZURE_API_KEY
    elif provider.lower() == "google":
        return settings.GOOGLE_API_KEY
    elif provider.lower() == "ollama":
        return None
    # Add more providers as needed
    raise ValueError(f"No API key configured for provider: {provider}")


async def summarize_text_with_llm_stream(
    text: str, model_name: str = "openai/gpt-4o-mini", api_key: str | None = None, **kwargs
) -> AsyncIterator[str]:
    """
    Summarize text using an LLM (OpenAI) with streaming response.

    Args:
        text: The text to summarize.
        model_name: Name of the LLM model to use.
        api_key: Optional API key for the model provider.
        **kwargs: Additional arguments to pass to the LLM completion call (e.g., api_base).

    Yields:
        Chunks of the summary text as they are generated.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = _get_api_key(provider)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Summarize the following text for efficacy and clarity.\n\n{text}"}
    ]
    response = completion(
        model=model_name,
        messages=messages,
        stream=True,
        api_key=api_key,
        **kwargs
    )
    for chunk in response:
        yield chunk.choices[0].delta.content


async def answer_question_with_llm_stream(
    text: str, question: str, model_name: str = "gpt-4o", api_key: str | None = None, **kwargs
) -> AsyncIterator[str]:
    """
    Answer a question about the text using an LLM (OpenAI) with streaming response.

    Args:
        text: The context text.
        question: The question to answer.
        model_name: Name of the LLM model to use.
        api_key: Optional API key for the model provider.
        **kwargs: Additional arguments to pass to the LLM completion call (e.g., api_base).

    Yields:
        Chunks of the answer text as they are generated.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = _get_api_key(provider)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about the provided text."},
        {"role": "user", "content": f"Given the following text, answer the question as accurately as possible.\n\nText:\n{text}\n\nQuestion: {question}"}
    ]
    response = completion(
        model=model_name,
        messages=messages,
        stream=True,
        api_key=api_key,
        **kwargs
    )
    for chunk in response:
        yield chunk.choices[0].delta.content


def summarize_text_with_llm(
    text: str, model_name: str = "gpt-4o", api_key: str | None = None, **kwargs
) -> str:
    """Summarize text using an LLM (OpenAI).

    Args:
        text: The text to summarize.
        model_name: Name of the LLM model to use.
        api_key: Optional API key for the model provider.
        **kwargs: Additional arguments to pass to the LLM completion call (e.g., api_base).

    Returns:
        The summary text.
    """
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = _get_api_key(provider)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Summarize the following text for efficacy and clarity.\n\n{text}"}
    ]
    response = completion(
        model=model_name,
        messages=messages,
        api_key=api_key,
        **kwargs
    )
    return response.choices[0].message.content


def answer_question_with_llm(
    text: str, question: str, model_name: str = "gpt-4o", api_key: str | None = None, **kwargs
) -> str:
    """Answer a question about the text using an LLM (OpenAI).

    Args:
        text: The context text.
        question: The question to answer.
        model_name: Name of the LLM model to use.
        api_key: Optional API key for the model provider.
        **kwargs: Additional arguments to pass to the LLM completion call (e.g., api_base).

    Returns:
        The answer text.
    """
    provider = _infer_provider_from_model(model_name)
    if api_key is None:
        api_key = _get_api_key(provider)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about the provided text."},
        {"role": "user", "content": f"Given the following text, answer the question as accurately as possible.\n\nText:\n{text}\n\nQuestion: {question}"}
    ]
    response = completion(
        model=model_name,
        messages=messages,
        api_key=api_key,
        **kwargs
    )
    return response.choices[0].message.content
