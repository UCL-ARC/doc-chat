"""Utility functions for managing LLMs with Ollama."""

import asyncio
import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)


async def list_local_models() -> list[str]:
    """
    Get the list of models available locally in Ollama.

    Returns:
        A list of model names available locally.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_API_BASE_URL}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            # The name is often like 'llama3:latest', we just want the base name
            return [model["name"] for model in models]
    except httpx.RequestError as e:
        log.error(f"Error connecting to Ollama to list models: {e}")
        return []


async def pull_model(model_name: str) -> None:
    """
    Pull a model from the Ollama registry.

    Args:
        model_name: The name of the model to pull.
    """
    log.info(f"Model '{model_name}' not found locally. Pulling from registry...")
    try:
        # Set a long timeout for potentially large model downloads
        async with httpx.AsyncClient(timeout=300.0) as client:
            payload = {"name": model_name, "stream": False}
            response = await client.post(f"{settings.OLLAMA_API_BASE_URL}/api/pull", json=payload)
            response.raise_for_status()

            if response.json().get("status") == "success":
                log.info(f"Successfully pulled model '{model_name}'.")
            else:
                log.warning(f"Finished pulling '{model_name}' with status: {response.json().get('status')}")

    except httpx.RequestError as e:
        log.error(f"Error pulling model '{model_name}' from Ollama: {e}")
        raise  # Re-raise the exception to be handled by the caller
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error pulling model '{model_name}': {e.response.text}")
        raise


async def ensure_model_is_available(model_name: str) -> None:
    """
    Check if a model is available locally, and if not, pull it.

    Args:
        model_name: The name of the model to check.
    """
    # Ollama model names can be like 'llama3' or 'llama3:latest'
    # We'll check for the base name.
    # base_model_name = model_name.split(":")[0]
    local_models = await list_local_models()
    if model_name not in local_models:
        await pull_model(model_name)
