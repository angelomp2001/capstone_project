import logging
import sys
import yaml
from pathlib import Path
import os
import json
from typing import List, Dict, Any, Optional
import streamlit as st
import requests


def setup_logging(log_level=logging.INFO, log_file='docs/logs/logger.txt'):
    """Set up logging for the project."""

    # Create a list to hold the logging handlers.
    handlers = [logging.StreamHandler(sys.stdout)]  # Log to console
    
    # Create the directory for the log file if it does not already exist.
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Append a FileHandler to log messages to the specified log file.
    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)

    # Configure the logging system.
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )



def load_config(config_path: str):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """Returns project root path."""
    return Path(__file__).parent.parent



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default Ollama HTTP endpoint
OLLAMA_URL_DEFAULT = "http://localhost:11434"

# Read configuration from environment variables (with defaults)
OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_URL_DEFAULT)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")  # change default model if you prefer
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

# Basic logger setup (you can customize this in your main app)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def is_ollama_available() -> bool:
    """Return True when the configured Ollama endpoint responds."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# -----------------------------------------------------------------------------
# Core LLM call
# -----------------------------------------------------------------------------


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Call an Ollama chat model with a system + user prompt and return the final text.

    Parameters
    ----------
    system_prompt : str
        Instruction describing the assistant's role and behavior.
    user_prompt : str
        The actual user query or content to process.
    model : str, optional
        Ollama model name. If None, OLLAMA_MODEL env or default is used.
    temperature : float, optional
        Sampling temperature (0.0 = deterministic). Default: 0.0.
    max_tokens : int, optional
        Not directly supported as 'num_predict' in Ollama; if provided, it will
        be passed as 'num_predict'.

    Returns
    -------
    str
        The assistant's reply as plain text.

    Raises
    ------
    RuntimeError
        If the request to Ollama fails or returns a non-200 status.
    """
    if model is None:
        model = OLLAMA_MODEL

    url = f"{OLLAMA_URL}/v1/chat/completions"

    # Ollama's API is OpenAI-compatible for this endpoint.
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    # Map max_tokens to Ollama's num_predict if provided
    if max_tokens is not None:
        payload["num_predict"] = max_tokens

    try:
        logger.debug("Sending request to Ollama at %s with model '%s'", OLLAMA_URL, model)
        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as e:
        msg = f"Error connecting to Ollama at {OLLAMA_URL}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    if response.status_code != 200:
        msg = (
            f"Ollama returned non-200 status code {response.status_code}: "
            f"{response.text}"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        msg = f"Failed to parse JSON from Ollama response: {e}. Raw text: {response.text[:500]}"
        logger.error(msg)
        raise RuntimeError(msg) from e

    # OpenAI-compatible structure: choices[0].message.content
    choices = data.get("choices")
    if not choices:
        msg = f"No 'choices' field in Ollama response: {data}"
        logger.error(msg)
        raise RuntimeError(msg)

    content = choices[0].get("message", {}).get("content")
    if content is None:
        msg = f"No 'content' in Ollama response message: {choices[0]}"
        logger.error(msg)
        raise RuntimeError(msg)

    return content


# -----------------------------------------------------------------------------
# Helper for JSON-structured responses (e.g., cleaning ops)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def call_llm_for_json_cached(system_prompt, user_prompt, temperature):
    return call_llm_for_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )

def call_llm_for_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Call the LLM and parse the response as JSON.

    This is useful when you instruct the model to output a JSON object or array,
    e.g., for data cleaning operations.

    Parameters
    ----------
    system_prompt : str
        Instruction describing the assistant's role and behavior.
    user_prompt : str
        The actual user query or content to process.
    model : str, optional
        Ollama model name. If None, OLLAMA_MODEL is used.
    temperature : float, optional
        Sampling temperature.

    Returns
    -------
    Any
        Parsed JSON structure (list/dict/primitive) on success.

    Raises
    ------
    ValueError
        If the response cannot be parsed as JSON.
    RuntimeError
        If the underlying LLM call fails.
    """
    text = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some models wrap JSON in markdown fences; try to extract.
        stripped = _extract_json_from_text(text)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.error("RAW LLM OUTPUT:\n%s", text)
            raise ValueError("LLM output is not valid JSON") from e


def _extract_json_from_text(text: str) -> str:
    """
    Extract a JSON substring from text that might contain code fences or extra text.

    This is a best-effort helper. For robust behavior, always instruct the model
    to return raw JSON with no commentary or code fences.
    """
    text = text.strip()

    # Identify and strip common markdown code block wrappers
    if text.startswith("```"):
        # Remove leading ```<lang> (if any) and trailing ```
        lines = text.splitlines()
        # Remove the first line (``` or ```json etc.)
        if lines:
            lines = lines[1:]
        # Remove final ``` line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to locate first '{' or '[' and last '}' or ']'
    start_candidates = [text.find("{"), text.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return text  # nothing better to do
    start = min(start_candidates)

    end_brace = text.rfind("}")
    end_bracket = text.rfind("]")
    end_candidates = [i for i in [end_brace, end_bracket] if i != -1]
    if not end_candidates:
        return text
    end = max(end_candidates) + 1

    return text[start:end]
