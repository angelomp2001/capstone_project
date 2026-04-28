import logging
import sys
import yaml
from pathlib import Path
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "docs" / "logs"
APP_LOG_FILE = LOG_DIR / "app.log"
TRACE_LOG_FILE = LOG_DIR / "trace.log"


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up console and file logging for the project."""
    ensure_log_dir()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    app_file_handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    app_file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[stream_handler, app_file_handler],
        force=True,
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("Logging initialized. Main log file: %s", APP_LOG_FILE)


def append_trace(message: str) -> None:
    """Append verbose trace text to a plain text trace file."""
    ensure_log_dir()
    timestamp = datetime.now().isoformat(timespec="seconds")
    with TRACE_LOG_FILE.open("a", encoding="utf-8") as trace_file:
        trace_file.write(f"[{timestamp}] {message}\n")


def get_log_locations() -> Dict[str, str]:
    ensure_log_dir()
    return {
        "app_log": str(APP_LOG_FILE),
        "trace_log": str(TRACE_LOG_FILE),
    }



def load_config(config_path: str):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """Returns project root path."""
    return PROJECT_ROOT



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default Ollama HTTP endpoint
OLLAMA_URL_DEFAULT = "http://localhost:11434"

# Read configuration from environment variables (with defaults)
OLLAMA_URL = os.getenv("OLLAMA_URL", OLLAMA_URL_DEFAULT)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")  # change default model if you prefer
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

logger = logging.getLogger(__name__)


def is_ollama_available() -> bool:
    """Return True when the configured Ollama endpoint responds."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        logger.info("Ollama availability check returned status %s from %s", response.status_code, OLLAMA_URL)
        return response.status_code == 200
    except requests.RequestException:
        logger.info("Ollama availability check failed for %s", OLLAMA_URL)
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
        append_trace(f"LLM REQUEST model={model} temperature={temperature} system={system_prompt!r} user={user_prompt!r}")
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

    append_trace(f"LLM RESPONSE content={content!r}")
    return content


# -----------------------------------------------------------------------------
# Helper for JSON-structured responses (e.g., cleaning ops)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def call_llm_for_json_cached(system_prompt, user_prompt, temperature):
    logger.info("Using cached JSON LLM wrapper.")
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
        parsed = json.loads(text)
        append_trace(f"LLM JSON PARSED direct={parsed!r}")
        return parsed
    except json.JSONDecodeError:
        # Some models wrap JSON in markdown fences; try to extract.
        stripped = _extract_json_from_text(text)
        try:
            parsed = json.loads(stripped)
            append_trace(f"LLM JSON PARSED extracted={parsed!r}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error("RAW LLM OUTPUT:\n%s", text)
            append_trace(f"LLM JSON PARSE ERROR raw={text!r}")
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
