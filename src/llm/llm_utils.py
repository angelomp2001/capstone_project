import logging
import sys
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

import json
from typing import List, Dict, Any, Optional
import streamlit as st
import requests
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "docs" / "logs"
APP_LOG_FILE = LOG_DIR / "app.log"
TRACE_LOG_FILE = LOG_DIR / "trace.log"


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up console and file logging for the project."""
    ensure_log_dir()

    # formatters
    app_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
    )
    trace_formatter = logging.Formatter("%(asctime)s | %(message)s")

    # Set up a stream handler for logging messages to the console (stdout).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(app_formatter) # Apply the application log formatter to the stream handler.

    app_file_handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    app_file_handler.setFormatter(app_formatter)

    trace_file_handler = logging.FileHandler(TRACE_LOG_FILE, encoding="utf-8")
    trace_file_handler.setFormatter(trace_formatter)

    # config for root logger, which will be inherited by logger and trace
    logging.basicConfig(
        level=log_level,
        handlers=[stream_handler, app_file_handler],
        force=True,
    )
    # root logger defaults double as the app logger, so only trace-specific config is needed below
    trace_logger = logging.getLogger("trace")
    trace_logger.handlers.clear()
    trace_logger.addHandler(trace_file_handler)
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False

    # Set higher logging levels for specific libraries to minimize log clutter.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)

    # Log an informational message indicating that logging has been initialized.
    logging.getLogger(__name__).info("Logging initialized. Main log file: %s", APP_LOG_FILE)
    trace_logger.info("TRACE LOGGER INITIALIZED file=%s", TRACE_LOG_FILE)


def get_log_locations() -> Dict[str, str]:
    ensure_log_dir()
    return {
        "app_log": str(APP_LOG_FILE),
        "trace_log": str(TRACE_LOG_FILE),
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """Returns project root path."""
    return PROJECT_ROOT


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config_path = PROJECT_ROOT / "configs" / "llm_utils.yml"
config = load_config(str(config_path)) if config_path.exists() else {}

# Read configuration from environment variables (with defaults from config file)
OLLAMA_URL = os.getenv("OLLAMA_URL", config.get("OLLAMA_URL", "https://api.tokenfactory.nebius.com/v1/"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", config.get("OLLAMA_MODEL"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", config.get("OLLAMA_TIMEOUT", 120.0)))

# set loggers
logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")


def is_llm_available() -> bool:
    """Return True when the configured LLM endpoint responds or NEBIUS_API_KEY is set."""
    if "nebius.com" in OLLAMA_URL: # or os.getenv("NEBIUS_API_KEY"):
        return True
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        logger.info("LLM availability check returned status %s from %s", response.status_code, OLLAMA_URL)
        return response.status_code == 200 # is true?
    except requests.RequestException:
        logger.info("LLM availability check failed for %s", OLLAMA_URL)
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

    if not model:
        raise RuntimeError(
            "LLM model is not configured. Set OLLAMA_MODEL in configs/llm_utils.yml or via the OLLAMA_MODEL environment variable."
        )

    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
    
    if NEBIUS_API_KEY:
        client = OpenAI(
            base_url=OLLAMA_URL,
            api_key=NEBIUS_API_KEY,
        )
    else:
        client = OpenAI(
            base_url=OLLAMA_URL if OLLAMA_URL.endswith("/v1/") else f"{OLLAMA_URL}/v1/",
            api_key="ollama", # required but ignored
        )

    try:
        logger.debug("Sending request to llm with model '%s'", model)
        trace.info("LLM REQUEST model=%r temperature=%r system=%r user=%r", model, temperature, system_prompt, user_prompt)
        
        extra_args = {}
        if max_tokens is not None:
            extra_args["max_tokens"] = max_tokens

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                },
            ],
            temperature=temperature,
            **extra_args
        )
        content = response.choices[0].message.content
        trace.info("LLM RESPONSE content=%r", content)
        return content
    except Exception as e:
        msg = f"Error connecting to LLM: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# -----------------------------------------------------------------------------
# Helper for JSON-structured responses (e.g., cleaning ops)
# -----------------------------------------------------------------------------

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

def call_llm_for_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Call the LLM and parse the response as JSON.

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
    llm_response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )

    # Try direct JSON parsing first
    try:
        parsed_json = json.loads(llm_response)
        trace.info("LLM JSON PARSED direct=%r", parsed_json)
        return parsed_json
    except json.JSONDecodeError:

        # Some models wrap JSON in markdown fences; try to extract.
        stripped = _extract_json_from_text(llm_response)
        try:
            # Final attempt to parse the extracted JSON substring
            parsed_json = json.loads(stripped)
            trace.info("LLM JSON PARSED extracted=%r", parsed_json)
            return parsed_json
        except json.JSONDecodeError as e:
            # Log the raw output for debugging
            trace.info("LLM JSON PARSE ERROR raw=%r", llm_response)
            logger.error("RAW LLM OUTPUT:\n%s", llm_response)
            raise ValueError("LLM output is not valid JSON") from e


