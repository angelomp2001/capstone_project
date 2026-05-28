import json
from typing import List, Dict, Any
import logging


import pandas as pd

from src.llm.llm_utils import (
    is_llm_available, 
    call_llm_for_json, 
    load_config_yml, 
    get_project_root
)
from src.llm.operations import SUPPORTED_OPS, get_ops_description

logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")

try:
    prompts_path = get_project_root() / "configs" / "llm_prompts.yml"
    PROMPTS = load_config_yml(str(prompts_path))
except Exception:
    PROMPTS = {}



def format_model_registry() -> str:
    """
    Format the MODEL_REGISTRY from src.registry into a readable string for the LLM prompt.
    """
    try:
        from src.registry import MODEL_REGISTRY
        lines = []
        for task_type, models in MODEL_REGISTRY.items():
            lines.append(f"{task_type}:")
            for model_name in sorted(models.keys()):
                lines.append(f"  - {model_name}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Failed to format model registry: %s", e)
        return ""


def llm_parses_to_ops(
    user_text: str,
    df: pd.DataFrame,
    *,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Use the LLM to parse a natural language cleaning instruction into
    a list of operation dicts compatible with `cleaning_operations.apply_operation`.

    Parameters
    ----------
    user_text : str
        The user's natural language cleaning instruction.
    df : pd.DataFrame
        The current DataFrame (used to expose column names and dtypes to the LLM).
    temperature : float, optional
        Sampling temperature for the LLM. Default: 0.0 (more deterministic).

    Returns
    -------
    List[Dict[str, Any]]
        A list of operation dictionaries. Each dict has the form:
        {
          "op": "<one of SUPPORTED_OPS>",
          "params": { ... }
        }

        Returns an empty list if parsing fails.
    """
    # info for the llm
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # prompt components
    system_prompt = PROMPTS.get("text_parser", {}).get("system_prompt", "")
    ops_description = get_ops_description()
    user_prompt_template = PROMPTS.get("text_parser", {}).get("user_prompt_template", "")

    # final user prompt to be sent to the llm
    user_prompt = user_prompt_template.format(
        dtypes_json=json.dumps(dtypes, indent=2),
        columns=columns,
        ops_description=ops_description,
        user_text=user_text,
        supported_ops=sorted(list(SUPPORTED_OPS)),
        model_registry=format_model_registry()
    )

    try:
        logger.info("Parsing instruction into operations. Text: %s", user_text)
        trace.info("PARSER START text=%r columns=%r dtypes=%r", user_text, columns, dtypes)
        
        if is_llm_available():
            parsed_json = call_llm_for_json(
                system_prompt,
                user_prompt,
                temperature=temperature
            )
            logger.info("LLM parser returned raw result: %s", parsed_json)
        else:
            msg = "LLM API is not available. Please configure it to parse instructions."
            logger.warning(msg)
            return [{"op": "error", "params": {"message": msg}}]
    except Exception as e:
        msg = f"LLM parsing failed: {str(e)}"
        logger.error(msg)
        trace.info("PARSER EXCEPTION error=%r", e)
        return [{"op": "error", "params": {"message": msg}}]

    # Normalize: ensure we always have a list
    if isinstance(parsed_json, dict):
        result = [parsed_json]
    elif isinstance(parsed_json, list):
        result = parsed_json
    else:
        result = []

    # Filter out malformed or unsupported ops
    cleaned_ops: List[Dict[str, Any]] = []
    for op in result:
        if not isinstance(op, dict):
            logger.warning("Skipping non-dict parser output: %s", op)
            continue
        op_type = op.get("op")
        params = op.get("params", {})
        if op_type not in SUPPORTED_OPS:
            logger.warning(f"Unsupported op from LLM: {op_type}")
            continue
        if not isinstance(params, dict):
            continue
        cleaned_ops.append({"op": op_type, "params": params})

    logger.info("Final parsed operations: %s", cleaned_ops)
    trace.info("PARSER FINAL ops=%r", cleaned_ops)
    return cleaned_ops
