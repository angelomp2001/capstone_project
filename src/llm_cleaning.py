import json
from typing import List, Dict, Any
import logging
import re
from difflib import get_close_matches


import pandas as pd

from .llm_utils import is_ollama_available
from .llm_utils import call_llm_for_json_cached, append_trace
from .cleaning_operations import SUPPORTED_OPS

logger = logging.getLogger(__name__)


def _coerce_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _match_column(candidate: str, columns: List[str]) -> str | None:
    if not candidate:
        return None

    normalized_candidate = _normalize_text(candidate)
    normalized_map = {_normalize_text(column): column for column in columns}

    if normalized_candidate in normalized_map:
        return normalized_map[normalized_candidate]

    for normalized_column, column in normalized_map.items():
        if normalized_candidate and normalized_candidate in normalized_column:
            return column

    matches = get_close_matches(
        normalized_candidate,
        list(normalized_map.keys()),
        n=1,
        cutoff=0.6,
    )
    return normalized_map[matches[0]] if matches else None


def _extract_columns_from_text(user_text: str, columns: List[str]) -> List[str]:
    found_columns: List[str] = []
    text_lower = user_text.lower()

    for column in columns:
        if column.lower() in text_lower:
            found_columns.append(column)

    if found_columns:
        return found_columns

    split_candidates = re.split(r",| and ", user_text)
    for candidate in split_candidates:
        match = _match_column(candidate.strip(), columns)
        if match and match not in found_columns:
            found_columns.append(match)

    return found_columns


def _heuristic_parse_instruction_to_ops(
    user_text: str,
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    columns = df.columns.tolist()
    text_lower = user_text.lower().strip()
    logger.info("Heuristic parser evaluating user text: %s", user_text)
    append_trace(f"HEURISTIC INPUT columns={columns!r} text={user_text!r}")

    if not text_lower:
        logger.info("Heuristic parser received empty input.")
        return []

    replace_match = re.search(
        r"replace\s+['\"]?(.+?)['\"]?(?:\s+category)?\s+(?:in|within)\s+([a-zA-Z0-9_ ]+?)\s+with\s+['\"]?([^,'\"]+)['\"]?",
        user_text,
        re.IGNORECASE,
    )
    if replace_match:
        old_value = replace_match.group(1).strip()
        target_column = _match_column(replace_match.group(2).strip(), columns)
        new_value = replace_match.group(3).strip()
        new_value = _coerce_literal(new_value)

        if target_column:
            ops = [{
                "op": "replace_value",
                "params": {
                    "column": target_column,
                    "old_value": old_value,
                    "new_value": new_value,
                },
            }]
            logger.info("Heuristic parser matched replace_value operation: %s", ops)
            append_trace(f"HEURISTIC OUTPUT replace_value={ops!r}")
            return ops

    if "drop column" in text_lower or "remove column" in text_lower:
        columns_to_drop = _extract_columns_from_text(user_text, columns)
        if columns_to_drop:
            ops = [{"op": "drop_columns", "params": {"columns": columns_to_drop}}]
            logger.info("Heuristic parser matched drop_columns: %s", ops)
            append_trace(f"HEURISTIC OUTPUT drop_columns={ops!r}")
            return ops

    if "drop rows" in text_lower or "drop missing" in text_lower or "drop null" in text_lower:
        subset = _extract_columns_from_text(user_text, columns)
        ops = [{
            "op": "dropna",
            "params": {"axis": 0, "subset": subset or None},
        }]
        logger.info("Heuristic parser matched dropna: %s", ops)
        append_trace(f"HEURISTIC OUTPUT dropna={ops!r}")
        return ops

    if "fill" in text_lower and ("missing" in text_lower or "null" in text_lower or "na" in text_lower):
        strategy = "constant"
        for candidate_strategy in ("median", "mean", "mode"):
            if candidate_strategy in text_lower:
                strategy = candidate_strategy
                break

        value = None
        constant_match = re.search(r"with\s+['\"]?([^,'\"]+)['\"]?$", user_text.strip(), re.IGNORECASE)
        if strategy == "constant" and constant_match:
            value = constant_match.group(1).strip()
            value = _coerce_literal(value)

        target_column = None
        fill_match = re.search(
            r"fill(?:\s+missing)?\s+([a-zA-Z0-9_ ]+?)\s+with",
            user_text,
            re.IGNORECASE,
        )
        if fill_match:
            target_column = _match_column(fill_match.group(1).strip(), columns)
        if target_column is None:
            extracted_columns = _extract_columns_from_text(user_text, columns)
            target_column = extracted_columns[0] if extracted_columns else None

        if target_column:
            ops = [{
                "op": "fillna",
                "params": {
                    "column": target_column,
                    "strategy": strategy,
                    "value": value,
                },
            }]
            logger.info("Heuristic parser matched fillna: %s", ops)
            append_trace(f"HEURISTIC OUTPUT fillna={ops!r}")
            return ops

    logger.warning("Heuristic parser could not map user text to a supported operation: %s", user_text)
    append_trace(f"HEURISTIC OUTPUT none text={user_text!r}")
    return []

def parse_instruction_to_ops(
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
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    system_prompt = (
        "You are a data cleaning assistant. "
        "You translate user instructions into a JSON list of pandas-style "
        "cleaning operations. Your output must be valid JSON."
    )

    # Describe supported operations precisely to the model
    ops_description = """
Supported operations and their JSON formats:

1. dropna
   - Drop rows or columns with missing values.
   - JSON format:
     {
       "op": "dropna",
       "params": {
         "axis": 0 or 1,                  # 0 = rows, 1 = columns
         "subset": null or [<column names>]
       }
     }

2. fillna
   - Fill missing values in a single column.
   - JSON format:
     {
       "op": "fillna",
       "params": {
         "column": "<existing column name>",
         "strategy": "mean" | "median" | "mode" | "constant",
         "value": <constant value or null if not needed>
       }
     }

3. drop_columns
   - Drop one or more columns.
   - JSON format:
     {
       "op": "drop_columns",
       "params": {
         "columns": ["col1", "col2", ...]
       }
     }

4. replace_value
   - Replace one existing value with a new value in a single column.
   - JSON format:
     {
       "op": "replace_value",
       "params": {
         "column": "<existing column name>",
         "old_value": "<existing value>",
         "new_value": "<new value>"
       }
     }
"""

    user_prompt = f"""
The current pandas DataFrame has these columns and dtypes:
{json.dumps(dtypes, indent=2)}

Column names you are allowed to reference:
{columns}

{ops_description}

User instruction:
\"\"\"{user_text}\"\"\"

Instructions for your response:
- Use only these operations: {sorted(list(SUPPORTED_OPS))}
- Only reference columns that exist.
- If the instruction implies multiple steps, return a JSON list with multiple operation objects.
- You must ALWAYS return a valid JSON list. Never return explanations. If unsure, return [].
  based on column names and types. For example, "fill missing ages" should target the "age" column.
- If the instruction truly cannot be mapped to any supported operation, return an empty JSON list: [].
- Respond with **only** a valid JSON list, e.g.:
  [
    {{
      "op": "fillna",
      "params": {{
        "column": "age",
        "strategy": "median",
        "value": null
      }}
    }}
  ]
No additional text or comments.
"""

    try:
        logger.info("Parsing instruction into operations. Text: %s", user_text)
        append_trace(f"PARSER START text={user_text!r} columns={columns!r} dtypes={dtypes!r}")
        heuristic_result = _heuristic_parse_instruction_to_ops(user_text, df)
        if heuristic_result:
            logger.info("Using heuristic parser result without LLM call: %s", heuristic_result)
            append_trace(f"PARSER USED HEURISTIC ops={heuristic_result!r}")
            result = heuristic_result
        elif is_ollama_available():
            result = call_llm_for_json_cached(
                system_prompt,
                user_prompt,
                temperature
            )
            logger.info("LLM parser returned raw result: %s", result)
        else:
            logger.info("Ollama is unavailable; using heuristic parser for POC mode.")
            result = heuristic_result
    except Exception as e:
        logger.error("LLM parsing failed, falling back to heuristic parser: %s", e)
        append_trace(f"PARSER EXCEPTION error={e!r}")
        result = _heuristic_parse_instruction_to_ops(user_text, df)

    # Normalize: ensure we always have a list
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return []

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
    append_trace(f"PARSER FINAL ops={cleaned_ops!r}")
    return cleaned_ops
