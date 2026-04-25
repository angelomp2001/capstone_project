import json
from typing import List, Dict, Any

import pandas as pd

from .llm_utils import call_llm_for_json
from .cleaning_operations import SUPPORTED_OPS


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
- If the instruction is ambiguous (e.g., doesn't specify the column), make a reasonable assumption
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
        result = call_llm_for_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    except Exception:
        # In the POC we just fail gracefully and let the caller know it didn't work
        return []

    # Normalize: ensure we always have a list
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return []

    # Filter out malformed or unsupported ops
    cleaned_ops: List[Dict[str, Any]] = []
    for op in result:
        if not isinstance(op, dict):
            continue
        op_type = op.get("op")
        params = op.get("params", {})
        if op_type not in SUPPORTED_OPS:
            continue
        if not isinstance(params, dict):
            continue
        cleaned_ops.append({"op": op_type, "params": params})

    return cleaned_ops