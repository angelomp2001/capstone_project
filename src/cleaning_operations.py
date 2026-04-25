import pandas as pd
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# Supported operations and their parameter schemas
# -----------------------------------------------------------------------------
# For the POC, we keep this intentionally small and simple.
#
# Operation: dropna
#   {
#     "op": "dropna",
#     "params": {
#       "axis": 0,                  # 0 = rows, 1 = columns
#       "subset": null or [cols],   # optional
#     }
#   }
#
# Operation: fillna
#   {
#     "op": "fillna",
#     "params": {
#       "column": "age",
#       "strategy": "mean" | "median" | "mode" | "constant",
#       "value": null or <constant>
#     }
#   }
#
# Operation: drop_columns
#   {
#     "op": "drop_columns",
#     "params": {
#       "columns": ["col1", "col2"]
#     }
#   }
# -----------------------------------------------------------------------------

# supported operations:
# copy_df, validate required, validate unique, validate categorical values, validate bounds, validate date range, drop duplicates, replace value, coerce dtype, dropna, fillna, drop_columns
SUPPORTED_OPS = {"dropna", "fillna", "drop_columns"}


def apply_operation(df: pd.DataFrame, op: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a single cleaning operation to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will not be modified in-place; a modified copy is returned).
    op : dict
        Operation dictionary with keys:
          - "op": operation name (e.g., "dropna", "fillna", "drop_columns")
          - "params": dict with operation-specific parameters

    Returns
    -------
    pd.DataFrame
        New DataFrame with the operation applied.
    """
    op_type = op.get("op")
    params = op.get("params", {})

    if op_type not in SUPPORTED_OPS:
        # Unsupported operation: no-op (for robustness in POC)
        return df

    df = df.copy()

    if op_type == "dropna":
        axis = params.get("axis", 0)
        subset = params.get("subset", None)

        # Normalize subset: can be string, list, or None
        if isinstance(subset, str):
            subset = [subset]
        if subset is not None:
            # Only keep columns that exist
            subset = [c for c in subset if c in df.columns]
            if not subset:
                subset = None

        df = df.dropna(axis=axis, subset=subset)

    elif op_type == "fillna":
        column = params.get("column")
        strategy = params.get("strategy", "mean")
        value = params.get("value", None)

        if column not in df.columns:
            # Invalid column: ignore for POC
            return df

        if strategy == "mean":
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == "median":
            df[column] = df[column].fillna(df[column].median())
        elif strategy == "mode":
            # mode() returns a Series; take first
            df[column] = df[column].fillna(df[column].mode().iloc[0])
        elif strategy == "constant":
            # If value is None, you might choose a default (e.g., 0)
            if value is None:
                value = 0
            df[column] = df[column].fillna(value)

    elif op_type == "drop_columns":
        columns = params.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        # Keep only existing columns
        columns = [c for c in columns if c in df.columns]
        if columns:
            df = df.drop(columns=columns)

    return df


def apply_operations(df: pd.DataFrame, ops: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a sequence of operations to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will not be modified in-place; a modified copy is returned).
    ops : list of dict
        List of operation dictionaries. Each dict must have the same structure
        as described in `apply_operation`.

    Returns
    -------
    pd.DataFrame
        New DataFrame after all operations have been applied in order.
    """
    result = df.copy()
    for op in ops:
        result = apply_operation(result, op)
    return result