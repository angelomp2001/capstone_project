import logging

import pandas as pd
from typing import Dict, Any, List
from .llm_utils import append_trace

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
logger = logging.getLogger(__name__)

SUPPORTED_OPS = {"dropna", "fillna", "drop_columns", "replace_value"}


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
    logger.info("Applying operation %s with params %s", op_type, params)
    append_trace(
        f"APPLY START op={op_type!r} params={params!r} "
        f"shape_before={df.shape!r} columns_before={df.columns.tolist()!r}"
    )

    if op_type not in SUPPORTED_OPS:
        # Unsupported operation: no-op (for robustness in POC)
        logger.warning(f"Unsupported op from LLM: {op_type}")
        append_trace(f"APPLY UNSUPPORTED op={op_type!r}")
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
        logger.info("dropna complete. axis=%s subset=%s shape_after=%s", axis, subset, df.shape)

    elif op_type == "fillna":
        column = params.get("column")
        strategy = params.get("strategy", "mean")
        value = params.get("value", None)

        if column not in df.columns:
            # Invalid column: ignore for POC
            logger.warning("fillna skipped because column does not exist: %s", column)
            append_trace(f"APPLY fillna skipped missing_column={column!r}")
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
        logger.info("fillna complete. column=%s strategy=%s value=%s", column, strategy, value)

    elif op_type == "drop_columns":
        columns = params.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        # Keep only existing columns
        columns = [c for c in columns if c in df.columns]
        if columns:
            df = df.drop(columns=columns)
        logger.info("drop_columns complete. dropped=%s shape_after=%s", columns, df.shape)

    elif op_type == "replace_value":
        column = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")

        if column not in df.columns:
            logger.warning("replace_value skipped because column does not exist: %s", column)
            append_trace(f"APPLY replace_value skipped missing_column={column!r}")
            return df

        replacement_count = int((df[column].astype(str) == str(old_value)).sum())
        if pd.api.types.is_categorical_dtype(df[column]) and new_value not in df[column].cat.categories:
            logger.info(
                "replace_value is converting categorical column '%s' to object so a new value can be inserted.",
                column,
            )
            df[column] = df[column].astype(object)
        df[column] = df[column].replace(old_value, new_value)
        logger.info(
            "replace_value complete. column=%s old_value=%s new_value=%s replacements=%s",
            column,
            old_value,
            new_value,
            replacement_count,
        )

    append_trace(
        f"APPLY END op={op_type!r} shape_after={df.shape!r} "
        f"columns_after={df.columns.tolist()!r} preview_after={df.head(3).to_dict(orient='records')!r}"
    )
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
    logger.info("Applying %s operation(s) in sequence.", len(ops))
    for op in ops:
        result = apply_operation(result, op)
    logger.info("Finished applying operations. Final shape=%s", result.shape)
    return result
