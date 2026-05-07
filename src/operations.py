import logging

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

logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")

SUPPORTED_OPS = {"dropna", "fillna", "drop_columns", "replace_value", "drop_column"}


class ApplyOperation:
    """
    Class containing methods for applying data cleaning operations.
    Each method corresponds to an operation the LLM can apply to a DataFrame.
    """

    @staticmethod
    def dropna(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
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
        return df

    @staticmethod
    def fillna(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        column = params.get("column")
        strategy = params.get("strategy", "mean")
        value = params.get("value", None)

        if column not in df.columns:
            # Invalid column: ignore for POC
            logger.warning("fillna skipped because column does not exist: %s", column)
            trace.info("APPLY fillna skipped missing_column=%r", column)
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
        return df

    @staticmethod
    def drop_column(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Drops one or more columns from the DataFrame.
        Supports both 'columns' (list) and 'column' (string) in params.
        """
        columns = params.get("columns", [])
        if not columns and "column" in params:
            columns = [params["column"]]
            
        if isinstance(columns, str):
            columns = [columns]
        # Keep only existing columns
        columns = [c for c in columns if c in df.columns]
        if columns:
            df = df.drop(columns=columns)
        logger.info("drop_column complete. dropped=%s shape_after=%s", columns, df.shape)
        return df

    # Alias drop_columns to drop_column to maintain backward compatibility with old op names
    drop_columns = drop_column

    @staticmethod
    def replace_value(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        column = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")

        if column not in df.columns:
            logger.warning("replace_value skipped because column does not exist: %s", column)
            trace.info("APPLY replace_value skipped missing_column=%r", column)
            return df

        replacement_count = int((df[column].astype(str) == str(old_value)).sum())
        
        # Use pandas recommended way to check categorical dtype (is_categorical_dtype is deprecated)
        if isinstance(df[column].dtype, pd.CategoricalDtype) and new_value not in df[column].cat.categories:
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
        return df


def apply_operation(df: pd.DataFrame, op: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a single cleaning operation to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (will not be modified in-place; a modified copy is returned).
    op : dict
        Operation dictionary with keys:
          - "op": operation name (e.g., "dropna", "fillna", "drop_column", "drop_columns")
          - "params": dict with operation-specific parameters

    Returns
    -------
    pd.DataFrame
        New DataFrame with the operation applied.
    """
    op_type = op.get("op")
    params = op.get("params", {})
    logger.info("Applying operation %s with params %s", op_type, params)
    trace.info(
        "APPLY START op=%r params=%r shape_before=%r columns_before=%r",
        op_type,
        params,
        df.shape,
        df.columns.tolist(),
    )

    if op_type not in SUPPORTED_OPS:
        # Unsupported operation: no-op (for robustness in POC)
        logger.warning(f"Unsupported op from LLM: {op_type}")
        trace.info("APPLY UNSUPPORTED op=%r", op_type)
        return df

    df = df.copy()

    if hasattr(ApplyOperation, op_type):
        method = getattr(ApplyOperation, op_type)
        df = method(df, params)
    else:
        logger.warning(f"Method {op_type} not found in ApplyOperation")

    trace.info(
        "APPLY END op=%r shape_after=%r columns_after=%r preview_after=%r",
        op_type,
        df.shape,
        df.columns.tolist(),
        df.head(3).to_dict(orient="records"),
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
