import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import re
import inspect
from pathlib import Path
import mlflow
import sys
# pyrefly: ignore [missing-import]
from src.llm.llm_utils import load_config_yml, get_project_root, call_llm
from src.model_target import (
    TEST_SIZE, POLY_DEGREE, RANDOM_STATE, CV_SPLITS,
    data_prep,
    run_model_selection,
    fit_final_model,
)
# config area ##########################
#######################################
# setup loggers
logger = logging.getLogger(__name__)
trace = logging.getLogger("trace")

# supported operations
SUPPORTED_OPS = {
    "dropna", 
    "fillna", 
    "replace_value", 
    "drop_column",
    "get_first_value_in_col",
    "split_alphanumeric",
    "model_target",
}


class ApplyOperation:
    """
    Class containing methods for applying data cleaning operations.
    Each method corresponds to an operation the LLM can apply to a DataFrame.
    """
    @staticmethod
    def _copy_data(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("copy_data complete. shape_after=%s", df.shape)
        return df.copy()
    
    @staticmethod
    def _return_n_elements(element, n_returns):
        return pd.Series([element] * n_returns)

    @staticmethod # get_first_value_in_col
    def get_first_value_in_col(
        df: pd.DataFrame,
        params: Dict[str, Any] # col, split_by
    ) -> Tuple[pd.DataFrame, str]:
        """
        Splits a column by a delimiter and returns only the first part of each element.
        
        JSON format:
        {
          "op": "get_first_value_in_col",
          "params": {
            "column": "<existing column name>",
            "split_by": "<character to split by, e.g. ','>"
          }
        }
        """
        #extract parameters
        col = params.get("column")
        split_by = params.get("split_by")

        def get_first_in_element(element: str):
            # case 0: check if na
            if pd.isna(element): 
                return ApplyOperation._return_n_elements(element, 1)

            # case 1: one or more values
            # get first value from list
            first_value = str(element).split(split_by)[0]

            return pd.Series([first_value])

        # apply element-wise function to all of column
        df[[f'{col}_first_value']] = df[col].apply(get_first_in_element)
        
        msg = f'I split {col} by {split_by}. Final shape={df.shape}'
        logger.info("get_first_value_in_col complete. col=%s split_by=%s shape_after=%s", col, split_by, df.shape)
        return df[[f'{col}_first_value']], msg
    
    @staticmethod # split_alphanumeric
    def split_alphanumeric(
        df: pd.DataFrame,   
        params: Dict[str, Any] # col
    ) -> Tuple[pd.DataFrame, str]:
        """
        Splits a column into two new columns based on alphanumeric values.
        
        JSON format:
        {
          "op": "split_alphanumeric",
          "params": {
            "column": "<existing column name>"
          }
        }
        """
        col = params.get("column")
        
        def splitter(element: str):
            # case 0: check if na
            if pd.isna(element):
                return ApplyOperation._return_n_elements(element, 2)

            # verify cabine matches pattern:(letter(s))(number(s)) or (letter(s))(number(s)):
            # 4 groups are outputted
            match = re.match(r'^([A-Za-z]+)(\d+)$|^(\d+)([A-Za-z]+)$', element)
            
            # return NA if NA
            if not match:
                return ApplyOperation._return_n_elements(pd.NA, 2)
            
            # return as two series
            # save groups as dtype var options 
            str1, num1, num2, str2 = match.groups()
            
            # if letter(s) number(s)
            if str1 is not None:
                return pd.Series([str1, int(num1)])
     
            # else number(s) letter(s)
            return pd.Series([int(num2), str2])

        # apply element-wise function to all of cabin column
        df[[f'{col}_left', f'{col}_right']] = df[col].apply(splitter)
        msg = f'I split by alphanumeric: {col}_left, {col}_right. shape after: {df.shape}'
        
        logger.info("split_alphanumeric complete. col=%s shape_after=%s", col, df.shape)
        return df, msg

    @staticmethod # dropna
    def dropna(
        df: pd.DataFrame,
        params: Dict[str, Any] # axis, subset (optional)
    ) -> Tuple[pd.DataFrame, str]:
        """
        Drop rows or columns with missing values.
        
        JSON format:
        {
          "op": "dropna",
          "params": {
            "axis": 0 or 1,                  # 0 = rows, 1 = columns
            "subset": null or [<column names>]
          }
        }
        """
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
        msg = f'I dropped NA. axis: {axis}, subset: {subset}, shape after: {df.shape}'

        logger.info("dropna complete. axis=%s subset=%s shape_after=%s", axis, subset, df.shape)
        return df, msg

    @staticmethod # fillna
    def fillna(
        df: pd.DataFrame,
        params: Dict[str, Any] # column, strategy, value (optional)
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fill missing values in a single column.
        
        JSON format:
        {
          "op": "fillna",
          "params": {
            "column": "<existing column name>",
            "strategy": "mean" | "median" | "mode" | "constant",
            "value": <constant value or null if not needed>
          }
        }
        """
        # extract parameters
        column = params.get("column")
        strategy = params.get("strategy", "mean")
        value = params.get("value", None)

        if column not in df.columns:
            # Invalid column
            msg = f"{column} does not exit, so I couldn't drop it."
            logger.warning("fillna skipped because column does not exist: %s", column)
            trace.info("APPLY fillna skipped missing_column=%r", column)
            return df, msg

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
        
        msg = f"fillna complete. column={column}, strategy={strategy}, value={value}"
        logger.info("fillna complete. column=%s strategy=%s value=%s", column, strategy, value)
        return df, msg

    @staticmethod # drop_column
    def drop_column(
        df: pd.DataFrame,
        params: Dict[str, Any] # columns (list), column (string, optional)
    ) -> Tuple[pd.DataFrame, str]:
        """
        Drop one or more columns.
        
        JSON format:
        {
          "op": "drop_column",
          "params": {
            "columns": ["col1", "col2", ...]
          }
        }
        """
        # in case there is column and not columns key 
        columns = params.get("columns", [])
        if not columns and "column" in params:
            columns = [params["column"]]
        
        # if there is a string instead of list   
        if isinstance(columns, str):
            columns = [columns]

        # keep only existing columns
        columns = [c for c in columns if c in df.columns]
        if columns:
            df = df.drop(columns=columns)
        msg = f"I dropped the columns {columns}.  The new df shape is {df.shape}"
        
        logger.info("drop_column complete. dropped=%s shape_after=%s", columns, df.shape)
        return df, msg

    # Alias drop_columns to drop_column to maintain backward compatibility with old op names
    # drop_columns = drop_column

    @staticmethod # replace_value
    def replace_value(
        df: pd.DataFrame,
        params: Dict[str, Any] # column, old_value, new_value
    ) -> Tuple[pd.DataFrame, str]:
        """
        Replace one existing value with a new value in a single column.
        
        JSON format:
        {
          "op": "replace_value",
          "params": {
            "column": "<existing column name>",
            "old_value": "<existing value>",
            "new_value": "<new value>"
          }
        }
        """
        # extract parameters
        column = params.get("column")
        old_value = params.get("old_value")
        new_value = params.get("new_value")

        # check if column exists
        if column not in df.columns:
            msg = f"{column} does not exist"
            logger.warning("replace_value skipped because column does not exist: %s", column)
            return df, msg

        # count occurrences
        replacement_count = int((df[column].astype(str) == str(old_value)).sum())
        
        # Use pandas recommended way to check categorical dtype (is_categorical_dtype is deprecated)
        if isinstance(df[column].dtype, pd.CategoricalDtype) and new_value not in df[column].cat.categories:
            logger.info(
                "replace_value is converting categorical column '%s' to object so a new value can be inserted.",
                column,
            )
            df[column] = df[column].astype(object)
        
        #replace value
        df[column] = df[column].replace(old_value, new_value)
        logger.info(
            "replace_value complete. column=%s old_value=%s new_value=%s replacements=%s",
            column,
            old_value,
            new_value,
            replacement_count,
        )
        msg = f"In {column}, I replaced all {old_value} values with {new_value}."
        return df, msg

    @staticmethod # model_target
    def model_target(
        df: pd.DataFrame,
        params: Dict[str, Any], # target, model (optional), features (optional)
    ) -> Tuple[pd.DataFrame, str]:
        """
        Model/predict the target column. Optionally select a specific model to use from the model registry.

        JSON format:
        {
          "op": "model_target",
          "params": {
            "target": "<column name>",
            "model": null or "<model name from model registry>"
            "features": {<column_name>: <value>}
          }
        }
        """
        model = params.get("model", None)
        target = params.get("target")
        features_dict = params.get("features", None)
        if not target:
            msg = "I need a target column to work with"
            logger.warning("model_target skipped because no target column was provided")
            return df, msg

        if target not in df.columns:
            msg = f"{target} does not exist."
            logger.warning("model_target skipped because column does not exist: %s", target)
            return df, msg

        if not isinstance(features_dict, dict):
            features_dict = None

        if features_dict:
            # drop features not in df
            feature_set, user_feature_set = set(df.columns), set(features_dict.keys())
            dropped_features = user_feature_set - feature_set
            
            if dropped_features and len(dropped_features) < len(user_feature_set):
                logger.warning("features dropped: %s", dropped_features)
            
            # keep only features that are in df
            features_dict = {k: features_dict[k] for k in user_feature_set & feature_set}
            if not features_dict:
                features_dict = None

        # drop rows with missing values
        original_rows = len(df)
        df = df.dropna().copy()
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            logger.warning("Dropped %s rows containing missing values", dropped_rows)

        if df.empty:
            msg = "I don't see a df."
            logger.warning("model_target skipped because dataframe is empty")
            return df, msg

        # prep data
        df_train, df_test, features, target, task_type, cat_features, num_features, metrics = data_prep(
            df=df,
            target=target,
            features=list(features_dict.keys()) if features_dict else None
        )

        # Configure MLflow
        project_root = Path(__file__).resolve().parent.parent.parent
        db_path = project_root / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        # Determine experiment name: Model_Target_Tests if testing, otherwise Model_Target_Experiment
        if "pytest" in sys.modules or "unittest" in sys.modules:
            experiment_name = "Model_Target_Tests"
        else:
            experiment_name = "Model_Target_Experiment"

        mlflow.set_experiment(experiment_name)

        try:
            with mlflow.start_run(run_name="best_model"):
                # Log high-level setup parameters in the parent run
                mlflow.log_params({
                    "target_column": target,
                    "cv_splits": CV_SPLITS,
                    "random_state": RANDOM_STATE,
                    "poly_degree": POLY_DEGREE,
                    "test_size": TEST_SIZE,
                    "task_type": task_type,
                })

                best_model_on_train, best_model_name = run_model_selection(
                    df_train=df_train,
                    features=features,
                    target=target,
                    task_type=task_type,
                    cat_features=cat_features,
                    num_features=num_features,
                    metrics=metrics,
                    experiment_name=experiment_name,
                    model=model
                )

                if best_model_on_train is None:
                    msg = "The run_model_selection function didn't output a best_model_on_train variable."
                    logger.warning("No best model was trained or selected. Skipping final model fitting.")
                    return df, msg

                df, target_hat = fit_final_model(
                    best_model_on_train=best_model_on_train,
                    df=df,
                    df_test=df_test,
                    features=features_dict if features_dict else features,
                    target=target,
                    task_type=task_type,
                    best_model_name=best_model_name
                )
                
                if target_hat is not None:
                    if isinstance(target_hat, (list, np.ndarray)) and len(target_hat) > 0:
                        val = target_hat[0]
                    elif hasattr(target_hat, "iloc") and len(target_hat) > 0:
                        val = target_hat.iloc[0]
                    else:
                        val = target_hat

                    # get llm interpretation of val
                    prompts_path = get_project_root() / "configs" / "llm_prompts.yml"
                    PROMPTS = load_config_yml(str(prompts_path))
                    system_prompt = PROMPTS["operations"]["system_prompt"]
                    user_prompt_template = PROMPTS["operations"]["user_prompt"]
                    user_prompt = user_prompt_template.format(
                        model_name=best_model_name,
                        df_head=df.head(5).to_string(),
                        feature_values=features_dict,
                        target=target,
                        target_hat=val,
                    )
                    msg = call_llm(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.0
                    )
                    trace.info("LLM INTERPRETATION model=%r target=%r val=%r", best_model_name, target, val)

                else:
                    msg = f"I finished modeling {target}"
                
                return df, msg
        except Exception as e:
            msg = f"Model target execution failed: {str(e)}"
            logger.exception("Error during model_target execution")
            return df, msg

def get_ops_description() -> str:
    """
    Dynamically generates the ops_description string for the LLM prompt
    by extracting the docstrings of the methods in ApplyOperation.
    """
    lines = ["Supported operations and their JSON formats:\n"]
    for i, method_name in enumerate(sorted(SUPPORTED_OPS), 1):
        method = getattr(ApplyOperation, method_name, None)
        if method and method.__doc__:
            doc = inspect.cleandoc(method.__doc__)
            lines.append(f"{i}. {method_name}")
            for line in doc.split("\\n"):
                lines.append(f"   {line}")
            lines.append("")
    return "\\n".join(lines)

def apply_operation(df: pd.DataFrame, op: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Apply a single operation to a DataFrame.

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
        # Unsupported operation
        msg = f"The operation I tried isn't supported. These are my options: {SUPPORTED_OPS}"
        logger.warning(f"Unsupported op from LLM: {op_type}")
        trace.info("APPLY UNSUPPORTED op=%r", op_type)
        return df, msg

    df = df.copy()

    if hasattr(ApplyOperation, op_type):
        method = getattr(ApplyOperation, op_type)
        df, msg = method(df, params) # all methods output df, msg
    else:
        logger.warning(f"Method {op_type} not found in ApplyOperation")

    trace.info(
        "APPLY END op=%r shape_after=%r columns_after=%r preview_after=%r",
        op_type,
        df.shape,
        df.columns.tolist(),
        df.head(3).to_dict(orient="records"),
    )
    return df, msg

def apply_operations(df: pd.DataFrame, ops: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, str]:
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
    df_copy = df.copy()
    logger.info("Applying %s operation(s) in sequence.", len(ops))
    msg = "No operations applied."
    for op in ops:
        df_copy, msg = apply_operation(df_copy, op)
    logger.info(f"{msg}")
    return df_copy, msg
