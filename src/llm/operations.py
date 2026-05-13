import logging
import mlflow
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import re
import inspect
from src.model_target import (
    data_splitter,
    define_features_by_type,
    define_target,
    numerical_transformer,
    categorical_transformer,
    tree_based_numerical_transformer,
    tree_based_categorical_transformer,
    preprocessor,
    tree_based_preprocessor,
    build_pipeline,
    train_model_cv,
    generate_cv_summary_df,
    log_results_to_mlflow,
    save_metrics
)
from src.registry import (
    MODEL_REGISTRY, # {model_name, model_instance}
    MODEL_GROUPS, # {"non-tree-based": [model_name], "tree-based": [model_name]}
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

# config area ##########################
TEST_SIZE = 0.2
RANDOM_STATE = 42
POLY_DEGREE = 2
METRICS = {
    'accuracy': accuracy_score,
    'roc_auc': roc_auc_score,
    'average_precision': average_precision_score,
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score,
}
CV_SPLITS = 5

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

    @staticmethod
    def get_first_value_in_col(
        df: pd.DataFrame,
        params: Dict[str, Any] # col, split_by
    ):
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
        logger.info("get_first_value_in_col complete. col=%s split_by=%s shape_after=%s", col, split_by, df.shape)
        return df[[f'{col}_first_value']]
    
    @staticmethod
    def split_alphanumeric(
        df: pd.DataFrame,   
        params: Dict[str, Any] # col
    ):
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
        logger.info("split_alphanumeric complete. col=%s shape_after=%s", col, df.shape)
        return df

    @staticmethod
    def dropna(
        df: pd.DataFrame,
        params: Dict[str, Any] # axis, subset (optional)
    ) -> pd.DataFrame:
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
        logger.info("dropna complete. axis=%s subset=%s shape_after=%s", axis, subset, df.shape)
        return df

    @staticmethod
    def fillna(
        df: pd.DataFrame,
        params: Dict[str, Any] # column, strategy, value (optional)
    ) -> pd.DataFrame:
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
    def drop_column(
        df: pd.DataFrame,
        params: Dict[str, Any] # columns (list), column (string, optional)
    ) -> pd.DataFrame:
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
        logger.info("drop_column complete. dropped=%s shape_after=%s", columns, df.shape)
        return df

    # Alias drop_columns to drop_column to maintain backward compatibility with old op names
    # drop_columns = drop_column

    @staticmethod
    def replace_value(
        df: pd.DataFrame,
        params: Dict[str, Any] # column, old_value, new_value
    ) -> pd.DataFrame:
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
            logger.warning("replace_value skipped because column does not exist: %s", column)
            return df

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
        return df

    @staticmethod
    def model_target(
        df: pd.DataFrame,
        params: Dict[str, Any] # column
    ) -> pd.DataFrame:
        """
        Set the target column for modeling.

        JSON format:
        {
          "op": "model_target",
          "params": {
            "column": "<column name>"
          }
        }
        """
        column = params.get("column")
        if column not in df.columns:
            logger.warning("model_target skipped because column does not exist: %s", column)
            return df
        
        # outer split
        remaining_df, validate_df = data_splitter(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # inner split
        train_df, test_df = data_splitter(remaining_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
       
        # train df:
        # define target
        y_train = define_target(train_df, column)

        # define features by type
        X_train = train_df.drop(columns=[column])

        # define features by type
        categorical_features, numerical_features = define_features_by_type(X_train, y_train)

        # non-tree based models
        # apply transformers:
        numerical_features = numerical_transformer()
        categorical_features = categorical_transformer()
         
        # apply preprocessors:
        non_tree_based_preprocessor = preprocessor(numerical_features, categorical_features, POLY_DEGREE)

        # tree based:
        # apply transformers:
        tree_based_numerical_transformer = tree_based_numerical_transformer()
        tree_based_categorical_transformer = tree_based_categorical_transformer()

        # apply preprocessors:
        tree_based_preprocessor = tree_based_preprocessor(tree_based_numerical_transformer, tree_based_categorical_transformer)
        
        # build pipelines by group:
        pipelines = {}
        ## cross-validate on train_df
        
        # group 1: non-tree based
        for model_name in MODEL_GROUPS["non-tree-based"]:
            pipelines[model_name] = build_pipeline(model_name, non_tree_based_preprocessor)     

        # group 2: tree-based
        for model_name in MODEL_GROUPS["tree-based"]:
            pipelines[model_name] = build_pipeline(MODEL_REGISTRY[model_name], tree_based_preprocessor)
        
        ## cross-validate on train_df
        all_fold_results = []
        fold_df = train_model_cv(X_train, y_train, pipelines, cv_splits=CV_SPLITS, random_state=RANDOM_STATE, metrics=METRICS)
        all_fold_results.append(fold_df)

        # Process and log results
        if all_fold_results:
            final_fold_results = pd.concat(all_fold_results, ignore_index=True)
            summary_df = generate_cv_summary_df(final_fold_results, METRICS)
            
            logger.info("\nCross Validation Results:\n" + summary_df.to_markdown(index=False))
            
            log_results_to_mlflow(final_fold_results, METRICS)
            saved_path = save_metrics(final_fold_results, "reports")
            mlflow.log_artifact(str(saved_path))
        else:
            logger.warning("No models were evaluated.")
        

        logger.info("model_target complete. column=%s", column)
        return df

        
        
        

        

        






        df[f'{column}_target'] = target
        logger.info("model_target complete. column=%s", column)
        return df

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

def apply_operation(df: pd.DataFrame, op: Dict[str, Any]) -> pd.DataFrame:
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
        # Unsupported operation: no-op (for robustness in POC)
        logger.warning(f"Unsupported op from LLM: {op_type}")
        trace.info("APPLY UNSUPPORTED op=%r", op_type)
        return df

    df = df.copy()

    if hasattr(ApplyOperation, op_type):
        method = getattr(ApplyOperation, op_type)
        df = method(df, params) # all methods output a df
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
