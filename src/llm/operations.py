import logging
import mlflow
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List
import re
import inspect
import numpy as np
from sklearn.base import clone
from src.model_target import (
    data_splitter,
    define_features,
    define_column_types,
    feature_engineering_pipeline,
    train_model,
    generate_cv_summary_df,
    log_results_to_mlflow,
    save_metrics
)
from src.registry import (
    MODEL_REGISTRY,
)
# config area ##########################
TEST_SIZE = 0.2
RANDOM_STATE = 42
POLY_DEGREE = 2
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
EVALUATION_METRICS = {
    'classification': {
        'accuracy': accuracy_score,
        'roc_auc': roc_auc_score,
        'average_precision': average_precision_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    },
    'regression': {
        'neg_root_mean_squared_error': lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        'neg_mean_absolute_error': lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
        'r2': r2_score,
    }
}
PRIMARY_METRIC = 'accuracy'
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
        Model the target column.

        JSON format:
        {
          "op": "model_target",
          "params": {
            "column": "<column name>"
          }
        }
        """
        
        # dropping missing for testing purposes:
        original_rows = len(df)
        df = df.dropna().copy()
        dropped_rows = original_rows - len(df)
        if dropped_rows > 0:
            logger.warning(
                "Dropped %s rows containing missing values",
                dropped_rows
            )

        if len(df) == 0:
            logger.warning("model_target skipped because dataframe is empty")
            return df

        # check if column exists
        target = params.get("column")
        if target not in df.columns:
            logger.warning("model_target skipped because column does not exist: %s", target)
            return df
        
        # train–test split
        df_train, df_test = data_splitter(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # define columns on df_train
        features = define_features(df_train, target)

        # define column types
        task_type, cat_features, num_features = define_column_types(
            df=df_train, 
            target=target,
            features=features
        )

        # define metrics
        metrics = list(EVALUATION_METRICS[task_type].keys())
        logger.info("Metrics: %s", metrics)

        # define feature engineering pipelines and get model parameters
        pipelines = {}
        model_params = {}
        for model_name, model_info in MODEL_REGISTRY[task_type].items():
            
            pipelines[model_name] = feature_engineering_pipeline(
                numerical_features=num_features,
                categorical_features=cat_features,
                poly_degree=POLY_DEGREE,
                model_instance=model_info["class"],
                model_type=model_info["type"]
            )   

            # get model parameters
            model_params[model_name] = model_info.get("params", {})

        # train models: CV + hyperparameter tuning on df_train
        scores, trained_models = train_model(
            df=df_train,
            features=features,
            target=target,
            pipelines=pipelines,
            param_grids=model_params,
            cv_splits=CV_SPLITS,
            tuning_cv=CV_SPLITS,
            random_state=RANDOM_STATE,
            metrics=metrics,
            metric_funcs=EVALUATION_METRICS[task_type],
            primary_metric=PRIMARY_METRIC,
            task_type=task_type
        )
                
        # log and save
        log_results_to_mlflow(scores, metrics)
        saved_path = save_metrics(scores, "reports")
        mlflow.log_artifact(str(saved_path))

        # generate summary
        summary_df = generate_cv_summary_df(scores, metrics)
        logger.info("\nCross Validation Results:\n" + summary_df.to_markdown(index=False))
        
        # choose best model
        best_model_name = (
            scores.groupby("Model")[f'test_{PRIMARY_METRIC}']
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
        
        best_estimator_train = trained_models[best_model_name]
        logger.info("Best model: %s", best_model_name)
        mlflow.sklearn.log_model(best_estimator_train, "best_model_train")
        
        # test performance on holdout test set
        eval_function = EVALUATION_METRICS[task_type][PRIMARY_METRIC]
        best_model_test_score = eval_function(
            df_test[target], best_estimator_train.predict(df_test[features])
        )
        logger.info("Holdout test score: %s", best_model_test_score)
        mlflow.log_metric(f"holdout_{PRIMARY_METRIC}", best_model_test_score)

        # final model:
        # Train best model on entire df to get y_hat for entire df
        if best_estimator_train is not None:
            final_estimator = clone(best_estimator_train)
            final_estimator.fit(df[features], df[target])

            # append y_hat to df
            df[f"{target}_hat"] = final_estimator.predict(df[features])

            # get score of best model on df
            best_model_score_overall = EVALUATION_METRICS[task_type][PRIMARY_METRIC](df[target], df[f"{target}_hat"])
            mlflow.log_metric(f'{target}_hat', best_model_score_overall)
            
        logger.info("model_target complete. target=%s", target)
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
