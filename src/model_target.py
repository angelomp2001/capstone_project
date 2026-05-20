import logging
from typing import Any

import mlflow
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, StratifiedKFold, HalvingRandomSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


def data_splitter(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_df, test_df


def define_features(df: pd.DataFrame, target_col: str) -> list[str]:
    """Return a feature column list excluding the target."""
    return df.columns.difference([target_col]).tolist()

def define_column_types(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task_type: str = None,
    ):
    """Define features by type"""

    # determine target column task type
    if task_type not in ["classification", "regression"]:
        # determine target column task type 
        if df[target].nunique() <= 10 or df[target].dtype == 'object': 
            task_type = "classification" 
        else: 
            task_type = "regression"
        logger.info("Task type: %s", task_type)
    
    # define features by type
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df[features].select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    return task_type, categorical_features, numerical_features

def feature_engineering_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    poly_degree: int,
    model_class: Any,
    model_type: str
):
    """Builds a preprocessing pipeline and attaches a fresh estimator instance."""

    def numerical_transformer(poly_degree):
        """Numerical transformer."""
        return Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=poly_degree))
        ])

    # non-tree-based models: categorical transformer
    def categorical_transformer():
        """Categorical transformer"""
        return Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

    # tree-based models: numerical transformer
    def tree_based_numerical_transformer():
        """Numerical transformer"""
        return Pipeline(steps=[
            ('no_scaler', 'passthrough')
        ])

    # tree-based models: categorical transformer
    def tree_based_categorical_transformer():
        """Categorical transformer"""
        return Pipeline(steps=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

    # Preprocessor selection based on model_type
    if model_type == "tree-based":
        prep = ColumnTransformer(transformers=[
            ('num', tree_based_numerical_transformer(), numerical_features),
            ('cat', tree_based_categorical_transformer(), categorical_features)
        ])
    else:
        prep = ColumnTransformer(transformers=[
            ('num', numerical_transformer(poly_degree), numerical_features),
            ('cat', categorical_transformer(), categorical_features)
        ])

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', prep),
            ('model', model_class())
        ]
    )
    return pipeline

def filter_param_grid(pipeline: Pipeline, raw_param_grid: dict) -> dict:
    """Filter hyperparameter grid based on valid pipeline keys."""
    valid_keys = pipeline.get_params().keys()
    model_param_grid = {}
    for k, v in raw_param_grid.items():
        new_k = k.replace("preprocess__", "preprocessor__").replace("clf__", "model__")
        if new_k in valid_keys:
            model_param_grid[new_k] = v
    return model_param_grid

def tune_hyperparameters(
    pipeline: Pipeline, 
    model_param_grid: dict, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    tuning_cv: int, 
    primary_metric: str, 
    random_state: int
) -> Any:
    """Perform hyperparameter tuning and return the best estimator."""
    if not model_param_grid:
        fitted_model = clone(pipeline)
        fitted_model.fit(X_train, y_train)
        return fitted_model
        
    search = HalvingRandomSearchCV(
        estimator=clone(pipeline),
        param_distributions=model_param_grid,
        cv=tuning_cv,
        factor=2,
        scoring=primary_metric,
        n_jobs=-1,
        random_state=random_state,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def evaluate_metrics(
    model: Any, 
    model_name: str, 
    fold_idx: int, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame | None, 
    y_val: pd.Series | None, 
    metrics: list[str], 
    metric_funcs: dict
) -> dict:
    """Evaluate metric functions and return a dictionary of results."""
    fold_result = {
        "Model": model_name,
        "Fold": fold_idx,
    }
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val) if X_val is not None else None
    
    for metric_name in metrics:
        metric_fn = metric_funcs[metric_name]
        train_score = metric_fn(y_train, y_train_pred)
        test_score = metric_fn(y_val, y_val_pred) if y_val_pred is not None else train_score
        
        if 'neg_' in metric_name:
            train_score = -train_score
            test_score = -test_score
            
        fold_result[f'train_{metric_name}'] = train_score
        fold_result[f'test_{metric_name}'] = test_score
        
    return fold_result

def cross_validate_model(
    name: str,
    pipeline: Pipeline,
    model_param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splitter: Any,
    tuning_cv: int,
    primary_metric: str,
    random_state: int,
    metrics: list[str],
    metric_funcs: dict
) -> list[dict]:
    """Perform outer cross validation."""
    fold_results = []
    for fold_idx, (train_index, test_index) in enumerate(cv_splitter.split(X_train, y_train), start=1):
        X_outer_train, X_outer_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_outer_train, y_outer_val = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model_best_estimator = tune_hyperparameters(
            pipeline, model_param_grid, X_outer_train, y_outer_train, 
            tuning_cv, primary_metric, random_state
        )
        
        fold_result = evaluate_metrics(
            model_best_estimator, name, fold_idx, X_outer_train, y_outer_train, 
            X_outer_val, y_outer_val, metrics, metric_funcs
        )
        fold_results.append(fold_result)
    return fold_results

def train_model(
    df: pd.DataFrame, 
    features: list[str],
    target: str,
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict | None, 
    cv_splits: int, 
    tuning_cv: int,
    random_state: int, 
    metrics: list[str],
    metric_funcs: dict,
    primary_metric: str,
    task_type: str
) -> tuple[pd.DataFrame, Any]:
    """
    Train a model with nested cross-validation and hyperparameter tuning.

    Args:
        df (pd.DataFrame): Dataframe containing training data.
        features (list[str]): Features list.
        target (str): Target column name.
        model_name (str): Name of the model.
        pipeline (Pipeline): Scikit-learn pipeline template.
        param_grid (dict): Hyperparameter grid for tuning.
        cv_splits (int): Number of outer CV folds for evaluation.
        tuning_cv (int): Number of CV folds for hyperparameter tuning.
        random_state (int): Random state for reproducibility.
        metrics (list): List of metric names.
        metric_funcs (dict): Functions to calculate the metrics.
        primary_metric (str): Primary evaluation metric.
        task_type (str): "classification" or "regression".

    Returns:
        tuple[pd.DataFrame, Any]:
            - pd.DataFrame: Evaluation metrics for each fold.
            - Any: Best trained pipeline on the full training set.
    """
    X_train = df[features]
    y_train = df[target]

    # Ensure cv_splits is at least 2
    cv_splits = max(cv_splits, 2)

    # Initialize cross-validation splitter
    if task_type == "classification":
        cv_splitter = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )
    else:
        cv_splitter = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state,
        )

    raw_param_grid = param_grid or {}
    model_param_grid = filter_param_grid(pipeline, raw_param_grid)

    # Perform nested cross-validation: outer loop for evaluation, inner loop for tuning
    scores = cross_validate_model(
        model_name,
        pipeline,
        model_param_grid,
        X_train,
        y_train,
        cv_splitter,
        tuning_cv,
        primary_metric,
        random_state,
        metrics,
        metric_funcs,
    )
    scores_df = pd.DataFrame(scores)

    # Train final model on full training set with best hyperparameters
    best_estimator = tune_hyperparameters(
        pipeline, model_param_grid, X_train, y_train,
        tuning_cv, primary_metric, random_state,
    )

    return scores_df, best_estimator

