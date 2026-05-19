from PIL.ImageFile import logger
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import mlflow
from pathlib import Path
import os
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Any


def data_splitter(df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    """Outer split"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def define_features(df: pd.DataFrame, target_col: str):
    """Define features"""
    features = df.drop(columns=[target_col])
    return features

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
    model_instance: Any,
    model_type: str
):
    """
    Pipelines for model target.
    """
    # non-tree-based models: numerical transformer
    def numerical_transformer(poly_degree):
        """Numerical transformer"""
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

    """
    Creates a scikit-learn Pipeline with the preprocessor and model.
    """    
    pipeline = Pipeline(steps=[
        ('preprocessor', prep),
        ('model', model_instance)
    ])
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

def compile_scores_to_df(results: list[dict]) -> pd.DataFrame:
    """Compile scores list of dicts to a pandas dataframe."""
    return pd.DataFrame(results)

def train_model(
    df: pd.DataFrame, 
    features: list[str],
    target: str,
    pipelines: dict,
    param_grids: dict | None, 
    cv_splits: int, 
    tuning_cv: int,
    random_state: int, 
    metrics: list[str],
    metric_funcs: dict,
    primary_metric: str,
    task_type: str
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate multiple models using cross-validation over individual folds.
    
    Args:
        df (pd.DataFrame): Dataframe containing training data.
        features (list[str]): Features list.
        target (str): Target column name.
        pipelines (dict): Dictionary of model names to instantiated pipelines.
        param_grids (dict): Dictionary of model names to hyperparameter distributions.
        cv_splits (int): Number of CV folds.
        tuning_cv (int): Number of CV folds for hyperparameter tuning.
        random_state (int): Random state for reproducibility.
        metrics (list): List of Scikit-learn scoring strings.
        metric_funcs (dict): Functions to calculate the metrics.
        primary_metric (str): Primary evaluation metric.
        task_type (str): Classification or Regression.

    Returns:
        tuple[pd.DataFrame, dict]: 
            - pd.DataFrame: A dataframe containing model evaluation metrics for EVERY fold.
            - dict: A dictionary of trained models.
    """

    # define training data
    X_train = df[features]
    y_train = df[target]

    # initialize outputs
    results = []
    trained_models = {}
    
    # define K-Fold strategy
    if task_type == "classification":
        cv_splitter = StratifiedKFold(
            n_splits=cv_splits if cv_splits > 1 else 2, # fallback, though we won't use it if cv <= 1
            shuffle=True,
            random_state=random_state
        )
    else:
        cv_splitter = KFold(
            n_splits=cv_splits if cv_splits > 1 else 2,
            shuffle=True,
            random_state=random_state
        )
    
    # iterate through models
    for name, pipeline in pipelines.items():
        try:
            # get parameter grid
            raw_param_grid = param_grids.get(name, {}) if param_grids else {}
            model_param_grid = filter_param_grid(pipeline, raw_param_grid)
            
            # define CV
            if cv_splits <= 1:
                # CV on whole dataset for hyperparameter tuning
                best_estimator = tune_hyperparameters(
                    pipeline, model_param_grid, X_train, y_train, 
                    tuning_cv, primary_metric, random_state
                )
                
                # metric evaluation
                fold_result = evaluate_metrics(
                    best_estimator, name, 1, X_train, y_train, 
                    None, None, metrics, metric_funcs
                )
                
                results.append(fold_result)
                trained_models[name] = best_estimator
            else:
                # outer cv loop
                fold_results = cross_validate_model(
                    name, pipeline, model_param_grid, X_train, y_train, 
                    cv_splitter, tuning_cv, primary_metric, random_state, 
                    metrics, metric_funcs
                )
                results.extend(fold_results)
            
                # final full training
                # train one model on all training data 
                # for later test-set evaluation
                final_best_estimator = tune_hyperparameters(
                    pipeline, model_param_grid, X_train, y_train, 
                    tuning_cv, primary_metric, random_state
                )
                trained_models[name] = final_best_estimator

        except Exception as e:
            logger.exception(f"Error evaluating {name}: {e}")

    return compile_scores_to_df(results), trained_models
      

def generate_cv_summary_df(fold_df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Generate the averaged summary dataframe with cosmetic formatting.
    
    Args:
        fold_df (pd.DataFrame): Dataframe containing model evaluation metrics for EVERY fold.
        metrics (list): List of metric names.
        
    Returns:
        pd.DataFrame: Formatted dataframe containing model evaluation metrics averaged per model.
    """
    if fold_df.empty:
        logger.warning("No models were successfully evaluated.")
        return pd.DataFrame()

    agg_dict = {}
    for metric_name in metrics:
        agg_dict[f'test_{metric_name}'] = 'mean'
        agg_dict[f'train_{metric_name}'] = 'mean'
        
    summary_df = fold_df.groupby('Model').agg(agg_dict).reset_index()

    # Formulate the cleanly averaged summary specifically requested for github README tables
    if metrics:
        first_metric = f"test_{metrics[0]}"
        if first_metric in summary_df.columns:
            summary_df = summary_df.sort_values(by=first_metric, ascending=False).reset_index(drop=True)
    
    # Condense visual clutter by rounding display columns neatly to the second decimal place
    return summary_df.round(2)

def log_results_to_mlflow(fold_df: pd.DataFrame, metrics: list) -> None:
    """
    Log individual fold metrics to MLflow for granular analysis.
    
    Args:
        fold_df (pd.DataFrame): Dataframe containing model evaluation metrics for EVERY fold.
        metrics (list): List of metrics to log.
    """
    for _, row in fold_df.iterrows():
        model_name = row['Model']
        fold_idx = int(row['Fold'])
        for metric_name in metrics:
            mlflow.log_metric(f"{model_name}_test_{metric_name}", float(row[f'test_{metric_name}']), step=fold_idx)
            mlflow.log_metric(f"{model_name}_train_{metric_name}", float(row[f'train_{metric_name}']), step=fold_idx)

def save_metrics(fold_results_df, reports_dir):
    """Serialize the extensive multi-fold metrics file locally to disk cleanly."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "cv_metrics.csv"
    fold_results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved detailed metrics safely to {metrics_path}")
    return metrics_path

