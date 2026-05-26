import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, HalvingRandomSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler

from src.registry import MODEL_REGISTRY
logger = logging.getLogger(__name__)

# model target defaults
TEST_SIZE = 0.2
RANDOM_STATE = 42
POLY_DEGREE = 2
PRIMARY_METRIC = 'accuracy'
CV_SPLITS = 5

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
    features = df.columns.difference([target_col]).tolist()
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
    
    # define features by type
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df[features].select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    return task_type, categorical_features, numerical_features

def feature_engineering_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    poly_degree: int,
    model_class: Any,
    model_type: str,
    default_params: dict | None = None
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
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
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
    model_kwargs = default_params or {}
    pipeline = Pipeline(
        steps=[
            ('preprocessor', prep),
            ('model', model_class(**model_kwargs))
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
    
    if mlflow.active_run() is not None:
        mlflow.log_metrics({f"fold_{fold_idx}_{model_name}_{k}": v for k, v in fold_result.items() if k not in ["Model", "Fold"]})
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
        logger.info("Fold %d: Best hyperparameters for model %s: %s", fold_idx, name, model_best_estimator.get_params())

        fold_result = evaluate_metrics(
            model_best_estimator, name, fold_idx, X_outer_train, y_outer_train, 
            X_outer_val, y_outer_val, metrics, metric_funcs
        )
        fold_results.append(fold_result)
        logger.info("Fold %d results for model %s: %s", fold_idx, name, fold_result)

    return fold_results

def train_model_cv(
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
    task_type: str,
    experiment_name: str
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
        cv_splitter = StratifiedKFold(n_splits=cv_splits,shuffle=True,random_state=random_state)
    else:
        cv_splitter = KFold(n_splits=cv_splits,shuffle=True,random_state=random_state)

    raw_param_grid = param_grid or {}
    model_param_grid = filter_param_grid(pipeline, raw_param_grid)
    logger.info("Starting model training and evaluation for %s with %d CV splits and tuning CV=%d", model_name, cv_splits, tuning_cv)
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
    logger.info("Cross-validation completed for model %s. Scores: %s", model_name, scores_df)
    if mlflow.active_run() is not None:
        mlflow.log_metric(f"final_model_{model_name}_{primary_metric}", scores_df[f'test_{primary_metric}'].mean())
        
    # Train final model on full training set with best hyperparameters
    best_estimator = tune_hyperparameters(
        pipeline, model_param_grid, X_train, y_train,
        tuning_cv, primary_metric, random_state,
    )
    logger.info("Best hyperparameters for model %s on full training set: %s", model_name, best_estimator.get_params())

    # save best hyperparameters
    best_hyperparams_path = Path("reports") / f'{model_name}' / f"best_hyperparameters_{model_name}.csv"
    best_hyperparams_path.parent.mkdir(exist_ok=True)
    pd.DataFrame([best_estimator.get_params().items()]).to_csv(best_hyperparams_path, index=False)
    logger.info("Best hyperparameters for model %s saved to %s", model_name, best_hyperparams_path)
    
    # Log best model hyperparameters
    if mlflow.active_run() is not None:
        with mlflow.start_run(run_name="best_hyperparameters", nested=True):
            mlflow.log_params({
                "model_name": model_name,
                "best_hyperparameters": best_estimator.get_params(),
            })
            mlflow.log_artifact(best_hyperparams_path)

    return scores_df, best_estimator

def data_prep(
    df: pd.DataFrame,
    target: str
):
    # train–test split
    df_train, df_test = data_splitter(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[target] if df[target].dtype == 'object' else None)
    logger.info("Data split into train and test sets. Train size: %d, Test size: %d", len(df_train), len(df_test))
    # define columns on df_train
    features = define_features(df_train, target)
    logger.info("Features defined: %s", features)

    # infer task type and feature categories on the training set only
    task_type, cat_features, num_features = define_column_types(
        df=df_train,
        target=target,
        features=features
    )
    logger.info("Task type inferred: %s, Categorical features: %s, Numerical features: %s", task_type, cat_features, num_features)
    # define metrics
    metrics = list(EVALUATION_METRICS[task_type].keys())
    logger.info("Metrics: %s", metrics)

    return df_train, df_test, features, target, task_type, cat_features, num_features, metrics


def run_model_selection(
        df_train,
        features,
        target,
        task_type,
        cat_features,
        num_features,
        metrics,
        experiment_name,
        model: str = None
):
    
    best_estimators = {}
    all_avg_scores = []
    
    models_to_run = MODEL_REGISTRY[task_type]
    if model is not None:
        if model in models_to_run:
            # use only the model the user specified
            models_to_run = {model: models_to_run[model]}
            logger.info("Model selected by user: %s", model)
        else:
            logger.info("Model %s not found in MODEL_REGISTRY[task_type]. Cancelling model selection.", model)
            return None, None
        
    # train each model in a single outer loop and save its scores
    for model_name, model_info in models_to_run.items():
        logger.info("For loop: %s", MODEL_REGISTRY[task_type].keys())
        pipeline = feature_engineering_pipeline(
            numerical_features=num_features,
            categorical_features=cat_features,
            poly_degree=POLY_DEGREE,
            model_class=model_info["class"],
            model_type=model_info["type"],
            default_params=model_info.get("default_params")
        )
        logger.info("Pipeline created for model %s: %s", model_name, pipeline)
        
        # filter param grid to valid pipeline keys
        model_param_grid = model_info.get("params", {})
        logger.info("Model parameters defined for %s", model_name)


        scores_df, best_estimator = train_model_cv(
                    df=df_train,
                    features=features,
                    target=target,
                    model_name=model_name,
                    pipeline=pipeline,
                    param_grid=model_param_grid,
                    cv_splits=CV_SPLITS,
                    tuning_cv=CV_SPLITS,
                    random_state=RANDOM_STATE,
                    metrics=metrics,
                    metric_funcs=EVALUATION_METRICS[task_type],
                    primary_metric=PRIMARY_METRIC,
                    task_type=task_type,
                    experiment_name=experiment_name
                )
        best_estimators[model_name] = best_estimator
        logger.info("Model %s trained with cross-validation. Scores: %s", model_name, scores_df[scores_df['Model'] == model_name])
        
        # compute average scores across folds for the current model
        avg_scores_df = (
            scores_df.drop(columns=["Fold"])
            .groupby("Model", as_index=False)
            .mean()
            .round(6)
        )
            
        if avg_scores_df.empty:
            logger.warning("No model scores were generated; skipping model selection.")
            continue

        all_avg_scores.append(avg_scores_df)
        logger.info("Average CV scores for model %s: %s", model_name, avg_scores_df[avg_scores_df['Model'] == model_name])

        # log avg cv scores to mlflow
        if mlflow.active_run() is not None:
            for metric_name in metrics:
                mlflow.log_metric(
                f"avg_test_{model_name}_{metric_name}",
                float(avg_scores_df.at[0, f"test_{metric_name}"]),
                )
                mlflow.log_metric(
                f"avg_train_{model_name}_{metric_name}",
                float(avg_scores_df.at[0, f"train_{metric_name}"]),
                )

        # save cv fold scores to csv and log as artifact
        report_dir = Path(f"reports/{model_name}")
        report_dir.mkdir(parents=True, exist_ok=True)
        saved_path = report_dir / "cv_metrics.csv"
        scores_df.to_csv(saved_path, index=False)
        logger.info("CV scores for model %s saved to %s", model_name, saved_path)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(saved_path))        

    if not all_avg_scores:
        logger.warning("No model scores were generated; skipping model selection.")
        return None, None

    # Concatenate all average scores
    total_avg_scores_df = pd.concat(all_avg_scores, ignore_index=True)

    # save average scores for all models to csv and log as artifact
    total_avg_scores_df.to_csv("reports/avg_cv_metrics.csv", index=False)
    if mlflow.active_run() is not None:
        mlflow.log_artifact("reports/avg_cv_metrics.csv")

    # select best model based on primary metric
    # Search runs using MLflow
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    runs_df = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="", output_format="pandas")
    logger.info("Runs DataFrame columns: %s", runs_df.columns)
    runs_df.to_csv("reports/runs.csv", index=False)
    
    # Select best model based on primary metric
    try:
        col_map = {f"metrics.avg_test_{m}_{PRIMARY_METRIC}": m for m in models_to_run}
        active_row = runs_df[runs_df["run_id"] == mlflow.active_run().info.run_id].iloc[0]
        best_col = active_row[[c for c in col_map if c in runs_df.columns]].astype(float).idxmax()
        best_model_name = col_map[best_col]
    except Exception:
        best_model_name = (
            total_avg_scores_df.set_index("Model")[f"test_{PRIMARY_METRIC}"]
            .sort_values(ascending=False)
            .index[0]
        )
    best_model_on_train = best_estimators[best_model_name]
    logger.info("Best model selected on training data: %s", best_model_name)
    
    if mlflow.active_run() is not None:
        # Log best model name as parameter and tag
        mlflow.log_param("best_model_name", best_model_name)
        mlflow.set_tag("best_model_name", best_model_name)
        
        # Log the best model's tuned hyperparameters
        model_info = MODEL_REGISTRY[task_type][best_model_name]
        raw_param_grid = model_info.get("params", {})
        model_param_grid = filter_param_grid(best_model_on_train, raw_param_grid)
        best_params = {k: best_model_on_train.get_params()[k] for k in model_param_grid.keys() if k in best_model_on_train.get_params()}
        if best_params:
            mlflow.log_params(best_params)

        mlflow.sklearn.log_model(best_model_on_train, best_model_name)
        mlflow.sklearn.log_model(best_model_on_train, "best_model_train")

    return best_model_on_train, best_model_name
    

def fit_final_model(
    best_model_on_train,
    df,
    df_test,
    features,
    target,
    task_type,
    best_model_name=None,
):
    # final untouched evaluation on holdout test set
    eval_function = EVALUATION_METRICS[task_type][PRIMARY_METRIC]
    best_model_test_score = eval_function(
        df_test[target], best_model_on_train.predict(df_test[features])
    )
    logger.info("Holdout test score: %s", best_model_test_score)
    if mlflow.active_run() is not None:
        mlflow.log_metric(f"holdout_{PRIMARY_METRIC}", best_model_test_score)

    # retrain the selected model on all available data after final evaluation
    final_estimator = clone(best_model_on_train)
    final_estimator.fit(df[features], df[target])
    logger.info("Final model retrained on full training data.")
    if mlflow.active_run() is not None:
        mlflow.sklearn.log_model(final_estimator, "best_model_df")
        if best_model_name:
            mlflow.sklearn.log_model(final_estimator, f"final_model_{best_model_name}")

    df[f"{target}_hat"] = final_estimator.predict_proba(df[features])[:, 1]

    logger.info("%s_hat appended to DataFrame", target)
    
    return df
