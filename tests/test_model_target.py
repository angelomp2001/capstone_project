import pandas as pd
import pytest
from unittest.mock import patch
from src.model_target import *
from app import create_initial_df
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin, RegressorMixin


def test_data_splitter():
    '''test the data_splitter function'''
    df = create_initial_df()
    test_size = 0.2
    random_state = 42
    stratify = df["Survived"]
    train_df, test_df = data_splitter(df, test_size, random_state, stratify)
    assert train_df.shape[0] > 0
    assert test_df.shape[0] > 0
    assert train_df.shape[1] > 0
    assert test_df.shape[1] > 0
    assert (train_df.shape[0] / df.shape[0]) - (1 - test_size) < 0.1
    assert (test_df.shape[0] / df.shape[0]) - (test_size) < 0.1

def test_define_features():
    '''test the define_features function'''
    df = create_initial_df()
    target_column = "Survived"
    feature_list = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked", "SibSp", "Parch"]
    features = define_features(df, target_column, feature_list)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)
    assert all(f in df.columns for f in features)

    features = define_features(df, target_column)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)
    assert all(f in df.columns for f in features)

def test_define_column_types():
    '''test the define_column_types function'''
    df = create_initial_df()
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    assert isinstance(task_type, str)
    assert isinstance(categorical_features, list)
    assert isinstance(numerical_features, list)
    assert task_type in ["classification", "regression"]
    assert all(isinstance(f, str) for f in categorical_features)
    assert all(isinstance(f, str) for f in numerical_features)
    assert all(f in features for f in categorical_features)
    assert all(f in features for f in numerical_features)
    assert len(categorical_features) + len(numerical_features) == len(features)

def test_feature_engineering_pipeline():
    '''test the feature_engineering_pipeline function'''
    df = create_initial_df()
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    assert isinstance(pipeline, Pipeline)

def test_filter_param_grid():
    '''test the filter_param_grid function'''
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', 'passthrough', ['num']),
            ('cat', 'passthrough', ['cat'])
        ])),
        ('model', LogisticRegression())
    ])
    raw_param_grid = {
        'preprocessor__num__scaler__scale': [1, 2],
        'model__C': [1, 2]
    }
    model_param_grid = filter_param_grid(pipeline, raw_param_grid)
    assert isinstance(model_param_grid, dict)
    assert len(model_param_grid) == 1
    assert 'model__C' in model_param_grid

def test_tune_hyperparameters():
    '''test the tune_hyperparameters function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    model_param_grid = filter_param_grid(pipeline, {})
    tuning_cv = 2
    primary_metric = "accuracy"
    random_state = 42
    tuned_model = tune_hyperparameters(pipeline, model_param_grid, X_train, y_train, tuning_cv, primary_metric, random_state)
    assert isinstance(tuned_model, Pipeline)

def test_evaluate_metrics():
    '''test the evaluate_metrics function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    
    # Fit the pipeline first so it can predict
    pipeline.fit(X_train, y_train)

    metrics = evaluate_metrics(
        model=pipeline,
        model_name="LogisticRegression",
        fold_idx=1,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        metrics=['accuracy', 'f1'],
        metric_funcs={'accuracy': accuracy_score, 'f1': f1_score}
    )
    assert isinstance(metrics, dict)
    assert len([col for col in metrics.keys() if col not in ["Model", "Fold"]]) == 4  # Model, Fold, train_accuracy, test_accuracy, train_f1, test_f1 (excluding Model & Fold in length? No, it has keys: Model, Fold, train_accuracy, test_accuracy, train_f1, test_f1 -> 6 items)
    assert len([col for col in metrics.keys() if col not in ["Model", "Fold"]]) >= 4
    assert 'train_accuracy' in metrics
    assert 'test_accuracy' in metrics

def test_cross_validation_model():
    '''test the cross_validation_model function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    model_name = "LogisticRegression"
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    model_param_grid = filter_param_grid(pipeline, {})
    tuning_cv = 2
    primary_metric = "accuracy"
    random_state = 42
    cv_splitter = StratifiedKFold(n_splits=tuning_cv, shuffle=True, random_state=random_state)
    fold_results = cross_validate_model(
        model_name=model_name,
        pipeline=pipeline,
        model_param_grid=model_param_grid,
        X_train=X_train,
        y_train=y_train,
        cv_splitter=cv_splitter,
        tuning_cv=tuning_cv,
        primary_metric=primary_metric,
        random_state=random_state,
        metrics=['accuracy', 'f1'],
        metric_funcs={'accuracy': accuracy_score, 'f1': f1_score}
    )
    assert isinstance(fold_results, list)
    assert len(fold_results) == tuning_cv

def test_train_model_cv():
    '''test the train_model_cv function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target = "Survived"
    features = define_features(df, target)
    task_type, categorical_features, numerical_features = define_column_types(df, target, features)
    poly_degree = 2
    model_class = LogisticRegression
    model_name = "LogisticRegression"
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    model_param_grid = filter_param_grid(pipeline, {})
    tuning_cv = 2
    primary_metric = "accuracy"
    random_state = 42
    scores_df, best_estimator = train_model_cv(
        df=df,
        target=target,
        features=features,
        model_name="LogisticRegression",
        pipeline=pipeline,
        param_grid=model_param_grid,
        cv_splits=2,
        tuning_cv=2,
        primary_metric=primary_metric,
        random_state=random_state,
        metrics=['accuracy', 'f1'],
        metric_funcs={'accuracy': accuracy_score, 'f1': f1_score},
        task_type=task_type,
        experiment_name="LogisticRegression"
    )
    assert isinstance(scores_df, pd.DataFrame)
    assert len(scores_df) == 2

def test_data_prep():
    '''test the data_prep function'''
    df = create_initial_df()
    target_column = "Survived"
    features = define_features(df, target_column)
    # data_prep returns: df_train, df_test, features, target, task_type, cat_features, num_features, metrics
    res = data_prep(df, target_column, features)
    assert len(res) == 8
    assert isinstance(res[0], pd.DataFrame)
    assert isinstance(res[1], pd.DataFrame)

def test_run_model_selection():
    '''test the run_model_selection function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    model_name = "LogisticRegression"
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    model_param_grid = filter_param_grid(pipeline, {})
    tuning_cv = 2
    primary_metric = "accuracy"
    random_state = 42

    scores_df, best_estimator = train_model_cv(
        df=df,
        features=features,
        target=target_column,
        model_name="LogisticRegression",
        pipeline=pipeline,
        param_grid=model_param_grid,
        cv_splits=2,
        tuning_cv=2,
        random_state=random_state,
        metrics=['accuracy', 'f1'],
        metric_funcs={'accuracy': accuracy_score, 'f1': f1_score},
        primary_metric=primary_metric,
        task_type=task_type,
        experiment_name="LogisticRegression"
    )
    assert isinstance(scores_df, pd.DataFrame)
    assert len(scores_df) == 2

def test_fit_final_model():
    '''test the fit_final_model function'''
    df = create_initial_df().dropna(subset=["Survived", "Age", "Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked"])
    target_column = "Survived"
    features = define_features(df, target_column)
    task_type, categorical_features, numerical_features = define_column_types(df, target_column, features)
    poly_degree = 2
    model_class = LogisticRegression
    model_name = "LogisticRegression"
    default_params = {}
    model_type = "classification"
    pipeline = feature_engineering_pipeline(numerical_features, categorical_features, poly_degree, model_class, model_type, default_params)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    model_param_grid = filter_param_grid(pipeline, {})
    tuning_cv = 2
    primary_metric = "accuracy"
    random_state = 42

    scores_df, best_estimator = train_model_cv(
        df=df,
        features=features,
        target=target_column,
        model_name="LogisticRegression",
        pipeline=pipeline,
        param_grid=model_param_grid,
        cv_splits=2,
        tuning_cv=2,
        random_state=random_state,
        metrics=['accuracy', 'f1'],
        metric_funcs={'accuracy': accuracy_score, 'f1': f1_score},
        primary_metric=primary_metric,
        task_type=task_type,
        experiment_name="LogisticRegression"
    )
    assert isinstance(scores_df, pd.DataFrame)
    assert len(scores_df) == 2