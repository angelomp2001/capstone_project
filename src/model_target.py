import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
import mlflow
from pathlib import Path

def data_splitter(df: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    """Outer split"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def define_features_by_type(df):
    """Define features by type"""
    # define features by type
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    return categorical_features, numerical_features


def define_target(df, target_col_name):
    """Define target"""
    target = df[target_col_name]
    return target

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
        ('no scaler', 'passthrough')
    ])

# tree-based models: categorical transformer
def tree_based_categorical_transformer():
    """Categorical transformer"""
    return Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

# non-tree-based models: preprocessor
def preprocessor(numerical_features, categorical_features, poly_degree):
    """Preprocessor"""
    prep = ColumnTransformer(transformers=[
        ('num', numerical_transformer(poly_degree), numerical_features),
        ('cat', categorical_transformer(), categorical_features)
    ])
    return prep

# tree-based models: preprocessor
def tree_based_preprocessor(numerical_features, categorical_features):
    """Preprocessor"""
    prep = ColumnTransformer(transformers=[
        ('num', tree_based_numerical_transformer(), numerical_features),
        ('cat', tree_based_categorical_transformer(), categorical_features)
    ])
    return prep

def build_pipeline(model_instance, preprocessor_obj):
    """
    Creates a scikit-learn Pipeline with the preprocessor and a.
    """    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_obj),
        ('model', model_instance)
    ])
    return pipeline

def train_model_cv(X_train, y_train, pipelines, cv_splits, random_state, metrics=None)->pd.DataFrame:
    """
    Evaluate multiple models using cross-validation over individual folds.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training target.
        pipelines (dict): Dictionary of model names to instantiated pipelines.
        cv_splits (int): Number of CV folds.
        random_state (int): Random state for reproducibility.
        metrics (list): List of Scikit-learn scoring strings.
        
    Returns:
        tuple (pd.DataFrame, pd.DataFrame): 
             - A dataframe containing model evaluation metrics for EVERY fold.
             - A dataframe containing model evaluation metrics averaged per model.
    """
    if metrics is None:
        metrics = ["neg_root_mean_squared_error"]
        
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []
    
    for name, model in pipelines.items():
        try:
            cv_results = cross_validate(
                model, X_train, y_train, 
                cv=kf, 
                scoring=metrics, 
                return_train_score=True
            )
            
            # Extract individual performance for every fold
            for fold_idx in range(cv_splits):
                fold_result = {
                    "Model": name,
                    "Fold": fold_idx + 1,
                    "Train Time (s)": cv_results['fit_time'][fold_idx],
                    "Pred Time (s)": cv_results['score_time'][fold_idx]
                }
                
                # Log all metrics systematically
                for metric_name in metrics:
                    test_val = cv_results[f'test_{metric_name}'][fold_idx]
                    train_val = cv_results[f'train_{metric_name}'][fold_idx]
                    
                    # Ensure minimization metrics visually invert to positive values for the user
                    if 'neg_' in metric_name:
                        test_val = -test_val
                        train_val = -train_val
                        
                    fold_result[f'test_{metric_name}'] = test_val
                    fold_result[f'train_{metric_name}'] = train_val
                    
                results.append(fold_result)
                
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # return fold_results as dataframe        
    return pd.DataFrame(results)

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
        return pd.DataFrame()

    agg_dict = {'Train Time (s)': 'mean', 'Pred Time (s)': 'mean'}
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