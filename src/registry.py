from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

MODEL_REGISTRY = {
    "regression": {
        "Linear": LinearRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "ElasticNet": ElasticNet,
        "SVR": SVR,
        "DecisionTree": DecisionTreeRegressor,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": XGBRegressor,
        "LightGBM": LGBMRegressor
    },
    "classification": {
        "Logistic": LogisticRegression,
        "SVC": SVC,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier
    }
}

MODEL_GROUPS = {
    "regression": {
        "non-tree-based": ["Linear", "Ridge", "Lasso", "ElasticNet", "SVR"],
        "tree-based": ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
    },
    "classification": {
        "non-tree-based": ["Logistic", "SVC"],
        "tree-based": ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
    }
}