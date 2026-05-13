from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


MODEL_REGISTRY = {
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
}

MODEL_GROUPS = {
    "non-tree-based": ["Linear", "Ridge", "Lasso", "ElasticNet", "SVR"],
    "tree-based": ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
}