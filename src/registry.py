from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from scipy.stats import loguniform

MODEL_REGISTRY = {
    "regression": {
        "Linear": {"class": LinearRegression, "params": {}, "type": "non-tree-based"},
        "Ridge": {
            "class": Ridge, 
            "params": {"model__alpha": loguniform(1e-3, 1e3)}, 
            "type": "non-tree-based"
        },
        "Lasso": {
            "class": Lasso, 
            "params": {"model__alpha": loguniform(1e-3, 1e3)}, 
            "type": "non-tree-based"
        },
        "ElasticNet": {
            "class": ElasticNet, 
            "params": {
                "model__alpha": loguniform(1e-3, 1e3),
                "model__l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
            }, 
            "type": "non-tree-based"
        },
        "SVR": {
            "class": SVR, 
            "params": {
                "model__C": loguniform(1e-3, 1e3),
                "model__gamma": ["scale", "auto"]
            }, 
            "type": "non-tree-based"
        },
        "DecisionTree": {
            "class": DecisionTreeRegressor,
            "params": {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },
            "type": "tree-based"
        },
        "RandomForest": {
            "class": RandomForestRegressor, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5, 10]
            }, 
            "type": "tree-based"
        },
        "GradientBoosting": {
            "class": GradientBoostingRegressor, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            }, 
            "type": "tree-based"
        },
        "XGBoost": {
            "class": XGBRegressor, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            }, 
            "type": "tree-based"
        },
        "LightGBM": {
            "class": LGBMRegressor, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__num_leaves": [31, 63, 127]
            }, 
            "type": "tree-based"
        }
    },
    "classification": {
        "Logistic": {
            "class": LogisticRegression, 
            "params": {
                "model__C": loguniform(1e-3, 1e3),
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"]
            }, 
            "type": "non-tree-based"
        },
        "SVC": {
            "class": SVC, 
            "default_params": {"probability": True},
            "params": {
                "model__C": loguniform(1e-3, 1e3),
                "model__gamma": ["scale", "auto"]
            }, 
            "type": "non-tree-based"
        },
        "DecisionTree": {
            "class": DecisionTreeClassifier,
            "params": {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },
            "type": "tree-based"
        },
        "RandomForest": {
            "class": RandomForestClassifier, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5, 10]
            }, 
            "type": "tree-based"
        },
        "GradientBoosting": {
            "class": GradientBoostingClassifier, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            }, 
            "type": "tree-based"
        },
        "XGBoost": {
            "class": XGBClassifier, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7]
            }, 
            "type": "tree-based"
        },
        "LightGBM": {
            "class": LGBMClassifier, 
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__num_leaves": [31, 63, 127]
            }, 
            "type": "tree-based"
        }
    }
}