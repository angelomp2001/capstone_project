import pandas as pd
import pytest
from unittest.mock import patch
# pyrefly: ignore [missing-import]
from src.llm.operations import apply_operation, apply_operations
from app import create_initial_df

def test_apply_operation_dropna():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    op = {"op": "dropna", "params": {"axis": 0, "subset": ["A"]}}
    new_df, msg = apply_operation(df, op)
    assert len(new_df) == 2
    assert pd.isna(new_df["A"]).sum() == 0
    assert "dropped" in msg or "dropped" in msg.lower()

@pytest.mark.dependency(name="test_apply_operation_fillna", scope="session")
def test_apply_operation_fillna():
    df = pd.DataFrame({"A": [1, None, 3]})
    op = {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}}
    new_df, msg = apply_operation(df, op)
    assert len(new_df) == 3
    assert new_df["A"].iloc[1] == 0
    assert "fillna" in msg

def test_apply_operation_drop_column():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    op = {"op": "drop_column", "params": {"columns": ["A"]}}
    new_df, msg = apply_operation(df, op)
    assert "A" not in new_df.columns
    assert "B" in new_df.columns
    assert "dropped" in msg or "dropped" in msg.lower()

def test_apply_operation_replace_value():
    df = pd.DataFrame({"A": ["old", "new", "old"]})
    op = {"op": "replace_value", "params": {"column": "A", "old_value": "old", "new_value": "changed"}}
    new_df, msg = apply_operation(df, op)
    assert new_df["A"].tolist() == ["changed", "new", "changed"]
    assert "replaced" in msg

def test_apply_operations():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "z"]})
    ops = [
        {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}},
        {"op": "drop_column", "params": {"columns": ["B"]}}
    ]
    new_df, msg = apply_operations(df, ops)
    assert "B" not in new_df.columns
    assert new_df["A"].isna().sum() == 0
    assert new_df["A"].iloc[1] == 0
    assert "dropped" in msg or "dropped" in msg.lower()

def test_apply_operation_get_first_value_in_col():
    df = pd.DataFrame({"A": ["abc,def", "ghi,jkl", pd.NA]})
    op = {"op": "get_first_value_in_col", "params": {"column": "A", "split_by": ","}}
    new_df, msg = apply_operation(df, op)
    assert "A_first_value" in new_df.columns
    assert new_df["A_first_value"].iloc[0] == "abc"
    assert new_df["A_first_value"].iloc[1] == "ghi"
    assert pd.isna(new_df["A_first_value"].iloc[2])
    assert "split" in msg

def test_apply_operation_split_alphanumeric():
    df = pd.DataFrame({"A": ["A123", "456B", "Invalid", pd.NA]})
    op = {"op": "split_alphanumeric", "params": {"column": "A"}}
    new_df, msg = apply_operation(df, op)
    assert "A_left" in new_df.columns
    assert "A_right" in new_df.columns
    assert new_df["A_left"].iloc[0] == "A"
    assert new_df["A_right"].iloc[0] == 123
    assert new_df["A_left"].iloc[1] == 456
    assert new_df["A_right"].iloc[1] == "B"
    assert pd.isna(new_df["A_left"].iloc[2])
    assert pd.isna(new_df["A_right"].iloc[2])
    assert pd.isna(new_df["A_left"].iloc[3])
    assert pd.isna(new_df["A_right"].iloc[3])
    assert "split" in msg

def test_apply_operation_model_target():
    df = create_initial_df()
    op = {"op": "model_target", "params": {"target": "Survived"}}
    new_df, msg = apply_operation(df, op)
    assert new_df.shape[0] > 0
    assert new_df.shape[1] > 0
    assert "Survived_hat" in new_df.columns
    assert msg == "I finished modeling Survived"

def test_apply_operation_model_target_no_target():
    df = create_initial_df()
    op = {"op": "model_target", "params": {"column": "Survived"}}
    new_df, msg = apply_operation(df, op)
    assert "I need a target column to work with" in msg
    assert "Survived_hat" not in new_df.columns

@patch("src.llm.operations.call_llm", return_value="The predicted Survived value is 0. This means the passenger did not survive.")
def test_apply_operation_model_target_classification_features(mock_llm):
    df = create_initial_df()
    op = {
        "op": "model_target",
        "params": {
            "target": "Survived",
            "features": {"Pclass": 3, "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25}
        }
    }
    new_df, msg = apply_operation(df, op)
    assert new_df.shape[0] > 0
    mock_llm.assert_called_once()
    assert msg == "The predicted Survived value is 0. This means the passenger did not survive."

@patch("src.llm.operations.call_llm", return_value="The predicted Age value is 28.5. This is the estimated age of the passenger.")
def test_apply_operation_model_target_regression_features(mock_llm):
    df = create_initial_df()
    op = {
        "op": "model_target",
        "params": {
            "target": "Age",
            "features": {"Pclass": 3, "SibSp": 1, "Parch": 0, "Fare": 7.25}
        }
    }
    new_df, msg = apply_operation(df, op)
    assert new_df.shape[0] > 0
    mock_llm.assert_called_once()
    assert msg == "The predicted Age value is 28.5. This is the estimated age of the passenger."

@patch("src.llm.operations.call_llm", return_value="The predicted Age value is 28.5. This is the estimated age of the passenger.")
def test_apply_operation_model_target_classification_features(mock_llm):
    df = create_initial_df()
    op = {
        "op": "model_target",
        "params": {
            "target": "Survived",
            "features": {"Pclass": 3, "SibSp": 1, "Parch": 0, "Fare": 7.25},
            # maybe "explain": True or False depending on your design
        }
    }
    new_df, msg = apply_operation(df, op)
    assert new_df.shape[0] > 0

    # Example if you want optional explanation:
    # If explain=False (default), then:
    mock_llm.assert_called_once()
    # mock_llm.assert_not_called()
    # and msg might be something like:
    # assert msg == "I finished modeling Age"