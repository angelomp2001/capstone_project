import pandas as pd
import pytest
from src.llm.operations import apply_operation, apply_operations
from app import create_initial_df

def test_apply_operation_dropna():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    op = {"op": "dropna", "params": {"axis": 0, "subset": ["A"]}}
    new_df = apply_operation(df, op)
    assert len(new_df) == 2
    assert pd.isna(new_df["A"]).sum() == 0

@pytest.mark.dependency(name="test_apply_operation_fillna", scope="session")
def test_apply_operation_fillna():
    df = pd.DataFrame({"A": [1, None, 3]})
    op = {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}}
    new_df = apply_operation(df, op)
    assert len(new_df) == 3
    assert new_df["A"].iloc[1] == 0

def test_apply_operation_drop_column():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    op = {"op": "drop_column", "params": {"columns": ["A"]}}
    new_df = apply_operation(df, op)
    assert "A" not in new_df.columns
    assert "B" in new_df.columns

def test_apply_operation_replace_value():
    df = pd.DataFrame({"A": ["old", "new", "old"]})
    op = {"op": "replace_value", "params": {"column": "A", "old_value": "old", "new_value": "changed"}}
    new_df = apply_operation(df, op)
    assert new_df["A"].tolist() == ["changed", "new", "changed"]

def test_apply_operations():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "z"]})
    ops = [
        {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}},
        {"op": "drop_column", "params": {"columns": ["B"]}}
    ]
    new_df = apply_operations(df, ops)
    assert "B" not in new_df.columns
    assert new_df["A"].isna().sum() == 0
    assert new_df["A"].iloc[1] == 0

def test_apply_operation_get_first_value_in_col():
    df = pd.DataFrame({"A": ["abc,def", "ghi,jkl", pd.NA]})
    op = {"op": "get_first_value_in_col", "params": {"column": "A", "split_by": ","}}
    new_df = apply_operation(df, op)
    assert "A_first_value" in new_df.columns
    assert new_df["A_first_value"].iloc[0] == "abc"
    assert new_df["A_first_value"].iloc[1] == "ghi"
    assert pd.isna(new_df["A_first_value"].iloc[2])

def test_apply_operation_split_alphanumeric():
    df = pd.DataFrame({"A": ["A123", "456B", "Invalid", pd.NA]})
    op = {"op": "split_alphanumeric", "params": {"column": "A"}}
    new_df = apply_operation(df, op)
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

def test_apply_operation_model_target():
    #df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df = create_initial_df()
    op = {"op": "model_target", "params": {"column": "Survived"}}
    new_df = apply_operation(df, op)
    # check that A is in the new df
    #assert "A_target" in new_df.columns
    assert new_df.shape[0] > 0
    assert new_df.shape[1] > 0

test_apply_operation_model_target()