import pandas as pd
import pytest
from src.operations import apply_operation, apply_operations

def test_apply_operation_dropna():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    op = {"op": "dropna", "params": {"axis": 0, "subset": ["A"]}}
    res = apply_operation(df, op)
    assert len(res) == 2
    assert pd.isna(res["A"]).sum() == 0

@pytest.mark.dependency(name="test_apply_operation_fillna", scope="session")
def test_apply_operation_fillna():
    df = pd.DataFrame({"A": [1, None, 3]})
    op = {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}}
    res = apply_operation(df, op)
    assert len(res) == 3
    assert res["A"].iloc[1] == 0

def test_apply_operation_drop_column():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    op = {"op": "drop_column", "params": {"columns": ["A"]}}
    res = apply_operation(df, op)
    assert "A" not in res.columns
    assert "B" in res.columns

def test_apply_operation_replace_value():
    df = pd.DataFrame({"A": ["old", "new", "old"]})
    op = {"op": "replace_value", "params": {"column": "A", "old_value": "old", "new_value": "changed"}}
    res = apply_operation(df, op)
    assert res["A"].tolist() == ["changed", "new", "changed"]

def test_apply_operations():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "z"]})
    ops = [
        {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}},
        {"op": "drop_column", "params": {"columns": ["B"]}}
    ]
    res = apply_operations(df, ops)
    assert "B" not in res.columns
    assert res["A"].isna().sum() == 0
    assert res["A"].iloc[1] == 0

def test_apply_operation_get_first_value_in_col():
    df = pd.DataFrame({"A": ["abc,def", "ghi,jkl", pd.NA]})
    op = {"op": "get_first_value_in_col", "params": {"column": "A", "split_by": ","}}
    res = apply_operation(df, op)
    assert "A_first_value" in res.columns
    assert res["A_first_value"].iloc[0] == "abc"
    assert res["A_first_value"].iloc[1] == "ghi"
    assert pd.isna(res["A_first_value"].iloc[2])

def test_apply_operation_split_alphanumeric():
    df = pd.DataFrame({"A": ["A123", "456B", "Invalid", pd.NA]})
    op = {"op": "split_alphanumeric", "params": {"column": "A"}}
    res = apply_operation(df, op)
    assert "A_left" in res.columns
    assert "A_right" in res.columns
    assert res["A_left"].iloc[0] == "A"
    assert res["A_right"].iloc[0] == 123
    assert res["A_left"].iloc[1] == 456
    assert res["A_right"].iloc[1] == "B"
    assert pd.isna(res["A_left"].iloc[2])
    assert pd.isna(res["A_right"].iloc[2])
    assert pd.isna(res["A_left"].iloc[3])
    assert pd.isna(res["A_right"].iloc[3])

