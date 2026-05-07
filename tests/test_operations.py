import pandas as pd
from src.operations import apply_operation, apply_operations

def test_apply_operation_dropna():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    op = {"op": "dropna", "params": {"axis": 0, "subset": ["A"]}}
    res = apply_operation(df, op)
    assert len(res) == 2
    assert pd.isna(res["A"]).sum() == 0

def test_apply_operation_fillna():
    df = pd.DataFrame({"A": [1, None, 3]})
    op = {"op": "fillna", "params": {"column": "A", "strategy": "constant", "value": 0}}
    res = apply_operation(df, op)
    assert len(res) == 3
    assert res["A"].iloc[1] == 0

def test_apply_operation_drop_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    op = {"op": "drop_columns", "params": {"columns": ["A"]}}
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
        {"op": "drop_columns", "params": {"columns": ["B"]}}
    ]
    res = apply_operations(df, ops)
    assert "B" not in res.columns
    assert res["A"].isna().sum() == 0
    assert res["A"].iloc[1] == 0
