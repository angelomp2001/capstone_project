import numpy as np
import pandas as pd
import pytest
from src.generate_sample_df import generate_sample_df
from src.llm.text_parser import llm_parses_to_ops
# Add tests for llm_parses_to_ops here in the future
@pytest.mark.dependency(
    depends=["test_call_llm_for_json", "test_apply_operation_fillna"],
    scope="session"
)
def test_llm_parses_to_ops():
    df = generate_sample_df(n_rows=10)
    # Introduce missing data to make the user text logical
    df.loc[0, "Survived"] = np.nan
    
    ops = llm_parses_to_ops(
        user_text="fill missing Survived with median", 
        df=df,
        temperature=0.0
        )
    assert isinstance(ops, list)
    assert len(ops) > 0
    assert ops[0]["op"] == "fillna"
    assert ops[0]["params"]["column"] == "uniform"
    assert ops[0]["params"]["strategy"] == "median"
    
