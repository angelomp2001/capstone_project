import pandas as pd
from src.generate_sample_df import generate_sample_df
from src.text_parser import llm_parses_to_ops

# Add tests for llm_parses_to_ops here in the future
def test_llm_parses_to_ops():
    df = generate_sample_df(n_rows=10)
    ops = llm_parses_to_ops(
        user_text="fill missing uniforms with median", 
        df=df,
        temperature=0.0
        )
    assert isinstance(ops, list)
    assert len(ops) > 0
    assert ops[0]["op"] == "fillna"
    assert ops[0]["params"]["column"] == "uniform"
    assert ops[0]["params"]["strategy"] == "median"
    
