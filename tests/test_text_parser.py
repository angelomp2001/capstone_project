import numpy as np
from unittest.mock import patch
# pyrefly: ignore [missing-import]
from src.generate_sample_df import generate_sample_df
from src.llm.text_parser import llm_parses_to_ops
from src.llm.llm_utils import logger, is_llm_available, call_llm_for_json

# Add tests for llm_parses_to_ops here in the future
# @pytest.mark.dependency(
#     depends=["test_call_llm_for_json", "test_apply_operation_fillna"],
#     scope="session"
# )
@patch("src.llm.text_parser.call_llm_for_json")
@patch("src.llm.text_parser.is_llm_available", return_value=True)
def test_llm_parses_to_ops(mock_is_available, mock_call_llm_for_json):
    # Make the LLM "return" exactly what you expect for the parser
    mock_call_llm_for_json.return_value = {
        "op": "fillna",
        "params": {"column": "uniform", "strategy": "median"}
    }

    df = generate_sample_df(n_rows=10)
    df.loc[0, "uniform"] = np.nan

    ops = llm_parses_to_ops(
        user_text="fill missing uniform with median",
        df=df,
        temperature=0.0,
    )
    print("mock_call_llm_for_json:", mock_call_llm_for_json)
    logger.info("call_llm_for_json object: %r", call_llm_for_json)
    logger.info("is_llm_available() -> %s", is_llm_available())
    assert isinstance(ops, list)
    assert len(ops) > 0
    assert ops[0]["op"] == "fillna"
    assert ops[0]["params"]["column"] == "uniform"
    assert ops[0]["params"]["strategy"] == "median"
    
