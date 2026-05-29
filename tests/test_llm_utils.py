import os
from unittest.mock import patch, MagicMock
import pytest
# pyrefly: ignore [missing-import]
from src.llm.llm_utils import call_llm, call_llm_for_json, load_config_yml, get_project_root, is_llm_available
from src.generate_sample_df import generate_sample_df
from src.llm.text_parser import llm_parses_to_ops
import numpy as np


try:
    prompts_path = get_project_root() / "configs" / "llm_prompts.yml"
    PROMPTS = load_config_yml(str(prompts_path))
except Exception:
    PROMPTS = {}

@patch("src.llm.llm_utils.OpenAI")
def test_call_llm(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Paris"
    mock_client.chat.completions.create.return_value = mock_response
    
    result = call_llm(
        system_prompt=PROMPTS.get("tests", {}).get("call_llm", {}).get("system_prompt", "You are a helpful assistant."),
        user_prompt=PROMPTS.get("tests", {}).get("call_llm", {}).get("user_prompt", "What is the capital of France?"),
    )
    
    assert result == "Paris"
    mock_client.chat.completions.create.assert_called_once()

@patch("src.llm.llm_utils.OpenAI")
def test_call_llm_for_json(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"status": "ok"}'
    mock_client.chat.completions.create.return_value = mock_response

    result = call_llm_for_json(
        system_prompt=PROMPTS.get("tests", {}).get("call_llm_for_json", {}).get("system_prompt", "You are a json bot."),
        user_prompt=PROMPTS.get("tests", {}).get("call_llm_for_json", {}).get("user_prompt", "Return some JSON"),
    )

    assert result == {"status": "ok"}
    mock_client.chat.completions.create.assert_called_once()

@patch("src.llm.llm_utils.call_llm")
def test_is_llm_available(mock_call_llm):
    mock_call_llm.return_value = "Paris"
    
    assert is_llm_available()
    mock_call_llm.assert_not_called()

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
    assert isinstance(ops, list)
    assert len(ops) > 0
    assert ops[0]["op"] == "fillna"
    assert ops[0]["params"]["column"] == "uniform"
    assert ops[0]["params"]["strategy"] == "median"