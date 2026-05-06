import os
from unittest.mock import patch, MagicMock
from src.llm_utils import call_llm, call_llm_for_json, load_config, get_project_root

try:
    prompts_path = get_project_root() / "configs" / "llm_prompts.yml"
    PROMPTS = load_config(str(prompts_path))
except Exception:
    PROMPTS = {}

@patch("src.llm_utils.OpenAI")
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

@patch("src.llm_utils.call_llm")
def test_call_llm_for_json(mock_call_llm):
    mock_call_llm.return_value = '{"status": "ok"}'
    
    result = call_llm_for_json(
        system_prompt=PROMPTS.get("tests", {}).get("call_llm_for_json", {}).get("system_prompt", "You are a json bot."),
        user_prompt=PROMPTS.get("tests", {}).get("call_llm_for_json", {}).get("user_prompt", "Return some JSON"),
    )
    
    assert result == {"status": "ok"}