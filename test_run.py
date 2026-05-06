import os
from src.llm_utils import call_llm, load_config, get_project_root

from dotenv import load_dotenv
load_dotenv()

try:
    prompts_path = get_project_root() / "configs" / "llm_prompts.yml"
    PROMPTS = load_config(str(prompts_path))
except Exception:
    PROMPTS = {}

if __name__ == "__main__":
    print(f"Testing call_llm...")
    print(f"NEBIUS_API_KEY is {'set' if os.environ.get('NEBIUS_API_KEY') else 'NOT set'}")
    
    try:
        response = call_llm(
            system_prompt=PROMPTS.get("test_run", {}).get("system_prompt", "You are a helpful assistant."),
            user_prompt=PROMPTS.get("test_run", {}).get("user_prompt", "Say the word 'success' and nothing else.")
        )
        print("Response received:")
        print(response)
    except Exception as e:
        print("Failed to call LLM:")
        print(e)
