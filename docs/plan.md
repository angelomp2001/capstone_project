Purpose: use a chatbot to clean your data and model a target column.

Components:
chat ui: user prompts df-edit task
llm: llm receives prompt + views function list -> outputs function.
parse: extract JSON from text
convert: convert JSON to function list
apply: apply to df

# data cleaning functions:
copy_df
validate required
validate unique
validate categorical values
validate bounds
validate date range
drop duplicates
replace value
coerce dtype
dropna
fillna
drop_columns

# POC
0. logger and trace
1. chat converts instructions into function and applies to data
2. update documentation

entry point: app.py
streamlit web page, loads ui
ui: 
    llm status
    df preview
      generate_sample_df.py/generate_sample_df
    chat
        submit button
            text_parser.py/llm_parses_to_ops
                text_parser.py/_algo_parses_to_ops | llm_utils.py/call_llm_for_json_cached
            cleaning_operations.py/apply_operations
    reset button
    operations log

# MVP
1. formalize project folder structure a bit 


| Module | Input | Output | config/ | src/ | scripts/ | tests/ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| gen_df | ❌ | df | \[1\] | \[1\] |   | \[1\] |
| App | user_text | new_df |   | \[❌\] | \[1\] |   |
| text_parser | user_text, df | JSON | \[❌\] | \[1\] | \[❌\] | \[1\] |
| operations  | df, JSON  | new_df |  | \[1\] | \[❌\] | \[1\] |
| llm_utils | ❌ | ❌  | \[1\] | \[1\] | \[❌\] | \[1\] |
|  |  |  |  | \[   \] | \[   \] | \[   \] | \[   \] |

2. containerize app with model via API and API key in github secrets. 
test container: clone, docker build, docker run, test in browser. 
update readme. 

# Version 1:
upload df
label target
predict target, mlflow (parameters, metrics, and model artifacts)
- multiple models
- 3 metrics
- mlflow
- experiment comparison script (programmatically identify best model)
- choose best model
- choose a df to load (not random)

# Version 2:



### Evaluation Criteria Checklist

**Data and Model Quality: (20 points)**
- [ ] Data preprocessing is complete and well-documented (5 points)
  - [ ] Missing values handled
  - [ ] Categoricals encoded
  - [ ] Features scaled
  - [ ] No data leakage
- [ ] At least 3 model configurations trained and compared (5 points)
  - [ ] Configurations are meaningfully different
- [ ] At least 3 evaluation metrics reported per model (4 points)
  - [ ] Metrics appropriate to the task reported on held-out test data
- [ ] Best model selection is justified (3 points)
  - [ ] Clear reasoning for the selected model choice
- [ ] Model achieves reasonable performance for the task (3 points)
  - [ ] Performance is not trivially low; evidence of iteration to improve results

**Experiment Tracking: (15 points)**
- [ ] MLflow integrated into training script (4 points)
  - [ ] Parameters, metrics, and model artifacts logged correctly
- [ ] At least 5 experiment runs logged (4 points)
  - [ ] Five distinct runs with different configurations visible in tracking
- [ ] All hyperparameters and metrics logged (3 points)
  - [ ] Nothing relevant missing from the logs
- [ ] Experiment comparison script identifies best model (4 points)
  - [ ] Uses `mlflow.search_runs()` to query and rank experiments programmatically

**LLM Interface: (30 points)**
- [ ] LLM correctly parses natural language input into model features (8 points)
  - [ ] Extracts feature values accurately from conversational text
- [ ] Trained model is loaded and invoked with parsed features (6 points)
  - [ ] Actual trained model produces the prediction (not a mock)
- [ ] Response is clear, contextual, and includes the prediction (8 points)
  - [ ] LLM explains the result in domain context, not just a raw number
- [ ] Edge cases handled gracefully (5 points)
  - [ ] Missing features, ambiguous input, and out-of-scope queries are managed
- [ ] Interface is functional and easy to use (3 points)
  - [ ] User can interact with it without confusion (notebook, CLI, or web app)

**Testing: (15 points)**
- [ ] At least 4 preprocessing unit tests (5 points)
  - [ ] Tests cover missing values, encoding, scaling, and immutability
- [ ] At least 2 model validation tests (3 points)
  - [ ] Tests verify prediction shape/type and minimum performance
- [ ] At least 2 interface tests (4 points)
  - [ ] Tests verify input parsing accuracy and edge case handling
- [ ] All tests pass (3 points)
  - [ ] `pytest tests/ -v` shows zero failures

**Documentation and Structure: (20 points)**
- [ ] Clean, organized repository structure (4 points)
  - [ ] Logical folder layout, no unnecessary files committed
- [ ] README is complete and clear (6 points)
  - [ ] Covers project description, setup, usage, architecture, results, and reflection
- [ ] YAML config file used for training parameters (3 points)
  - [ ] No hardcoded hyperparameters in training script
- [ ] Data and model files excluded from Git (3 points)
  - [ ] Proper `.gitignore` or DVC setup
- [ ] Requirements file with pinned versions (2 points)
  - [ ] All dependencies listed and version-pinned
- [ ] Demo included (recording or live) (2 points)
  - [ ] Shows the application working end-to-end with at least one edge case
- [ ] Dockerfile included (bonus) (up to 3 points)
  - [ ] Working Dockerfile that builds and runs the application