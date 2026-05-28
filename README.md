# Capstone Project POC
This project is a small Streamlit app for experimenting with AI-assisted data cleaning and analysis. It shows a sample table, lets you type a cleaning instruction in plain English, translates that instruction into one or more data operations, and then applies those operations to the table. The user can also ask to model a target variable and submit feature values to get a prediction with an interpretation.

Right now the app supports these operations:

- `dropna`
- `fillna`
- `drop_columns`
- `replace_value`
- `rename_columns`
- `change_data_type`
- `extract_values`
- `model_target`

## Architecture
### src

The project originally included generating a sample df and then modeling it - hence they are in the root src folder.  Since the model registry was sophisticated, I kept it as its own py file.  

Files associated with the llm (call, parse to operations, the operations themselves) are grouped in the src/llm folder as llm_utils, operations, and text_parser, respectively.  This keeps file associted with the llm in one place, and each file only has one job.

### tools
DVC is used to track the data
Github Actions is used to auto test code as I go. 
Streamlit is used to host the ui.
MLflow is used to track model information. 
Docker is used to containerize the application for others to run. 
Poetry is used for dependency management. 

## How to add an operation:
- Add method(df, params) -> pd.DataFrame: to ApplyOperation class
  - **IMPORTANT**: add docstring to auto gen prompt
- add method to SUPPORTED_OPS tuple

# default data set
https://www.kaggle.com/datasets/pranjalyadav92905/titanic-eda-data

Titanic:
PassengerId: Passenger unique identifier.
Survived: binary target variable, 1 = survived, 0 = died
Pclass: proxy for the socio-economic status (SES) of the passenger.
- 1: upper class
- 2: middle class
- 3: lower class
Name: Passenger name
Sex: Passenger gender
Age: Passenger age
SibSp: Number of siblings or spouses aboard the Titanic
Parch: Number of parents or children aboard the Titanic
Ticket: Ticket number
Fare: Passenger fare
Cabin: Cabin number
Embarked: Port of embarkation

## github API key
Github has a copy of the API key to pass the CI/CD.

## Fastest Local Run
Option 1 — Run Locally (Recommended for Development)
1. Clone the repository
```
git clone <your-repo-url>
cd capstone_project
```
2. Install Poetry

Install Poetry if you do not already have it:

https://python-poetry.org/docs/#installation

Verify installation:
```
poetry --version
```
3. Install dependencies
```
poetry install
```
This installs all dependencies defined in:

pyproject.toml
poetry.lock

Poetry is the single source of truth for dependency versions.

4. Run the app
```
poetry run streamlit run app.py
```
The app should open automatically in your browser.

If not, visit:

http://localhost:8501

Option 2 — Run with Docker
1. Install Docker Desktop

Install:

https://www.docker.com/products/docker-desktop/

IMPORTANT:

Docker Desktop must be running before executing Docker commands.
Wait until Docker shows:
Engine running
2. Build the Docker image

From the repository root:
```
docker build --no-cache -t capstone-app .
```
3. Run the container
```
docker run -p 8501:8501 capstone-app
```
4. Open the app

Visit:

http://localhost:8501

### mlflow
Launch web interface:
```powershell
mlflow ui
```
view web interface:
http://127.0.0.1:5000



## Docker From The Terminal

If you want Docker, this is the command-line version rather than clicking around in Docker Desktop.

Build the image:

```powershell
docker build -t capstone-project .
```

Run the image:

```powershell
docker run --env-file .env --rm -p 8501:8501 capstone-project
```

Then open [http://localhost:8501](http://localhost:8501).

## How To Read Logs

The app writes logs to plain text files in [docs/logs](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs).

Important files:

- [app.log](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs/app.log): normal application logging
- [trace.log](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs/trace.log): very verbose step-by-step tracing

These logs are meant to help you debug after the fact, even if you closed the browser tab or were not watching the terminal.

Examples of what gets logged:

- when the app starts
- whether Ollama was detected
- the exact user instruction you typed
- the operations the parser decided to use
- each DataFrame operation that was applied
- the shape and preview of the data before and after changes
- parser failures and fallback behavior

## Example Debugging Workflow

If something weird happens:

1. Run the app.
2. Reproduce the problem.
3. Open [docs/logs/app.log](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs/app.log).
4. Open [docs/logs/trace.log](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs/trace.log).
5. Search for your prompt text and review what operation the parser actually chose.

For example, if you type `replace med category in risk_level with 1`, the logs should now show a `replace_value` operation instead of a strange `drop_columns` plus `fillna` sequence.

## GitHub Actions

The workflow file is [.github/workflows/poc.yml](/C:/Users/Angelo/Documents/github/capstone_project/.github/workflows/poc.yml).

It now runs:

- on every push
- on pull requests
- manually with `workflow_dispatch`

The workflow:

- installs dependencies
- runs [scripts/validate_ci.py](/C:/Users/Angelo/Documents/github/capstone_project/scripts/validate_ci.py)
- builds the Docker image

You can run the same local validation yourself:

```powershell
python scripts\validate_ci.py
```
