# Capstone Project POC

This project is a small Streamlit app for experimenting with AI-assisted data cleaning. It shows a sample table, lets you type a cleaning instruction in plain English, translates that instruction into one or more data operations, and then applies those operations to the table.

Right now the app supports these operations:

- `dropna`
- `fillna`
- `drop_columns`
- `replace_value`

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

## Fastest Local Run For A Beginner

If you have never run a Python app before, use these exact steps.

1. Download this repository to your computer.
2. Open the project folder in File Explorer.
3. Click the folder path bar, type `powershell`, and press Enter.
4. In that PowerShell window, run:

```powershell
python -m venv .venv
```

6. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

7. Install the app dependencies:

```powershell
pip install -r requirements.txt
```

8. Start the app:

```powershell
streamlit run app.py
```

9. Open your browser to [http://localhost:8501](http://localhost:8501).
10. When you are done, go back to PowerShell and press `Ctrl+C` to stop the app.

If PowerShell blocks the activate command, run this once in the same PowerShell window and then try step 6 again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Proper Terminal Workflow

This is the normal command-line way to run the project without Docker Desktop.

```powershell
cd C:\path\to\capstone_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

After the app starts, open [http://localhost:8501](http://localhost:8501).

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

The app now writes logs to plain text files in [docs/logs](/C:/Users/Angelo/Documents/github/capstone_project/docs/logs).

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

## Notes

- Ollama is optional for this POC.
- If Ollama is unavailable, the built-in parser handles simple instructions.
- This is still a POC, so unsupported prompts can happen. The new logs are there to make those failures visible and easier to reason about.
