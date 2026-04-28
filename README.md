# Capstone Project POC

This repository contains a small Streamlit proof of concept for LLM-assisted data cleaning. It starts with a sample pandas DataFrame, lets a user type a cleaning instruction, translates that instruction into a supported operation, and applies the result to the data preview.

For this first POC, the supported operations are:

- `dropna`
- `fillna`
- `drop_columns`

The app can use Ollama when it is available locally, but it also includes a built-in fallback parser so the demo still works in GitHub Actions and Docker without requiring a local model server.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in your terminal, usually `http://localhost:8501`.

## Run with Docker

Build the container:

```bash
docker build -t capstone-project-poc .
```

Run it:

```bash
docker run --rm -p 8501:8501 capstone-project-poc
```

Then open `http://localhost:8501`.

## GitHub Actions

The workflow in [.github/workflows/poc.yml](/C:/Users/Angelo/Documents/github/capstone_project/.github/workflows/poc.yml) does two things on every push and pull request:

- installs dependencies and runs [scripts/validate_ci.py](/C:/Users/Angelo/Documents/github/capstone_project/scripts/validate_ci.py)
- builds the Docker image to confirm containerization stays healthy

You can run the same non-Docker validation locally with:

```bash
python scripts/validate_ci.py
```

## Notes

- Ollama is optional for this POC. If it is not running, the app uses a simple built-in parser for common instructions like dropping columns, dropping missing rows, and filling missing values.
