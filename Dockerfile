FROM python:3.11-slim

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install curl and other required tools
RUN apt-get update && apt-get install -y curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull the model at build time (optional but recommended)
# This makes the image larger but avoids a long initial pull on first run
RUN ollama pull llama3.1

# Create app directory
WORKDIR /app

# Python env vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Streamlit env vars
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .
COPY src ./src

# Expose Streamlit port
EXPOSE 8501

# Start both Ollama and Streamlit
# 1) start ollama in the background
# 2) wait a bit for ollama to be ready
# 3) run streamlit
CMD /bin/sh -c "\
    ollama serve & \
    sleep 5 && \
    streamlit run app.py --server.address=0.0.0.0 --server.port=8501 \
"