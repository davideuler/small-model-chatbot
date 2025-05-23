FROM python:3.12-slim

# Install uv and other dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y build-essential gcc gfortran curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Create virtual environment and install dependencies with retry and alternative approaches
RUN (uv sync --index-url https://pypi.org/simple || \
     uv pip install -r requirements.txt --index-url https://pypi.org/simple) && \
    (uv add spaces || uv pip install spaces --index-url https://pypi.org/simple || \
     echo "Warning: space package installation failed, continuing without it")

ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["uv", "run", "app.py"]
