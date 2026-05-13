# Dockerfile for lang-learn inference service
# Targets Apple Silicon (arm64) with MLX acceleration
#
# Build:
#   docker build -t lang-learn-inference ./inference
#
# Run (mount the model from the host):
#   docker run --rm -p 3000:3000 \
#     -v $(pwd)/models/gemma-mlx:/app/models/gemma-mlx:ro \
#     lang-learn-inference

FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY prompts/ prompts/

# Default model path (override with -e LANG_LEARN_MODEL_PATH=...)
ENV LANG_LEARN_MODEL_PATH=/app/models/gemma-mlx

EXPOSE 3000

CMD ["bentoml", "serve", "src.service:LangLearnService", "--host", "0.0.0.0", "--port", "3000"]
