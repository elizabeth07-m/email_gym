# Use the official OpenEnv base image per hackathon spec
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Copy pyproject.toml first for dependency caching
COPY pyproject.toml .

# Install runtime dependencies (no GPU / train extras in server container)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.5" \
    "httpx>=0.27" \
    openai \
    "gradio>=4.26" \
    websockets

# Copy full source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Expose OpenEnv API port
EXPOSE 8000

# Serve the FastAPI OpenEnv server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
