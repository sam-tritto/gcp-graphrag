FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency configuration
COPY pyproject.toml .

# Compile requirements from pyproject.toml and install into the system layer
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Copy application source code
COPY . .

# Cloud Run routes external ingress traffic to port 8080
EXPOSE 8080
CMD ["python", "app.py"]
