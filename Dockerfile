# Builder
FROM python:3.11.8-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

# Copy dependency and source code
COPY ["pyproject.toml", "./"] 
COPY ["uv.lock", "./"]
COPY ["src/", "./src/"]
COPY ["README.md", "./"]

# Install dependecies
RUN uv sync --frozen --no-dev

# Runtime
FROM python:3.11.8-slim

RUN useradd -m -s /bin/bash appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ /app/src/
COPY model_package/ /app/model_package/

RUN chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

ENV MODEL_URI="/app/model_package"

USER appuser
EXPOSE 8000

CMD ["uvicorn", "diabetes_api.app:app", "--host", "0.0.0.0", "--port", "8000"]