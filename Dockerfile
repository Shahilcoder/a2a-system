# One image for every service: `python -m a2a_system <service>` picks which one runs.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"

# Dependencies first (cached layer), then the project itself (installed editable, so a mounted ./src is used).
# The dev group (pytest) is included on purpose: this image is for local use and also runs the tests.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project
COPY src ./src
RUN uv sync --frozen
COPY tests ./tests

ENTRYPOINT ["python", "-m", "a2a_system"]
