FROM python:3.12-slim
RUN pip install uv
WORKDIR /app
COPY . .
RUN uv sync --no-dev
CMD uv run adk api_server . --host 0.0.0.0 --port ${PORT:-8080}