# Use a lightweight python image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY requirements.txt .

# Install dependencies globally
RUN uv pip install --system -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose all three default ports
# 8001: KidsLearn Agent
# 8080: EvalSuite ADK Web Server
# 8502: Streamlit Dashboard
EXPOSE 8001 8080 8502

# Make start script executable
RUN chmod +x entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
