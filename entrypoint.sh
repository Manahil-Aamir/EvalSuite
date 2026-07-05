#!/bin/bash
set -e

# Default to running all three if MODE is not specified or is "all"
MODE=${MODE:-all}

echo "Starting EvalSuite System in mode: $MODE"

if [ "$MODE" = "all" ] || [ "$MODE" = "agent" ]; then
    echo "Starting KidsLearn support agent on port 8001..."
    cd /app/kidslearn-agent
    uv run python -c "import uvicorn; from app.fast_api_app import app; uvicorn.run(app, host='0.0.0.0', port=8001)" &
    cd /app
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "evalsuite" ]; then
    echo "Starting EvalSuite pipeline on port 8080..."
    cd /app/evalsuite
    uv run adk web . --host 0.0.0.0 --port 8080 &
    cd /app
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "dashboard" ]; then
    echo "Starting Streamlit dashboard on port 8502..."
    cd /app/dashboard
    uv run streamlit run app.py --server.port 8502 --server.address 0.0.0.0 &
    cd /app
fi

# Wait for background processes
wait
