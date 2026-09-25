#!/bin/bash
# Start hoopla Streamlit app in the background
# Can be called by cron, systemd, or manually.

set -e

PROJECT_DIR="/home/nachiket/projects/hoopla"
cd "$PROJECT_DIR"

# Check for active user sessions before restarting.
# Streamlit keeps a WebSocket open per user, so any non-localhost ESTABLISHED
# connection on port 8501 means someone is using the app.
active_sessions=$(ss -tn "sport = :8501" 2>/dev/null | tail -n +2 | grep -v "127.0.0.1" | wc -l)
if [ "$active_sessions" -gt 0 ] 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Skipping restart: ${active_sessions} active user session(s) detected"
    exit 0
fi

# Kill any existing hoopla processes on port 8501 (may be multiple PIDs)
existing_pids=$(pgrep -f "streamlit run.*8501" 2>/dev/null || true)
if [ -n "$existing_pids" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Killing existing hoopla PIDs: $(echo $existing_pids | tr '\n' ' ')"
    kill $existing_pids 2>/dev/null || true
    sleep 2
    kill -9 $existing_pids 2>/dev/null || true
fi

# Start hoopla
echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting hoopla..."
nohup .venv/bin/python .venv/bin/streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.address 127.0.0.1 \
    >> streamlit.log 2>&1 &

echo "$(date '+%Y-%m-%d %H:%M:%S') | Started hoopla PID $!"
