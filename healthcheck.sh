#!/bin/bash
# Hoopla health check — tests the Streamlit WebSocket script health endpoint
# Returns 0 if healthy, non-zero if unhealthy.

PORT=8501
HEALTH_URL="http://localhost:${PORT}/_stcore/health"
STREAM_URL="http://localhost:${PORT}/_stcore/stream"
LOG_FILE="/home/nachiket/projects/hoopla/healthcheck.log"
MAX_LOG_SIZE_MB=5

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

# Rotate log if needed
if [ -f "$LOG_FILE" ]; then
    size=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$size" -gt $((MAX_LOG_SIZE_MB * 1024 * 1024)) ]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
    fi
fi

# Step 1: Basic HTTP health check
health_response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_URL" 2>/dev/null)
if [ "$health_response" != "200" ]; then
    log "FAIL: /health returned ${health_response}, restarting..."
    systemctl --user restart hoopla 2>/dev/null || \
        /home/nachiket/projects/hoopla/start.sh
    exit 1
fi

# Step 2: WebSocket endpoint reachability
# Streamlit's Tornado responds 405 (Method Not Allowed) to non-WebSocket requests
# on the /_stcore/stream endpoint. A timeout or connection refused means the
# script thread is dead while Tornado is still alive.
ws_check=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    -H "Connection: Upgrade" -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
    "$STREAM_URL" 2>/dev/null)

if [ "$ws_check" != "405" ] && [ "$ws_check" != "101" ]; then
    log "FAIL: /_stcore/stream returned ${ws_check} (expected 405 or 101), restarting..."
    systemctl --user restart hoopla 2>/dev/null || \
        /home/nachiket/projects/hoopla/start.sh
    exit 1
fi

# Step 3: Memory check — restart if RSS exceeds 1.5GB
pid=$(pgrep -f "streamlit run.*8501" | head -1)
if [ -n "$pid" ]; then
    rss_kb=$(awk '/VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null)
    if [ -n "$rss_kb" ] && [ "$rss_kb" -gt 1500000 ]; then
        log "FAIL: RSS ${rss_kb}KB exceeds 1.5GB, restarting..."
        systemctl --user restart hoopla 2>/dev/null || \
            /home/nachiket/projects/hoopla/start.sh
        exit 1
    fi
fi

exit 0
