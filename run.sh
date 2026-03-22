#!/bin/bash
# EN↔JP Translator — start script
# Usage: bash run.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

# Load secrets from .env (never commit .env to git)
if [ -f "$DIR/.env" ]; then
    set -a; source "$DIR/.env"; set +a
else
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your keys."
    exit 1
fi

PID_FILE="$DIR/app.pid"
LOG="$DIR/app.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running → http://localhost:$PORT"
    exit 0
fi

# Install deps if needed
if ! python3 -c "import fastapi, anthropic, uvicorn" 2>/dev/null; then
    echo "Installing dependencies…"
    pip3 install -r "$DIR/requirements.txt" -q
fi

echo "Starting EN↔JP Translator → http://localhost:$PORT"

(
    while true; do
        echo "[$(date)] Starting…" >> "$LOG"
        python3 "$DIR/app.py" >> "$LOG" 2>&1
        echo "[$(date)] Exited — restarting in 3s…" >> "$LOG"
        sleep 3
    done
) &

LOOP_PID=$!
echo $LOOP_PID > "$PID_FILE"
disown $LOOP_PID

sleep 3
if curl -s http://localhost:$PORT/api/status > /dev/null 2>&1; then
    echo "✓ App is up → http://localhost:$PORT"
else
    echo "⚠ Still starting — check app.log"
fi
