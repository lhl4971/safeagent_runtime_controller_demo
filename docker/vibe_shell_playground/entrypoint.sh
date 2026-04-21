#!/usr/bin/env bash
set -e

APP_TARGET="/home/ubuntu/mcp/vibeshell.py"
APP_SOURCE_MOUNT="/input/vibe_shell.py"
APP_SOURCE_FALLBACK="/opt/bootstrap/vibe_shell.py"
LOG_FILE="/home/ubuntu/mcp/vibe_shell.log"
PYTHON_BIN="/home/ubuntu/mcp/.venv/bin/python"

mkdir -p /home/ubuntu/mcp
touch "$LOG_FILE"

# Prefer the externally provided application file if mounted
if [ -f "$APP_SOURCE_MOUNT" ]; then
    cp "$APP_SOURCE_MOUNT" "$APP_TARGET"
elif [ -f "$APP_SOURCE_FALLBACK" ]; then
    cp "$APP_SOURCE_FALLBACK" "$APP_TARGET"
else
    echo "[entrypoint] No vibe_shell.py found." >> "$LOG_FILE"
    exit 1
fi

chmod 644 "$APP_TARGET"

# Start the SSH daemon
sudo /usr/sbin/sshd

# Start the MCP server and append logs to the log file
nohup "$PYTHON_BIN" "$APP_TARGET" >> "$LOG_FILE" 2>&1 &

# Keep the container alive and stream logs
tail -F "$LOG_FILE"