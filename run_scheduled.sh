#!/bin/bash
# Scheduled autonomous run — logs to timestamped files
cd /home/asdf/PLGenesis-Market-Agent
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/runs"
mkdir -p "$LOG_DIR"

echo "[$TIMESTAMP] Starting autonomous run..." >> "$LOG_DIR/schedule.log"
python3 main.py --autonomous > "$LOG_DIR/run_${TIMESTAMP}.log" 2>&1
EXIT_CODE=$?
echo "[$TIMESTAMP] Complete (exit: $EXIT_CODE)" >> "$LOG_DIR/schedule.log"

# Copy agent_log.json with timestamp for track record
if [ -f agent_log.json ]; then
    cp agent_log.json "$LOG_DIR/agent_log_${TIMESTAMP}.json"
fi
