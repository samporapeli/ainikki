#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 TOPIC" >&2
    exit 2
fi

TOPIC="$1"
validate_env

CONFIG_VALUES=$("$VENV_PYTHON" "$SCRIPT_DIR/agent/topic_config.py" \
    "$SCRIPT_DIR/config/topics/${TOPIC}.yaml")
IFS=$'\t' read -r PERIOD TELEGRAM_CHANNEL_ID_ENV <<< "$CONFIG_VALUES"

END_DATE=$(date -d yesterday +%Y-%m-%d)
case "$PERIOD" in
    daily) DAYS=1 ;;
    weekly) DAYS=7 ;;
    monthly) DAYS=30 ;;
esac
START_DATE=$(date -d "$END_DATE - $((DAYS - 1)) days" +%Y-%m-%d)
DISPLAY_DATE=$(date +%Y-%m-%d)

cd "$SCRIPT_DIR"
echo "== Collect and process $TOPIC ($START_DATE .. $END_DATE) =="
run_pipeline "$TOPIC" "$PERIOD" "$START_DATE" "$END_DATE" "$DISPLAY_DATE"

echo "== Build and deploy =="
deploy_site

CHAT_ID="${!TELEGRAM_CHANNEL_ID_ENV:-}"
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "$CHAT_ID" ]; then
    AUDIO_PATH="data/output/${TOPIC}_${PERIOD}_${DISPLAY_DATE}_audio.ogg"
    send_telegram "$TOPIC" "$PERIOD" "$DISPLAY_DATE" "$CHAT_ID" "$AUDIO_PATH"
else
    echo "Telegram: TELEGRAM_BOT_TOKEN/$TELEGRAM_CHANNEL_ID_ENV not set, skipping."
fi
