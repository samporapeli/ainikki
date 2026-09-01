#!/usr/bin/env bash
# Daily digest: collect yesterday -> build site -> deploy -> send Telegram.
# Usage: (see README)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

validate_env

YESTERDAY="$(date -d yesterday +%Y-%m-%d)"
TODAY="$(date +%Y-%m-%d)"

echo "== Collect and process yesterday's news ($YESTERDAY) =="
cd "$SCRIPT_DIR"
run_pipeline "ai" "daily" "$YESTERDAY" "$YESTERDAY" "$TODAY"

echo "== Build and deploy =="
deploy_site

# --- Telegram notification ---
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID_AI:-}" ]; then
    AUDIO_PATH="data/output/ai_daily_${TODAY}_audio.ogg"
    send_telegram "ai" "daily" "$TODAY" "$TELEGRAM_CHAT_ID_AI" "$AUDIO_PATH"
else
    echo "Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID_AI not set, skipping."
fi
