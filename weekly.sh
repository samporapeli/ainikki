#!/usr/bin/env bash
# Weekly digest: collect last week -> build site -> deploy -> send Telegram.
# Intended for Monday morning cron (processes the previous Mon-Sun).
# Usage: (see README)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

validate_env

THIS_MONDAY=$(date -d "$(( $(date +%u) - 1 )) days ago" +%Y-%m-%d)
WEEK_START=$(date -d "$THIS_MONDAY - 7 days" +%Y-%m-%d)
WEEK_END=$(date -d "$THIS_MONDAY - 1 day" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "== Collect and process last week's news ($WEEK_START .. $WEEK_END) =="
cd "$SCRIPT_DIR"
run_pipeline "teknologia" "weekly" "$WEEK_START" "$WEEK_END" "$TODAY"

echo "== Build and deploy =="
deploy_site

# --- Telegram notification ---
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID_TEKNOLOGIA:-}" ]; then
    AUDIO_PATH="data/output/teknologia_weekly_${TODAY}_audio.ogg"
    audio_path=$(prepare_audio "$AUDIO_PATH")
    send_telegram "teknologia" "weekly" "$TODAY" "$TELEGRAM_CHAT_ID_TEKNOLOGIA" "$audio_path"
else
    echo "Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID_TEKNOLOGIA not set, skipping."
fi