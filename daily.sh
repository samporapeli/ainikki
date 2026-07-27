#!/usr/bin/env bash
# Daily digest: collect yesterday → build site → deploy → send Telegram.
# Usage: (see README)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
SITE_BASE_URL="https://sampo.website/ainikki"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Error: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

if [ -z "${DEPLOY_TARGET:-}" ]; then
  echo "Error: DEPLOY_TARGET is not set." >&2
  exit 1
fi

YESTERDAY="$(date -d yesterday +%Y-%m-%d)"
TODAY="$(date +%Y-%m-%d)"

send_error() {
  local msg="$1"
  if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_ERROR_CHAT_ID:-}" ]; then
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      --data-urlencode "text=${msg}" \
      -d chat_id="${TELEGRAM_ERROR_CHAT_ID}" > /dev/null
  fi
}

echo "== Collect and process yesterday's news ($YESTERDAY) =="
cd "$SCRIPT_DIR"
PIPELINE_LOG=$(mktemp)
PIPELINE_EXIT=0
"$VENV_PYTHON" -m agent.pipeline \
  --topic ai \
  --since "$YESTERDAY" \
  --until "$YESTERDAY" \
  --display-date "$TODAY" \
  --verbose 2> >(tee "$PIPELINE_LOG" >&2) || PIPELINE_EXIT=$?

if [ "$PIPELINE_EXIT" -ne 0 ]; then
  ERROR_MSG=$(tail -1 "$PIPELINE_LOG")
  rm -f "$PIPELINE_LOG"
  send_error "Ainikki epäonnistui ($YESTERDAY): $ERROR_MSG"
  exit "$PIPELINE_EXIT"
fi
rm -f "$PIPELINE_LOG"

echo "== Build and deploy =="
DEPLOY_LOG=$(mktemp)
DEPLOY_EXIT=0
"$SCRIPT_DIR/site/deploy.sh" 2> >(tee "$DEPLOY_LOG" >&2) || DEPLOY_EXIT=$?

if [ "$DEPLOY_EXIT" -ne 0 ]; then
  ERROR_MSG=$(tail -1 "$DEPLOY_LOG")
  rm -f "$DEPLOY_LOG"
  send_error "Ainikki deploy epäonnistui ($YESTERDAY): $ERROR_MSG"
  exit "$DEPLOY_EXIT"
fi
rm -f "$DEPLOY_LOG"

# --- Telegram notification ---
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  JSON_FILE="data/output/ai_daily_${TODAY}.json"
  if [ -f "$JSON_FILE" ]; then
    OVERVIEW=$("$VENV_PYTHON" -c "
import json, sys
d = json.load(open('$JSON_FILE'))
print(d.get('overview', 'No overview available.'))
")
    DISPLAY_DATE=$("$VENV_PYTHON" -c "
import json, sys
d = json.load(open('$JSON_FILE'))
print(d.get('display_date_fi', '$TODAY'))
")
    LINK="${SITE_BASE_URL}/ai/${TODAY}/"
    MSG="${DISPLAY_DATE}

${OVERVIEW}

${LINK}"

    echo "== Sending Telegram notification =="
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      -d chat_id="${TELEGRAM_CHAT_ID}" \
      -d text="${MSG}" \
      -d parse_mode="Markdown" \
      --data-urlencode "text=${MSG}" > /dev/null
    echo "Telegram: done."
  else
    echo "Warning: ${JSON_FILE} not found, Telegram message not sent." >&2
  fi
else
  echo "Telegram: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set, skipping."
fi
