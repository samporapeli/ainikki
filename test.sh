#!/usr/bin/env bash
# Run the test topic without deploying it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$SCRIPT_DIR/lib.sh"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Error: OPENROUTER_API_KEY is not set." >&2
    exit 1
fi
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID_TEST:-}" ]; then
    echo "Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID_TEST are required." >&2
    exit 1
fi

# The RSS fixtures are dated 2026-08-21; override this only with matching data.
TEST_DATE="${AINIKKI_TEST_DATE:-2026-08-21}"
SITE_BASE_URL="${AINIKKI_SITE_BASE_URL:-https://sampo.website/ainikki}"

cd "$SCRIPT_DIR"
python3 -m http.server 8765 --bind 127.0.0.1 --directory tests/fixtures/test_rss \
    >/tmp/ainikki-test-rss.log 2>&1 &
RSS_SERVER_PID=$!
trap 'kill "$RSS_SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do
    if curl -fsS http://127.0.0.1:8765/feed-a.xml >/dev/null; then break; fi
    sleep 0.1
done

echo "== Collect and process test news ($TEST_DATE) =="
run_pipeline "test" "daily" "$TEST_DATE" "$TEST_DATE" "$TEST_DATE"
AUDIO_PATH="data/output/test_daily_${TEST_DATE}_audio.ogg"
if [ -f "$AUDIO_PATH" ]; then
    ls -la "data/output/" >&2
fi
send_telegram "test" "daily" "$TEST_DATE" "$TELEGRAM_CHAT_ID_TEST" "$AUDIO_PATH"

mkdir -p site/public/data/output site/public/data/pipeline
rsync -a data/output/ site/public/data/output/
rsync -a data/pipeline/ site/public/data/pipeline/

echo "Test output: data/output/test_daily_${TEST_DATE}.json"
echo "Test pipeline data: data/pipeline/test_daily_${TEST_DATE}.json"
echo "No deployment was performed."
