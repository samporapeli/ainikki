#!/usr/bin/env bash
# Päivittäinen digest: kerää eilinen → builaa sivusto → deployaa.
# Käyttö: OPENROUTER_API_KEY=... DEPLOY_TARGET=user@host:/var/www/ainikki ./daily.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Virhe: OPENROUTER_API_KEY ei ole asetettu." >&2
  exit 1
fi

if [ -z "${DEPLOY_TARGET:-}" ]; then
  echo "Virhe: DEPLOY_TARGET ei ole asetettu." >&2
  exit 1
fi

YESTERDAY="$(date -d yesterday +%Y-%m-%d)"

echo "== Kerää ja prosessoi eilisen uutiset ($YESTERDAY) =="
cd "$SCRIPT_DIR"
"$VENV_PYTHON" -m agent.pipeline \
  --topic ai \
  --since "$YESTERDAY" \
  --until "$YESTERDAY" \
  --verbose

echo "== Buildataan ja deployataan =="
"$SCRIPT_DIR/site/deploy.sh"
