#!/usr/bin/env bash
set -euo pipefail

# Build the site and deploy with rsync.
# Usage: DEPLOY_TARGET=user@host:/var/www/ainikki ./deploy.sh

DEPLOY_TARGET="${DEPLOY_TARGET:-}"

if [ -z "$DEPLOY_TARGET" ]; then
  echo "Usage: DEPLOY_TARGET=user@host:/var/www/ainikki $0"
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd .. && pwd)"

public_topics() {
  "$PROJECT_DIR/venv/bin/python" - "$PROJECT_DIR/config/topics" <<'PY'
from pathlib import Path
import sys

import yaml

topics_dir = Path(sys.argv[1])
for path in sorted(topics_dir.glob("*.yaml")):
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("public"), bool):
        raise SystemExit(f"Topic configuration must define boolean public: {path}")
    if config["public"]:
        print(f"{path.stem}:{config.get('period', 'daily')}")
PY
}

echo "Installing dependencies..."
npm install

echo "Building..."
rm -f public/data/output/*.json public/data/pipeline/*.json
AINIKKI_PUBLIC_ONLY=1 npm run build

echo "Copying data files..."
mkdir -p dist/data/output dist/data/pipeline
rm -f dist/data/output/*.json dist/data/pipeline/*.json
mkdir -p public/data/output public/data/pipeline
while IFS=: read -r topic period; do
  cp "$PROJECT_DIR/data/output/${topic}_${period}_"*.json dist/data/output/ 2>/dev/null || true
  cp "$PROJECT_DIR/data/pipeline/${topic}_${period}_"*.json dist/data/pipeline/ 2>/dev/null || true
  cp "$PROJECT_DIR/data/output/${topic}_${period}_"*.json public/data/output/ 2>/dev/null || true
  cp "$PROJECT_DIR/data/pipeline/${topic}_${period}_"*.json public/data/pipeline/ 2>/dev/null || true
done < <(public_topics)

echo "Deploying to $DEPLOY_TARGET..."
rsync -avz --delete dist/ "$DEPLOY_TARGET"

echo "Done."
