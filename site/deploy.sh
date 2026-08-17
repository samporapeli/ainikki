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

echo "Installing dependencies..."
npm install

echo "Building..."
npm run build

echo "Copying data files..."
mkdir -p dist/data/output dist/data/pipeline
cp ../data/output/*.json dist/data/output/ 2>/dev/null || true
cp ../data/pipeline/*.json dist/data/pipeline/ 2>/dev/null || true
# Also copy to public/ for dev server access
mkdir -p public/data/output public/data/pipeline
cp ../data/output/*.json public/data/output/ 2>/dev/null || true
cp ../data/pipeline/*.json public/data/pipeline/ 2>/dev/null || true

echo "Deploying to $DEPLOY_TARGET..."
rsync -avz --delete dist/ "$DEPLOY_TARGET"

echo "Done."
