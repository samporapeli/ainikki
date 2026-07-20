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

echo "Deploying to $DEPLOY_TARGET..."
rsync -avz --delete dist/ "$DEPLOY_TARGET"

echo "Done."
