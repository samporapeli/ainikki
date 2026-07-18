#!/usr/bin/env bash
set -euo pipefail

# Buildataan sivusto ja deployataan rsync:llä.
# Käyttö: DEPLOY_TARGET=user@host:/var/www/ainikki ./deploy.sh

DEPLOY_TARGET="${DEPLOY_TARGET:-}"

if [ -z "$DEPLOY_TARGET" ]; then
  echo "Käyttö: DEPLOY_TARGET=user@host:/var/www/ainikki $0"
  exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Asennetaan riippuvuudet..."
npm install

echo "Buildataan..."
npm run build

echo "Deployataan $DEPLOY_TARGET..."
rsync -avz --delete dist/ "$DEPLOY_TARGET"

echo "Valmis."
