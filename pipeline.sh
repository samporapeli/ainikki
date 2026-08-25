#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m agent.pipeline "$@"
mkdir -p site/public/data/output site/public/data/pipeline
rsync -a data/output/ site/public/data/output/
rsync -a data/pipeline/ site/public/data/pipeline/
