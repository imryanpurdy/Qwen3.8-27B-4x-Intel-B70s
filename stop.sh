#!/usr/bin/env bash
set -euo pipefail
CONTAINER="${CONTAINER:-qwen28tp4m}"
docker rm -f "$CONTAINER"
echo "stopped $CONTAINER"
