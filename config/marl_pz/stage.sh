#!/usr/bin/env bash
# Stage one Paper 9 multi-seed config as config/default.yaml (with a backup).
# Usage:  ./config/marl_pz/stage.sh config/marl_pz/tier1/s0__seed10.yaml
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <path-to-config-under-config/marl_pz/>" >&2
    exit 1
fi

SRC="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DST="$REPO_ROOT/config/default.yaml"

if [ ! -f "$SRC" ]; then
    echo "not found: $SRC" >&2
    exit 1
fi

if [ -f "$DST" ] && [ ! -f "$DST.bak" ]; then
    cp "$DST" "$DST.bak"
    echo "backed up existing config/default.yaml -> config/default.yaml.bak"
fi

cp "$SRC" "$DST"
echo "staged: $(basename "$SRC")  ->  config/default.yaml"
grep -E "^random_seed:|  strategy:|  strategies:" "$DST" | sed 's/^/    /'
echo "now run:  python3 main.py"
echo "restore your interactive config later with:  cp config/default.yaml.bak config/default.yaml"
