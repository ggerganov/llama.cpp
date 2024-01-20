#!/bin/bash
set -euo pipefail
this=$(realpath "$0"); readonly this
cd "$(dirname "$this")"
shellcheck "$this"

if (( $# != 1 && $# != 2  )); then
    cat >&2 <<'EOF'
usage:
    ci-run.sh <tmp_dir> [<cache_dir>]

This script wraps ci/run.sh:
* If <tmp_dir> is a ramdisk, you can reduce writes to your SSD.
    (~30GB per run with openllama_3b_v2)
* Persistent model and data files are synced to and from <cache_dir>,
    excluding generated .gguf files.
* <cache_dir> defaults to  ~/.cache/llama.cpp
EOF
    exit 1
fi

cd .. # => llama.cpp repo root

tmp="$1"
mkdir -p "$tmp"
tmp=$(realpath "$tmp")
echo >&2 "Using tmp=$tmp"

cache="${2-$HOME/.cache/llama.cpp}"
mkdir -p "$cache"
cache=$(realpath "$cache")
echo >&2 "Using cache=$cache"

_sync() {
    local from="$1"; shift
    local to="$1"; shift

    echo >&2 "Syncing from $from to $to"
    mkdir -p "$from" "$to"
    rsync -a "$from" "$to" --delete-during "$@"
}

_sync "$(realpath .)/" "$tmp/llama.cpp"
_sync "$cache/ci-mnt/models/" "$tmp/llama.cpp/ci-mnt/models/"

cd "$tmp/llama.cpp"
bash ci/run.sh ci-out ci-mnt

_sync 'ci-mnt/models/' "$cache/ci-mnt/models/" --exclude='*.gguf' -P
