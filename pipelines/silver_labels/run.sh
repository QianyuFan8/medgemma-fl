#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")" && pwd)"
if [ "${CAPSTONE_USE_EXISTING_ENV:-0}" = "1" ]; then
  if [ "${1:-}" = "--self-test" ]; then
    exec "${CAPSTONE_PYTHON:-python3}" -m unittest discover -s "$task_root/tests" -v
  fi
  exec "${CAPSTONE_PYTHON:-python3}" "$task_root/run.py" "$@"
fi
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo 'Usage: bash run.sh --data-dir /path/to/COMET_RMS_Data [--selector both|limma|variance]'
  echo 'First run creates a private Python + R + LIMMA environment (internet required).'
  echo 'Modes: --setup-only; --self-test; --preflight-only; --include-test (after decisions are locked).'
  exit 0
fi
mkdir -p "$task_root/.tools"
manager="$task_root/.tools/bin/micromamba"
if [ ! -x "$manager" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) task_platform=osx-arm64 ;;
    Darwin-x86_64) task_platform=osx-64 ;;
    Linux-x86_64) task_platform=linux-64 ;;
    Linux-aarch64|Linux-arm64) task_platform=linux-aarch64 ;;
    *) echo 'Use macOS, Linux, or Windows WSL2 (Ubuntu).' >&2; exit 2 ;;
  esac
  echo 'Downloading the standalone environment manager from the official endpoint...'
  curl --fail --location --retry 3 "https://micro.mamba.pm/api/micromamba/$task_platform/latest" \
    --output "$task_root/.tools/micromamba.tar.bz2"
  tar -xjf "$task_root/.tools/micromamba.tar.bz2" -C "$task_root/.tools" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$task_root/.mamba"
export XDG_CACHE_HOME="$task_root/.tools/cache"
if command -v sha256sum >/dev/null 2>&1; then
  task_spec_hash="$(sha256sum "$task_root/environment.yml" | awk '{print $1}')"
else
  task_spec_hash="$(shasum -a 256 "$task_root/environment.yml" | awk '{print $1}')"
fi
if [ ! -f "$task_root/.env/.capstone-ready" ] || [ "$(cat "$task_root/.env/.capstone-ready")" != "$task_spec_hash" ]; then
  task_env_action=create
  if [ -d "$task_root/.env/conda-meta" ]; then task_env_action=install; fi
  "$manager" --no-rc "$task_env_action" --yes --prefix "$task_root/.env" --file "$task_root/environment.yml" --strict-channel-priority
  "$manager" run --prefix "$task_root/.env" python -c 'import numpy,pandas,sklearn,joblib,openpyxl'
  "$manager" run --prefix "$task_root/.env" Rscript --vanilla -e 'stopifnot(requireNamespace("limma",quietly=TRUE),requireNamespace("matrixStats",quietly=TRUE))'
  "$manager" list --prefix "$task_root/.env" --explicit > "$task_root/.env/environment-explicit.txt"
  printf '%s\n' "$task_spec_hash" > "$task_root/.env/.capstone-ready"
fi
if [ "${1:-}" = "--setup-only" ]; then
  echo "Environment ready: $task_root/.env"
  exit 0
fi
if [ "${1:-}" = "--self-test" ]; then
  exec "$manager" run --prefix "$task_root/.env" python -m unittest discover -s "$task_root/tests" -v
fi
exec "$manager" run --prefix "$task_root/.env" python "$task_root/run.py" "$@"
