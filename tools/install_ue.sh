#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NAME="orbia-ue"
DRY_RUN=0
while (($#)); do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) printf '%s\n' 'Usage: bash tools/install_ue.sh [--name NAME] [--dry-run]'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
run() {
  if ((DRY_RUN)); then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ((!DRY_RUN)); then
  command -v conda >/dev/null || { printf 'Install conda before running this installer.\n' >&2; exit 1; }
fi
if ((DRY_RUN)) || ! conda run --no-capture-output -n "$NAME" python --version >/dev/null 2>&1; then
  run conda create -y -n "$NAME" python=3.10
fi
py() { run conda run --no-capture-output -n "$NAME" python "$@"; }
py -m pip install -U pip 'setuptools<81' wheel
py -m pip install -r "$ROOT/requirements/ue.txt"
py -m pip check
printf 'Activate UE conversion: conda activate %s\nRendering runs separately in Unreal Editor.\n' "$NAME"
