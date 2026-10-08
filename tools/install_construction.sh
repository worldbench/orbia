#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NAME="orbia-construction"
SOURCES="$ROOT/third_party"
DRY_RUN=0
while (($#)); do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --source-dir) SOURCES=$(realpath -m -- "$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) printf '%s\n' 'Usage: bash tools/install_construction.sh [--name NAME] [--source-dir DIR] [--dry-run]'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
run() {
  if ((DRY_RUN)); then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
clone() {
  if [[ ! -d "$SOURCES/$2/.git" ]]; then run git clone --recursive "$1" "$SOURCES/$2"; fi
}
if ((!DRY_RUN)); then
  for tool in conda git; do
    command -v "$tool" >/dev/null || { printf 'Install %s before running this installer.\n' "$tool" >&2; exit 1; }
  done
fi
run mkdir -p "$SOURCES"
if ((DRY_RUN)) || ! conda run --no-capture-output -n "$NAME" python --version >/dev/null 2>&1; then
  run conda create -y -n "$NAME" python=3.12
fi
# Conda provides compatible native packages and Python 3.12 wheel metadata.
run conda install -y -n "$NAME" --override-channels -c conda-forge python=3.12 open3d=0.19 decord=0.6 'numpy>=1.26,<2'
py() { run conda run --no-capture-output -n "$NAME" python "$@"; }
CONSTRAINTS="$ROOT/requirements/construction-constraints.txt"
py -m pip install -U pip 'setuptools<81' wheel
py -m pip install torch==2.10.0 torchvision==0.25.0 xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu128
py -m pip install -r "$ROOT/requirements/construction.txt" -c "$CONSTRAINTS"
clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
clone https://github.com/facebookresearch/sam3.git sam3
run git -C "$SOURCES/Depth-Anything-3" submodule update --init --recursive
py -m pip install -e "$SOURCES/Depth-Anything-3" -e "$SOURCES/sam3" -c "$CONSTRAINTS"
py -m pip check
printf 'Activate construction: conda activate %s\n' "$NAME"
