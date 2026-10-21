#!/usr/bin/env bash
set -e

# Unitree G1 Humanoid Unified Runner
# Determines python environment (.venv, python3)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python 3 not found on PATH." >&2
    exit 1
fi

MODE="${1:-teleop}"
shift || true

# Native Wayland GLFW currently fails during libdecor initialization on some
# Linux desktops.  XWayland is the stable fallback and remains overridable:
# `PYGLFW_LIBRARY_VARIANT=wayland ./run.sh dance` opts back into Wayland.
if [[ "$(uname -s)" == "Linux" && -n "${DISPLAY:-}" && -z "${PYGLFW_LIBRARY_VARIANT:-}" ]]; then
    export PYGLFW_LIBRARY_VARIANT=x11
    echo " Viewer backend: GLFW X11/XWayland (override with PYGLFW_LIBRARY_VARIANT)"
fi

echo "============================================================"
echo " Starting Unitree G1 Framework (Mode: $MODE)"
echo " Python: $PYTHON_BIN"
echo "============================================================"

exec "$PYTHON_BIN" main.py "$MODE" "$@"
