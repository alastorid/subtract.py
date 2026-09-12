#!/bin/sh
set -eu

ENV_DIR="${HOME}/.roformer"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: native MLX setup requires macOS on Apple Silicon (arm64)." >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required. Install an arm64 Python 3.10 or newer." >&2
    exit 2
fi

if [ "$(python3 -c 'import platform; print(platform.machine())')" != "arm64" ]; then
    echo "ERROR: python3 is not an arm64 interpreter. Install a native Apple Silicon Python." >&2
    exit 2
fi

python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
# PyPI 0.1.5 predates the native MLX backend. Pin the inspected official
# revision so setup stays reproducible and cannot silently install Torch-only code.
"$ENV_DIR/bin/python" -m pip install --upgrade \
    "melband-roformer-infer[mlx] @ git+https://github.com/openmirlab/melband-roformer-infer.git@325b02d3a98756232d09dafd4c78d9d21ec7666d"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "NOTE: install ffmpeg for FLAC, MP3, M4A, MP4, and MOV input: brew install ffmpeg" >&2
fi

echo "Ready. split.py will automatically use $ENV_DIR."
