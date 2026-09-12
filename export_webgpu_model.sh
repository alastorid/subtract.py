#!/bin/sh
set -eu

EXPORT_REVISION="6048cc596415025f4c796b40081b9317a934eede"
PYTHON_BIN="${HOME}/.roformer/bin/python"
MODEL_ROOT="${HOME}/.cache/melband-roformer-infer/melband-roformer-kim-vocals"
CHECKPOINT_PATH="${MODEL_ROOT}/MelBandRoformer.ckpt"
CONFIG_PATH="${MODEL_ROOT}/config_vocals_mel_band_roformer.yaml"
OUTPUT_ROOT="$(pwd)/model-build"
TEMP_ROOT="$(mktemp -d)"
TOOLKIT_ROOT="${TEMP_ROOT}/musetric-toolkit"

cleanup() {
    rm -rf "${TEMP_ROOT}"
}
trap cleanup EXIT HUP INT TERM

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: ${PYTHON_BIN} is missing. Run ./setup_roformer.sh first." >&2
    exit 2
fi
if [ ! -f "${CHECKPOINT_PATH}" ] || [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: the verified Kim model is not cached. Run ./split.py on one file first." >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required to install the offline export tools." >&2
    exit 2
fi

uv pip install --python "${PYTHON_BIN}" \
    "onnx==1.22.0" "onnxconverter-common==1.16.0" \
    "onnxscript>=0.7,<1" "onnxruntime>=1.30,<2"
git clone --quiet https://github.com/musetric/musetric-toolkit.git "${TOOLKIT_ROOT}"
git -C "${TOOLKIT_ROOT}" checkout --quiet "${EXPORT_REVISION}"
mkdir -p "${OUTPUT_ROOT}"

PYTHONPATH="${TOOLKIT_ROOT}" "${PYTHON_BIN}" \
    "${TOOLKIT_ROOT}/scripts/onnx/roformer/build_full_onnx.py" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --config "${CONFIG_PATH}" \
    --output "${OUTPUT_ROOT}/kim_vocals_core_t801.onnx" \
    --core-only --fuse-rmsnorm --attn-block 64 --all-fp16 \
    --frames 801 --skip-gate --split-rows 4

PYTHONPATH="${TOOLKIT_ROOT}" "${PYTHON_BIN}" \
    "${TOOLKIT_ROOT}/scripts/onnx/roformer/split_concat_webgpu.py" \
    --input "${OUTPUT_ROOT}/kim_vocals_core_t801.onnx" \
    --output "${OUTPUT_ROOT}/kim_vocals_core_t801_webgpu.onnx"

PYTHONPATH="${TOOLKIT_ROOT}" "${PYTHON_BIN}" \
    "${TOOLKIT_ROOT}/scripts/onnx/roformer/fp16_epsilon_audit.py" \
    "${OUTPUT_ROOT}/kim_vocals_core_t801_webgpu.onnx"

shasum -a 256 \
    "${OUTPUT_ROOT}/kim_vocals_core_t801_webgpu.onnx" \
    "${OUTPUT_ROOT}/kim_vocals_core_t801_webgpu.onnx.data"
