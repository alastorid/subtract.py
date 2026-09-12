# subtract.py

Adaptive audio subtraction for extracting a wanted source from a mix when a
separate reference track is available.

The script aligns the two recordings, matches their intro levels, compensates
for timing drift and spectral differences, subtracts the reference, and uses an
ONNX source-separation model to clean the residual.

## Requirements

- Python 3.12 or newer
- [FFmpeg](https://ffmpeg.org/) available on `PATH`

On macOS with Homebrew:

```bash
brew install ffmpeg
python3 -m venv ~/.subenv
~/.subenv/bin/pip install -r requirements.txt
```

`sub.py` automatically uses `~/.subenv/bin/python` when that environment
exists, so manual activation is optional.

## Usage

```bash
./sub.py MIX.wav REFERENCE.wav OUTPUT.wav
```

For vocal extraction, `MIX.wav` is the vocal-and-instrumental recording and
`REFERENCE.wav` is the instrumental-only recording.

The inputs are aligned before their levels are compared. By default, the first
aligned second is assumed to contain the same instrumental material in both
files; the quieter input is amplified to match the louder input's RMS level.

## Options through environment variables

Change the duration used for intro level matching:

```bash
SUB_INTRO_SECONDS=2 ./sub.py MIX.wav REFERENCE.wav OUTPUT.wav
```

Skip AI cleanup while retaining alignment, adaptive subtraction, and final
reference-guided cleanup:

```bash
SUB_SKIP_AI=1 ./sub.py MIX.wav REFERENCE.wav OUTPUT.wav
```

The AI model is downloaded automatically on first use and cached by
`audio-separator`.

## Standalone vocal separation (`split.py`)

`split.py` is a separate, single-input workflow for making high-quality vocals
and instrumental stems with Mel-Band RoFormer. On Apple Silicon it uses native
MLX by default, falls back to Torch MPS only when MLX is unavailable, and never
silently runs on CPU.

One-time setup:

```bash
./setup_roformer.sh
```

The setup creates `~/.roformer`; `split.py` automatically uses it without shell
activation. The first separation downloads and verifies the recommended Kim
vocals checkpoint (about 913 MB), which is then cached.

```bash
./split.py input.wav
```

This writes exactly `input_vocals.wav` and `input_instrumental.wav` beside the
input. WAV, FLAC, MP3, M4A, MP4, and MOV input are supported. Useful options:

```bash
./split.py input.m4a --output-dir stems --backend mlx --format flac
./split.py input.wav --backend mps
./split.py input.wav --backend cpu  # explicit opt-in only
```

Run `./split.py --help` for all options.
