# subtract.py

Adaptive audio subtraction for extracting a wanted source from a mix when a
separate reference track is available.

> This repository is orchestration and signal-processing glue around substantial
> open-source and research work by the audio-separation community. Please see
> [Credits and acknowledgments](CREDITS.md) for the people and projects that make
> it useful.

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

## Credits

The heavy lifting comes from [MelBand-RoFormer-Infer](https://github.com/openmirlab/melband-roformer-infer),
[Kimberley Jensen's Mel-Band RoFormer vocal model](https://huggingface.co/KimberleyJSN/melbandroformer),
[BS-RoFormer](https://github.com/lucidrains/BS-RoFormer),
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
[Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui),
[KUIELAB-MDX-Net](https://github.com/kuielab/mdx-net), and
[Apple MLX](https://github.com/ml-explore/mlx), along with FFmpeg, NumPy, SciPy,
SoundFile, ONNX Runtime, and PyTorch.

See [CREDITS.md](CREDITS.md) for individual attribution, the original research
citation, and licensing notes.

## WebGPU stem studio

The repository also contains a private, browser-only version of `split.py`.
It runs the neural network, FFT, and overlap-add pipeline with WebGPU; audio is
decoded, separated, auditioned, and exported on the user's device. No audio is
uploaded to a server.

The production site is deployed from `main` with GitHub Pages. For local
development:

```bash
npm install
npm run dev
```

Use a current Chrome or Edge build with WebGPU enabled. The first separation
downloads the FP16 model from the repository's `webgpu-model-v1` release and
caches it in the browser. The page writes 32-bit float WAV files named
`TRACK_vocals.wav` and `TRACK_instrumental.wav`.

The WebGPU model is an export of the exact Kimberley Jensen checkpoint used by
the default native workflow. It keeps the native 352,800-sample chunk size,
edge reflection, linear crossfades, and residual subtraction. The browser uses
an optimized two-second overlap to reduce GPU work.
See [WEBGPU.md](WEBGPU.md) for architecture, hashes, and reproducible export
instructions.
