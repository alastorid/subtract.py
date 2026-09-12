#!/usr/bin/env python3
"""High-quality two-stem vocal separation for Apple Silicon.

Uses melband-roformer-infer's native MLX backend when available, with an
explicit Torch MPS fallback. CPU inference is never selected unless requested.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROFORMER_ENV = Path.home() / ".roformer"
SUPPORTED_INPUTS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mov"}
VERIFIED_DEFAULT_MODEL = "melband-roformer-kim-vocals"


class SplitError(RuntimeError):
    """An actionable command-line error."""


def _env_has_dependencies(python: Path) -> bool:
    """Check a candidate interpreter before replacing this process with it."""
    probe = (
        "import importlib.util as u; "
        "raise SystemExit(0 if all(u.find_spec(x) for x in "
        "('mel_band_roformer', 'soundfile', 'torch')) else 1)"
    )
    try:
        return subprocess.run(
            [str(python), "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _maybe_reexec() -> None:
    """Prefer ~/.roformer only when it is a usable environment."""
    if os.environ.get("SPLIT_ROFORMER_REEXEC") == "1":
        return
    candidate = ROFORMER_ENV / "bin" / "python"
    try:
        already_there = Path(sys.prefix).resolve() == ROFORMER_ENV.resolve()
    except OSError:
        already_there = False
    if not already_there and candidate.is_file() and _env_has_dependencies(candidate):
        env = os.environ.copy()
        env["SPLIT_ROFORMER_REEXEC"] = "1"
        os.execve(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_maybe_reexec()


def _load_api():
    try:
        from mel_band_roformer import MelBandRoformerSession
        from mel_band_roformer.backends import get_backend
        from mel_band_roformer.model_registry import DEFAULT_MODEL, MODEL_REGISTRY
    except ImportError as exc:
        setup = Path(__file__).with_name("setup_roformer.sh")
        raise SplitError(
            "Mel-Band RoFormer is not installed. Run "
            f"{setup} (or: pip install 'melband-roformer-infer[mlx]'). "
            f"Import error: {exc}"
        ) from exc
    return MelBandRoformerSession, get_backend, DEFAULT_MODEL, MODEL_REGISTRY


def _build_parser(default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split one audio/video file into vocals and instrumental stems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input WAV, FLAC, MP3, M4A, MP4, or MOV")
    parser.add_argument("--output-dir", type=Path, help="output directory (default: beside input)")
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "mps", "cpu"),
        default="auto",
        help="compute backend; auto tries MLX, then MPS, and never CPU",
    )
    parser.add_argument("--cpu", action="store_true", help="convenience alias for --backend cpu")
    parser.add_argument("--model", default=default_model, help="registry model slug, name, or checkpoint")
    parser.add_argument("--format", choices=("wav", "flac"), default="wav", help="stem format")
    parser.add_argument("--keep-temp", action="store_true", help="retain staged WAV and raw model output")
    return parser


def _resolve_backend(requested: str, get_backend) -> tuple[str, str, str]:
    if platform.machine().lower() != "arm64" and requested != "cpu":
        raise SplitError(
            f"{requested} requires native Apple Silicon Python (arm64); "
            f"this interpreter reports {platform.machine()!r}. Use an arm64 Python or --backend cpu."
        )

    mlx_reason = None
    if requested in {"auto", "mlx"}:
        try:
            mlx_backend = get_backend("mlx")
            if mlx_backend.is_available():
                return "mlx", "mlx", "mps"
            mlx_reason = "the mlx and/or mlx-spectro package is unavailable"
        except Exception as exc:  # backend discovery should become a useful CLI error
            mlx_reason = str(exc)
        if requested == "mlx":
            raise SplitError(
                "MLX was requested but is unavailable: "
                f"{mlx_reason}. Run setup_roformer.sh to install the MLX extra."
            )

    if requested in {"auto", "mps"}:
        try:
            import torch
        except ImportError as exc:
            mps_reason = f"PyTorch is unavailable ({exc})"
        else:
            built = bool(torch.backends.mps.is_built())
            available = bool(torch.backends.mps.is_available())
            if built and available:
                if requested == "auto":
                    print(f"MLX unavailable ({mlx_reason}); falling back to Torch MPS.", flush=True)
                return "mps", "torch", "mps"
            mps_reason = f"torch.backends.mps: built={built}, available={available}"
        if requested == "mps":
            raise SplitError(f"Torch MPS was requested but is unavailable: {mps_reason}.")
        raise SplitError(
            "No accelerated backend is available. "
            f"MLX: {mlx_reason}; MPS: {mps_reason}. Run setup_roformer.sh. "
            "CPU is intentionally not automatic; request it explicitly with --backend cpu."
        )

    return "cpu", "torch", "cpu"


def _run_ffmpeg(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SplitError(
            f"ffmpeg is required to decode {source.suffix.lower()} input. "
            "Install it with: brew install ffmpeg"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_f32le",
        str(destination),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or f"ffmpeg exited with status {result.returncode}"
        raise SplitError(f"Could not decode the input audio: {detail}")


def _stage_input(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".wav":
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    else:
        print(f"Decoding {source.suffix.lower()} to lossless float PCM…", flush=True)
        _run_ffmpeg(source, destination)


def _audio_info(path: Path):
    try:
        import soundfile as sf

        return sf.info(path)
    except Exception as exc:
        raise SplitError(f"Could not read audio metadata from {path}: {exc}") from exc


def _validate_stem(path: Path, reference) -> None:
    info = _audio_info(path)
    if info.samplerate != reference.samplerate:
        raise SplitError(f"Model output changed sample rate: {info.samplerate} != {reference.samplerate}")
    if info.channels != reference.channels:
        raise SplitError(f"Model output changed channel count: {info.channels} != {reference.channels}")
    if info.frames != reference.frames:
        raise SplitError(f"Model output changed duration: {info.frames} != {reference.frames} frames")


def _select_outputs(manifest: list[dict]) -> tuple[Path, Path]:
    outputs = {str(entry["output_id"]).lower(): Path(entry["output_path"]) for entry in manifest}
    vocals = outputs.get("vocals")
    instrumental = outputs.get("instrumental") or outputs.get("other")
    if not vocals or not instrumental:
        found = ", ".join(sorted(outputs)) or "none"
        raise SplitError(
            "The selected model did not produce a vocals/instrumental pair "
            f"(found: {found}). Choose a dedicated vocals model."
        )
    return vocals, instrumental


def _install_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, staged)
        os.replace(staged, destination)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace, api) -> tuple[Path, Path]:
    MelBandRoformerSession, get_backend, _default_model, registry = api
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SplitError(f"Input file does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_INPUTS:
        allowed = ", ".join(sorted(SUPPORTED_INPUTS))
        raise SplitError(f"Unsupported input type {source.suffix!r}; expected one of: {allowed}")
    if args.cpu:
        if args.backend not in {"auto", "cpu"}:
            raise SplitError("--cpu conflicts with a non-CPU --backend selection")
        args.backend = "cpu"

    try:
        model = registry.get(args.model)
    except KeyError as exc:
        raise SplitError(f"Unknown model {args.model!r}. {exc}") from exc
    if model.category not in {"vocals", "general", "karaoke", "instvoc", "instrumental"}:
        raise SplitError(
            f"Model {model.name!r} is categorized as {model.category!r}, not vocal separation."
        )

    display_backend, package_backend, device = _resolve_backend(args.backend, get_backend)
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else source.parent)
    suffix = ".wav" if args.format == "wav" else ".flac"
    final_vocals = output_dir / f"{source.stem}_vocals{suffix}"
    final_instrumental = output_dir / f"{source.stem}_instrumental{suffix}"

    temp_root = Path(tempfile.mkdtemp(prefix="split-roformer-"))
    started = time.perf_counter()
    try:
        input_dir = temp_root / "input"
        raw_dir = temp_root / "output"
        input_dir.mkdir()
        raw_dir.mkdir()
        staged_input = input_dir / "track.wav"
        _stage_input(source, staged_input)
        reference = _audio_info(staged_input)
        if reference.channels > 2:
            raise SplitError(
                f"Input has {reference.channels} channels; this model supports mono or stereo audio."
            )

        duration = reference.frames / reference.samplerate
        print(f"Input:   {source}", flush=True)
        print(
            f"Audio:   {reference.samplerate} Hz, {reference.channels} channel(s), {duration:.2f} s",
            flush=True,
        )
        print(f"Model:   {model.name} ({model.slug})", flush=True)
        print(f"Backend: {display_backend} ({device})", flush=True)
        print("Loading model (the verified checkpoint downloads once if absent)…", flush=True)

        output_format = "wav_float32" if args.format == "wav" else "flac16"
        with MelBandRoformerSession(
            model_name=model.slug,
            backend=package_backend,
            device=device,
            progress=True,
        ) as session:
            info = session.cache_info()
            print(f"Checkpoint: {info['checkpoint_path']}", flush=True)
            manifest = session.infer(
                input_dir,
                store_dir=raw_dir,
                verbose=True,
                output_format=output_format,
            )

        raw_vocals, raw_instrumental = _select_outputs(manifest)
        _validate_stem(raw_vocals, reference)
        _validate_stem(raw_instrumental, reference)
        _install_output(raw_vocals, final_vocals)
        _install_output(raw_instrumental, final_instrumental)
        elapsed = time.perf_counter() - started
        realtime = elapsed / duration if duration else 0.0
        print(f"Finished in {elapsed:.2f} s ({realtime:.2f}× realtime).", flush=True)
        print(f"Vocals:      {final_vocals}", flush=True)
        print(f"Instrumental:{final_instrumental}", flush=True)
        return final_vocals, final_instrumental
    finally:
        if args.keep_temp:
            print(f"Temporary files kept at: {temp_root}", flush=True)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    try:
        parser = _build_parser(VERIFIED_DEFAULT_MODEL)
        args = parser.parse_args()
        api = _load_api()
        if args.model == VERIFIED_DEFAULT_MODEL and api[2] != VERIFIED_DEFAULT_MODEL:
            raise SplitError(
                "The installed package changed its recommended default model "
                f"from {VERIFIED_DEFAULT_MODEL!r} to {api[2]!r}. Please choose --model explicitly."
            )
        run(args, api)
        return 0
    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: separation failed: {exc}", file=sys.stderr)
        print("Re-run with --keep-temp to retain intermediate files for diagnosis.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
