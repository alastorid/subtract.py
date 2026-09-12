#!/usr/bin/env python3
"""
Adaptive lossy-source audio subtraction.

Usage:
    python sub.py a.wav b.wav c.wav

Optional environment variables:
    SUB_SKIP_AI=1            Skip AI cleanup but keep adaptive subtraction.
    SUB_INTRO_SECONDS=1.0    Duration used for aligned RMS level matching.

Goal:
    C = A - matched(B)

Designed for cases where B is approximately the component contained
inside A, but B/A may have:
    - different lossy encoding
    - gain differences
    - EQ differences
    - phase differences
    - small timing offset
    - slowly changing clock/timing drift
"""

import os
import sys

# Prefer the dedicated environment automatically when the script is invoked
# directly from a shell that has not activated it yet.
SUBENV_PYTHON = os.path.expanduser("~/.subenv/bin/python")
SUBENV_ROOT = os.path.dirname(os.path.dirname(SUBENV_PYTHON))

if os.path.isfile(SUBENV_PYTHON) and sys.prefix != SUBENV_ROOT:
    os.execv(SUBENV_PYTHON, [SUBENV_PYTHON, __file__, *sys.argv[1:]])

import numpy as np
import soundfile as sf
import tempfile
import shutil

from scipy.signal import stft, istft
from audio_separator.separator import Separator

from scipy.signal import (
    correlate,
    correlation_lags,
    resample_poly,
    fftconvolve,
)
from scipy.linalg import lstsq
from scipy.interpolate import CubicSpline
from math import gcd


# ============================================================
# Configuration
# ============================================================

# Initial alignment search range
MAX_INITIAL_OFFSET_SEC = 30.0

# Alignment block spacing
DRIFT_BLOCK_SEC = 10.0

# Length of audio examined for each local alignment
DRIFT_ANALYSIS_SEC = 4.0

# Local drift search around predicted position
LOCAL_SEARCH_MS = 50.0

# Downsampling used during coarse alignment
COARSE_DS = 8

# Adaptive subtraction block
PROCESS_BLOCK_SEC = 4.0

# 50% overlap
OVERLAP = 0.5

# FIR used to transform B toward the version present in A
FIR_TAPS = 129

# Regularization prevents insane FIR coefficients
RIDGE = 1e-5

# Don't train on nearly silent sections
SILENCE_PERCENTILE = 15

# Output safety
OUTPUT_PEAK = 0.999


# ============================================================
# Utility
# ============================================================

def mono(x):
    if x.shape[1] == 1:
        return x[:, 0]
    return np.mean(x, axis=1)


def normalize_corr(x):
    x = x.astype(np.float64)
    x -= np.mean(x)

    s = np.std(x)

    if s > 1e-12:
        x /= s

    return x


def match_intro_level(a, b, sr, seconds=1.0):
    """Boost the quieter aligned input to match first-second RMS."""
    n = min(len(a), len(b), max(1, int(sr * seconds)))
    rms_a = float(np.sqrt(np.mean(a[:n] ** 2)))
    rms_b = float(np.sqrt(np.mean(b[:n] ** 2)))

    print("\nMatching aligned first-second levels...")
    print(f"  A RMS: {20 * np.log10(max(rms_a, 1e-12)):.2f} dBFS")
    print(f"  B RMS: {20 * np.log10(max(rms_b, 1e-12)):.2f} dBFS")

    if rms_a < 1e-8 or rms_b < 1e-8:
        print("  Intro is too quiet for reliable level matching; skipped.")
        return a, b

    if rms_a < rms_b:
        gain = rms_b / rms_a
        a = a * gain
        target = "A"
    else:
        gain = rms_a / rms_b
        b = b * gain
        target = "B"

    print(f"  Boosted {target} by {20 * np.log10(gain):.2f} dB ({gain:.6f}x).")
    return a, b


def ai_residual_cleanup(
    residual,
    sr,
    work_dir=None,
    ai_strength=2.0,
    silence_strength=0.003,
    vocal_threshold=0.025,
    n_fft=4096,
    hop=512,
):
    """
    residual:
        numpy array, shape (samples, channels)

    Returns:
        cleaned residual

    Pipeline:
        residual
            ↓
        ONNX vocal estimate
            ↓
        AI-derived Wiener mask
            ↓
        mask applied to ORIGINAL residual
            ↓
        aggressive cleanup where AI says no wanted source exists
    """

    if residual.ndim == 1:
        residual = residual[:, None]

    own_tmp = work_dir is None

    if own_tmp:
        work_dir = tempfile.mkdtemp(
            prefix="audio_ai_cleanup_"
        )

    os.makedirs(
        work_dir,
        exist_ok=True
    )

    residual_path = os.path.join(
        work_dir,
        "residual_for_ai.wav"
    )

    vocal_path = os.path.join(
        work_dir,
        "ai_vocal.wav"
    )

    instrumental_path = os.path.join(
        work_dir,
        "ai_instrumental.wav"
    )

    # ---------------------------------------------------------
    # Save residual for AI
    # ---------------------------------------------------------

    sf.write(
        residual_path,
        residual,
        sr,
        subtype="FLOAT"
    )

    print()
    print("================================")
    print("Running AI residual separator...")
    print("================================")

    # Use a compact ONNX model. It avoids the RoFormer/beartype issue on
    # Python 3.14 and downloads much faster on a fresh host.
    separator = Separator(
        output_dir=work_dir,
        output_format="wav",
    )

    # Models download automatically the first time.
    separator.load_model("UVR-MDX-NET-Inst_HQ_5.onnx")

    outputs = separator.separate(
        residual_path,
        custom_output_names={
            "Vocals": "ai_vocal",
            "Instrumental": "ai_instrumental",
        },
    )

    print("AI output:")
    for p in outputs:
        print(" ", p)

    if not os.path.exists(vocal_path):
        # Fallback: identify returned vocal file.
        candidates = [
            p for p in outputs
            if "vocal" in os.path.basename(p).lower()
        ]

        if not candidates:
            raise RuntimeError(
                "AI separator did not produce a vocal stem."
            )

        vocal_path = candidates[0]

    # ---------------------------------------------------------
    # Read AI estimate
    # ---------------------------------------------------------

    vocal, vocal_sr = sf.read(
        vocal_path,
        dtype="float64",
        always_2d=True
    )

    if vocal_sr != sr:
        vocal = resample_audio(
            vocal,
            vocal_sr,
            sr
        )

    # Channel compatibility
    if vocal.shape[1] != residual.shape[1]:

        if vocal.shape[1] == 1:
            vocal = np.repeat(
                vocal,
                residual.shape[1],
                axis=1
            )

        elif residual.shape[1] == 1:
            vocal = np.mean(
                vocal,
                axis=1,
                keepdims=True
            )

        else:
            channels = min(
                vocal.shape[1],
                residual.shape[1]
            )

            vocal = vocal[:, :channels]
            residual = residual[:, :channels]

    length = min(
        len(residual),
        len(vocal)
    )

    residual = residual[:length]
    vocal = vocal[:length]

    result = np.zeros_like(
        residual
    )

    # =========================================================
    # AI → Wiener mask
    # =========================================================

    for ch in range(
        residual.shape[1]
    ):

        print(
            f"AI masking channel {ch + 1}..."
        )

        r = residual[:, ch]
        v = vocal[:, ch]

        noverlap = (
            n_fft -
            hop
        )

        _, _, R = stft(
            r,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )

        _, _, V = stft(
            v,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )

        # Make dimensions match
        freq_bins = min(
            R.shape[0],
            V.shape[0]
        )

        frames = min(
            R.shape[1],
            V.shape[1]
        )

        R = R[
            :freq_bins,
            :frames
        ]

        V = V[
            :freq_bins,
            :frames
        ]

        eps = 1e-12

        mag_r = np.abs(R)
        mag_v = np.abs(V)

        # -----------------------------------------------------
        # Estimate what AI considers unwanted residue.
        #
        # R = wanted + remaining junk
        #
        # AI says V ≈ wanted
        #
        # so:
        #
        # unwanted ≈ R - V
        # -----------------------------------------------------

        N = R - V

        mag_n = np.abs(N)

        # -----------------------------------------------------
        # Wiener-like mask
        #
        # |V|²
        # --------------------
        # |V|² + α|N|²
        #
        # Increasing ai_strength removes more music.
        # -----------------------------------------------------

        wanted_power = (
            mag_v ** 2
        )

        noise_power = (
            mag_n ** 2
        )

        mask = (
            wanted_power /
            (
                wanted_power +
                ai_strength *
                noise_power +
                eps
            )
        )

        # A mild exponent makes confident decisions stronger.
        mask = (
            mask ** 0.75
        )

        # =====================================================
        # Detect frames where AI says:
        #
        # "there is basically NO wanted source here"
        #
        # This directly attacks the symptom you described:
        #
        # quiet ghost music during supposed silence.
        # =====================================================

        residual_frame_energy = np.sqrt(
            np.mean(
                mag_r ** 2,
                axis=0
            )
        )

        vocal_frame_energy = np.sqrt(
            np.mean(
                mag_v ** 2,
                axis=0
            )
        )

        vocal_ratio = (
            vocal_frame_energy /
            (
                residual_frame_energy +
                eps
            )
        )

        # -----------------------------------------------------
        # Soft silence probability
        # -----------------------------------------------------

        wanted_probability = np.clip(
            vocal_ratio /
            vocal_threshold,
            0.0,
            1.0
        )

        # Smooth in time.
        if len(
            wanted_probability
        ) >= 5:

            kernel = np.array([
                1,
                2,
                4,
                2,
                1
            ], dtype=float)

            kernel /= np.sum(
                kernel
            )

            wanted_probability = np.convolve(
                wanted_probability,
                kernel,
                mode="same"
            )

        # -----------------------------------------------------
        # When AI says silence:
        #
        # allow up to ~50 dB suppression.
        #
        # When AI says wanted audio:
        #
        # trust Wiener mask normally.
        # -----------------------------------------------------

        silence_gain = (
            silence_strength +
            (
                1.0 -
                silence_strength
            ) *
            wanted_probability
        )

        mask *= silence_gain[
            None,
            :
        ]

        # =====================================================
        # Avoid musical-noise artifacts
        # =====================================================

        # Frequency smoothing
        if mask.shape[0] > 4:

            mask[2:-2] = (
                mask[:-4]
                +
                2 * mask[1:-3]
                +
                4 * mask[2:-2]
                +
                2 * mask[3:-1]
                +
                mask[4:]
            ) / 10.0

        # Time smoothing
        if mask.shape[1] > 4:

            mask[:, 2:-2] = (
                mask[:, :-4]
                +
                2 * mask[:, 1:-3]
                +
                4 * mask[:, 2:-2]
                +
                2 * mask[:, 3:-1]
                +
                mask[:, 4:]
            ) / 10.0

        # -----------------------------------------------------
        # Preserve tiny amount instead of binary zero.
        #
        # Prevents weird hard spectral holes.
        # -----------------------------------------------------

        mask = np.clip(
            mask,
            0.0005,
            1.0
        )

        # -----------------------------------------------------
        # CRITICAL:
        #
        # Apply AI mask to ORIGINAL residual,
        # NOT to AI generated audio.
        #
        # This preserves your original phase/transients.
        # -----------------------------------------------------

        CLEAN = (
            R *
            mask
        )

        _, cleaned = istft(
            CLEAN,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )

        cleaned = cleaned[
            :length
        ]

        if len(cleaned) < length:

            cleaned = np.pad(
                cleaned,
                (
                    0,
                    length -
                    len(cleaned)
                )
            )

        result[:, ch] = cleaned

        # diagnostics

        before = np.sqrt(
            np.mean(
                r ** 2
            )
        )

        after = np.sqrt(
            np.mean(
                cleaned ** 2
            )
        )

        if (
            before > 1e-12
            and
            after > 1e-12
        ):

            reduction = (
                20 *
                np.log10(
                    before /
                    after
                )
            )

            print(
                f"  AI cleanup reduction: "
                f"{reduction:.2f} dB"
            )

    # ---------------------------------------------------------
    # Cleanup temporary files
    # ---------------------------------------------------------

    if own_tmp:
        shutil.rmtree(
            work_dir,
            ignore_errors=True
        )

    return result

def resample_audio(x, old_sr, new_sr):

    if old_sr == new_sr:
        return x

    g = gcd(old_sr, new_sr)

    up = new_sr // g
    down = old_sr // g

    print(f"Resampling B: {old_sr} -> {new_sr}")

    channels = []

    for ch in range(x.shape[1]):

        channels.append(
            resample_poly(
                x[:, ch],
                up,
                down
            )
        )

    n = min(len(c) for c in channels)

    return np.column_stack([
        c[:n] for c in channels
    ])


# ============================================================
# Initial integer alignment
# ============================================================

def initial_alignment(a, b, sr):

    print("\nFinding initial alignment...")

    a = mono(a)
    b = mono(b)

    ds = COARSE_DS

    # Use first ~90 seconds
    n = min(
        len(a),
        len(b),
        int(sr * 90)
    )

    aa = normalize_corr(a[:n:ds])
    bb = normalize_corr(b[:n:ds])

    corr = correlate(
        aa,
        bb,
        mode="full",
        method="fft"
    )

    lags = correlation_lags(
        len(aa),
        len(bb),
        mode="full"
    )

    maxlag = int(
        MAX_INITIAL_OFFSET_SEC *
        sr / ds
    )

    mask = np.abs(lags) <= maxlag

    c = corr[mask]
    l = lags[mask]

    best = np.argmax(np.abs(c))

    lag = int(l[best] * ds)

    print(
        f"Initial offset: {lag} samples "
        f"({1000 * lag / sr:.3f} ms)"
    )

    return lag


# ============================================================
# Fractional delay estimation
# ============================================================

def parabolic_peak(y, i):
    """
    Sub-sample peak estimate using a parabola through
    correlation samples i-1, i, i+1.
    """

    if i <= 0 or i >= len(y) - 1:
        return float(i)

    y1 = y[i - 1]
    y2 = y[i]
    y3 = y[i + 1]

    denom = y1 - 2 * y2 + y3

    if abs(denom) < 1e-20:
        return float(i)

    delta = 0.5 * (y1 - y3) / denom

    # avoid pathological estimate
    delta = np.clip(delta, -1.0, 1.0)

    return i + delta


def local_offset(a, b, expected, sr):

    half = int(
        DRIFT_ANALYSIS_SEC *
        sr / 2
    )

    search = int(
        LOCAL_SEARCH_MS *
        sr / 1000
    )

    center_a = expected

    a0 = center_a - half
    a1 = center_a + half

    if a0 < 0 or a1 >= len(a):
        return None

    aa = a[a0:a1]

    # B is assumed approximately same timeline here.
    b0 = a0 - search
    b1 = a1 + search

    if b0 < 0 or b1 >= len(b):
        return None

    bb = b[b0:b1]

    aa = normalize_corr(aa)
    bb = normalize_corr(bb)

    corr = correlate(
        bb,
        aa,
        mode="valid",
        method="fft"
    )

    # use magnitude so polarity reversal still aligns
    abs_corr = np.abs(corr)

    idx = np.argmax(abs_corr)

    fractional_idx = parabolic_peak(
        abs_corr,
        idx
    )

    offset = (
        b0 +
        fractional_idx -
        a0
    )

    return offset


# ============================================================
# Drift map
# ============================================================

def build_drift_map(a, b, sr):

    print("\nMeasuring timing drift...")

    am = mono(a)
    bm = mono(b)

    step = int(
        DRIFT_BLOCK_SEC * sr
    )

    points_t = []
    points_offset = []

    start = int(
        DRIFT_ANALYSIS_SEC *
        sr
    )

    end = min(len(am), len(bm)) - start

    for pos in range(start, end, step):

        off = local_offset(
            am,
            bm,
            pos,
            sr
        )

        if off is None:
            continue

        points_t.append(pos)
        points_offset.append(off)

    points_t = np.asarray(
        points_t,
        dtype=np.float64
    )

    points_offset = np.asarray(
        points_offset,
        dtype=np.float64
    )

    if len(points_t) < 2:

        print("Not enough drift points.")
        return None

    # Reject obvious bad correlation hits using median
    med = np.median(points_offset)

    mad = np.median(
        np.abs(points_offset - med)
    ) + 1e-9

    mask = (
        np.abs(points_offset - med)
        < max(10.0, 10 * mad)
    )

    points_t = points_t[mask]
    points_offset = points_offset[mask]

    print(
        f"Valid drift measurements: "
        f"{len(points_t)}"
    )

    if len(points_t) < 2:
        return None

    print(
        f"Timing offset range: "
        f"{np.min(points_offset):.3f} .. "
        f"{np.max(points_offset):.3f} samples"
    )

    drift = CubicSpline(
        points_t,
        points_offset,
        extrapolate=True
    )

    return drift


# ============================================================
# Fractional resampling / warping
# ============================================================

def warp_b_to_a(b, drift):

    if drift is None:
        return b

    print("\nApplying fractional timing correction...")

    n = len(b)

    output = np.zeros_like(b)

    # timeline of A
    t = np.arange(
        n,
        dtype=np.float64
    )

    # where to sample B
    source_t = t + drift(t)

    valid = (
        (source_t >= 0) &
        (source_t <= n - 1)
    )

    for ch in range(b.shape[1]):

        output[:, ch] = np.interp(
            source_t,
            t,
            b[:, ch],
            left=0.0,
            right=0.0
        )

    return output


# ============================================================
# FIR estimation
# ============================================================

def estimate_fir(a, b, taps):

    n = min(len(a), len(b))

    if n <= taps + 100:
        return None

    # Maximum training rows per block.
    # Enough for accurate estimation without huge matrices.
    max_rows = 30000

    positions = np.arange(
        taps,
        n
    )

    if len(positions) > max_rows:

        select = np.linspace(
            0,
            len(positions) - 1,
            max_rows,
            dtype=int
        )

        positions = positions[select]

    X = np.empty(
        (len(positions), taps),
        dtype=np.float64
    )

    for k in range(taps):
        X[:, k] = b[
            positions - k
        ]

    y = a[positions]

    energy = np.max(
        np.abs(X),
        axis=1
    )

    threshold = np.percentile(
        energy,
        SILENCE_PERCENTILE
    )

    good = energy > threshold

    X = X[good]
    y = y[good]

    if len(y) < taps * 2:
        return None

    # Ridge regularization
    XtX = X.T @ X

    scale = (
        np.trace(XtX) /
        taps
    )

    XtX += (
        np.eye(taps) *
        RIDGE *
        max(scale, 1e-12)
    )

    Xty = X.T @ y

    h = np.linalg.solve(
        XtX,
        Xty
    )

    return h


# ============================================================
# Adaptive block cancellation
# ============================================================

def process_channel(a, b, sr, ch):

    print(
        f"\nProcessing channel {ch + 1}..."
    )

    n = min(len(a), len(b))

    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)

    block = int(
        PROCESS_BLOCK_SEC * sr
    )

    hop = int(
        block * (1.0 - OVERLAP)
    )

    if hop <= 0:
        hop = block // 2

    # Hann overlap/add
    window = np.hanning(block)

    output = np.zeros(n)
    weight = np.zeros(n)

    reductions = []

    total_blocks = max(
        1,
        (n - block) // hop + 1
    )

    block_num = 0

    for start in range(
        0,
        max(1, n - block + 1),
        hop
    ):

        end = min(
            start + block,
            n
        )

        if end - start < FIR_TAPS * 4:
            break

        aa = a[start:end].copy()
        bb = b[start:end].copy()

        # DC removal for fitting
        mean_a = np.mean(aa)
        mean_b = np.mean(bb)

        aa0 = aa - mean_a
        bb0 = bb - mean_b

        h = estimate_fir(
            aa0,
            bb0,
            FIR_TAPS
        )

        if h is None:
            residual = aa0

        else:

            matched = fftconvolve(
                bb0,
                h,
                mode="full"
            )[:len(bb0)]

            # One final scalar optimization
            denom = np.dot(
                matched,
                matched
            )

            if denom > 1e-20:

                gain = (
                    np.dot(
                        aa0,
                        matched
                    ) /
                    denom
                )

                # prevent pathological block fits
                gain = np.clip(
                    gain,
                    0.25,
                    4.0
                )

            else:
                gain = 1.0

            matched *= gain

            residual = (
                aa0 -
                matched
            )

        rms_before = np.sqrt(
            np.mean(aa0 ** 2)
        )

        rms_after = np.sqrt(
            np.mean(residual ** 2)
        )

        if (
            rms_before > 1e-12 and
            rms_after > 1e-12
        ):
            db = 20 * np.log10(
                rms_before /
                rms_after
            )
            reductions.append(db)

        w = window[:len(residual)]

        output[start:end] += (
            residual * w
        )

        weight[start:end] += w

        block_num += 1

        if block_num % 10 == 0:
            print(
                f"  {block_num}/{total_blocks}"
            )

    good = weight > 1e-10

    output[good] /= weight[good]

    # Edges not covered by Hann window:
    output[~good] = a[~good]

    if reductions:

        print(
            f"  Median cancellation: "
            f"{np.median(reductions):.2f} dB"
        )

        print(
            f"  Best block: "
            f"{np.max(reductions):.2f} dB"
        )

    return output


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 4:

        print(
            "Usage:\n"
            "  python sub.py "
            "a.wav b.wav c.wav"
        )

        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    file_c = sys.argv[3]

    print("Adaptive lossy-source subtraction")
    print("---------------------------------")
    print(f"A = {file_a}")
    print(f"B = {file_b}")
    print(f"C = {file_c}")

    a, sr_a = sf.read(
        file_a,
        dtype="float64",
        always_2d=True
    )

    b, sr_b = sf.read(
        file_b,
        dtype="float64",
        always_2d=True
    )

    print(
        f"\nA: {sr_a} Hz, "
        f"{a.shape[1]} ch, "
        f"{len(a)/sr_a:.2f}s"
    )

    print(
        f"B: {sr_b} Hz, "
        f"{b.shape[1]} ch, "
        f"{len(b)/sr_b:.2f}s"
    )

    # --------------------------------------------------------
    # Sample rate
    # --------------------------------------------------------

    b = resample_audio(
        b,
        sr_b,
        sr_a
    )

    # --------------------------------------------------------
    # Channels
    # --------------------------------------------------------

    if a.shape[1] != b.shape[1]:

        if b.shape[1] == 1:

            print(
                "Duplicating mono B."
            )

            b = np.repeat(
                b,
                a.shape[1],
                axis=1
            )

        elif a.shape[1] == 1:

            print(
                "Downmixing B to mono."
            )

            b = np.mean(
                b,
                axis=1,
                keepdims=True
            )

        else:

            raise RuntimeError(
                "Incompatible channel counts."
            )

    # --------------------------------------------------------
    # Initial alignment
    # --------------------------------------------------------

    lag = initial_alignment(
        a,
        b,
        sr_a
    )

    if lag > 0:

        a = a[lag:]

    elif lag < 0:

        b = b[-lag:]

    n = min(
        len(a),
        len(b)
    )

    a = a[:n]
    b = b[:n]

    # Match gain only after integer alignment so the first second in both
    # arrays refers to the same instrumental passage.
    intro_seconds = float(
        os.environ.get("SUB_INTRO_SECONDS", "1.0")
    )

    if intro_seconds <= 0:
        raise ValueError("SUB_INTRO_SECONDS must be greater than zero.")

    a, b = match_intro_level(
        a,
        b,
        sr_a,
        seconds=intro_seconds,
    )

    # --------------------------------------------------------
    # Local fractional drift
    # --------------------------------------------------------

    drift = build_drift_map(
        a,
        b,
        sr_a
    )

    b = warp_b_to_a(
        b,
        drift
    )

    # --------------------------------------------------------
    # Adaptive cancellation
    # --------------------------------------------------------

    output = np.zeros_like(a)

    for ch in range(a.shape[1]):

        output[:, ch] = process_channel(
            a[:, ch],
            b[:, ch],
            sr_a,
            ch
        )

    # ============================================================
    # AI cleanup stage
    # ============================================================

    print()
    print("Starting AI cleanup...")

    if os.environ.get("SUB_SKIP_AI") == "1":
        print("AI cleanup skipped (SUB_SKIP_AI=1).")
    else:
        try:
            output = ai_residual_cleanup(
                output,
                sr_a,

                # Larger = more aggressive removal
                ai_strength=2.5,

                # Gain in regions considered silence.
                # 0.003 ≈ -50 dB
                silence_strength=0.003,

                # How much AI vocal energy must exist before
                # we're confident this isn't silence.
                vocal_threshold=0.025,

                n_fft=4096,
                hop=512,
            )
        except Exception as exc:
            print(
                "\nWARNING: AI cleanup unavailable; "
                f"keeping adaptive residual: {exc}"
            )

    print()
    print("Final reference-guided cleanup...")

    for ch in range(output.shape[1]):

        output[:, ch] = spectral_residual_suppress(
            output[:, ch],
            b[:len(output), ch],
            sr_a,
            n_fft=4096,
            hop=1024,
            strength=1.2,
            floor=0.03,
            threshold_db=-42.0
        )

    # --------------------------------------------------------
    # Safety against clipping
    # --------------------------------------------------------

    peak = np.max(
        np.abs(output)
    )

    print(
        f"\nOutput peak: {peak:.6f}"
    )

    if peak > OUTPUT_PEAK:

        scale = (
            OUTPUT_PEAK /
            peak
        )

        output *= scale

        print(
            f"Peak protection scale: "
            f"{scale:.6f}"
        )

    # Floating-point WAV avoids another integer
    # quantization step.
    sf.write(
        file_c,
        output,
        sr_a,
        subtype="FLOAT"
    )

    print(
        f"\nDone: {file_c}"
    )

from scipy.signal import stft, istft


def spectral_residual_suppress(
    residual,
    reference,
    sr,
    n_fft=4096,
    hop=1024,
    strength=1.8,
    floor=0.03,
    threshold_db=-38.0
):
    """
    Reference-guided spectral suppression.

    residual  = result after adaptive subtraction
    reference = time-aligned B

    strength:
        1.0 = gentle
        1.5-2.5 = stronger music removal

    floor:
        minimum retained gain.
        0.03 = max ~30 dB attenuation.

    threshold_db:
        only suppress where reference is sufficiently present.
    """

    noverlap = n_fft - hop

    f, t, R = stft(
        residual,
        fs=sr,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros"
    )

    _, _, B = stft(
        reference,
        fs=sr,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros"
    )

    mag_r = np.abs(R)
    mag_b = np.abs(B)

    eps = 1e-12

    # Normalize reference locally per frame.
    frame_peak = np.max(
        mag_b,
        axis=0,
        keepdims=True
    ) + eps

    b_norm = mag_b / frame_peak

    # Only act where B has meaningful spectral content.
    threshold = 10.0 ** (
        threshold_db / 20.0
    )

    active = np.clip(
        (b_norm - threshold) /
        max(1e-12, 1.0 - threshold),
        0.0,
        1.0
    )

    # Estimate how much of residual still correlates with B's spectrum.
    ratio = mag_b / (
        mag_r + mag_b + eps
    )

    suppression = (
        active *
        ratio *
        strength
    )

    gain = 1.0 - suppression

    gain = np.clip(
        gain,
        floor,
        1.0
    )

    # Smooth across frequency.
    gain[1:-1] = (
        gain[:-2] +
        2.0 * gain[1:-1] +
        gain[2:]
    ) / 4.0

    # Smooth across time.
    gain[:, 1:-1] = (
        gain[:, :-2] +
        2.0 * gain[:, 1:-1] +
        gain[:, 2:]
    ) / 4.0

    cleaned = R * gain

    _, y = istft(
        cleaned,
        fs=sr,
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True
    )

    return y[:len(residual)]


if __name__ == "__main__":
    main()
