# WebGPU architecture and model export

## Runtime

The browser pipeline keeps both the neural network and spectral processing on
the GPU:

1. Browser audio decoding and 44.1 kHz stereo resampling.
2. WebGPU Hann-window STFT and packed complex layout.
3. ONNX Runtime Web's WebGPU execution provider runs the Mel-Band RoFormer
   neural core.
4. WebGPU mask application, inverse FFT, and overlap-add.
5. The same 352,800-sample chunks, reflected borders, and linear fades as the
   native `split.py` backend, with a two-second browser overlap to reduce GPU work.
6. Instrumental is computed as `original - vocals`, without normalization.

The graph is split from its external weights. Both files are published in the
`webgpu-model-v1` GitHub release; the Pages workflow verifies and stages them
under the site's own origin, where the application's service worker caches them.
This avoids cross-origin restrictions without committing large binaries to Git.

| File | SHA-256 |
| --- | --- |
| `kim_vocals_core_t801_webgpu.onnx` | `6fde2f095b72ce1af3b4b299465833a72842b14752e42ae0f29a5ecc4f3d9430` |
| `kim_vocals_core_t801_webgpu.onnx.data` | `d32e854683bd2fd9db7ccd25dbce74c7dbb3360cdf8e4870bdf062e8499cc53c` |

The graph accepts `stft_repr` with shape `[1, 2050, 801, 2]` and returns
complex `masks` with the same shape. FP16 weights reduce the model from about
913 MB to 437 MB. Graph input/output stay FP32.

## Reproduce the model

Run `split.py` once so the verified checkpoint and configuration are present,
then run:

```bash
./export_webgpu_model.sh
```

The script pins Musetric's exporter revision, creates a temporary checkout,
applies query blocking and row splitting for browser GPU limits, fuses safe
FP16 RMS normalization, rewrites wide split/concatenation nodes, and runs the
epsilon audit. Outputs are written to the ignored `model-build/` directory.

The conversion intentionally exports 801 STFT frames because that corresponds
to the native model's configured 352,800-sample window at a 441-sample hop.

## Compatibility

WebGPU and at least nine storage buffers per shader stage are required. The app
does not silently replace this model with a CPU or WASM separator. WASM remains
packaged by ONNX Runtime for runtime support, while the requested inference path
is explicitly WebGPU.
