# Credits and acknowledgments

`subtract.py` supplies workflow glue: transfer and environment automation,
recording alignment, level matching, drift correction, adaptive subtraction,
output validation, and a convenient two-stem command. The source-separation
models, architectures, inference engines, and foundational DSP/ML libraries are
the work of the people and projects below.

## Vocal-separation models and inference

- [Kimberley Jensen (KimberleyJSN)](https://github.com/KimberleyJensen) trained
  the recommended [Mel-Band RoFormer vocal-separation checkpoint](https://huggingface.co/KimberleyJSN/melbandroformer)
  used by `split.py`.
- [OpenMIRLab's MelBand-RoFormer-Infer](https://github.com/openmirlab/melband-roformer-infer)
  provides the packaged inference API, model registry, verified checkpoint
  downloading, output manifests, and the native MLX backend used by `split.py`.
- [Phil Wang (lucidrains)](https://github.com/lucidrains) created the
  [BS-RoFormer PyTorch implementation](https://github.com/lucidrains/BS-RoFormer)
  on which the Mel-Band/Band-Split RoPE Transformer implementation is based.
- [ssmall256's mlx-audio-separator](https://github.com/ssmall256/mlx-audio-separator)
  is credited upstream as the source of the vendored MLX RoFormer implementation.
- [Andrew Beveridge (nomadkaraoke)](https://github.com/nomadkaraoke) maintains
  [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator),
  the programmatic inference wrapper and model registry used by `sub.py`.
- [Anjok07](https://github.com/Anjok07) created
  [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui)
  (UVR), the primary upstream source of python-audio-separator's inference code
  and pretrained model ecosystem. UVR also credits
  [DilanBoskan](https://github.com/DilanBoskan) for essential early contributions.
- [KUIELAB and Woosung Choi](https://github.com/kuielab) developed the original
  [MDX-Net code](https://github.com/kuielab/mdx-net). `sub.py` uses UVR's
  `UVR-MDX-NET-Inst_HQ_5.onnx` model through python-audio-separator.
- UVR/python-audio-separator also credit
  [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) for chunked MDX-Net
  inference and [zhzhongshi](https://github.com/zhzhongshi) for MDXC support.

## Original research

Mel-Band RoFormer and Band-Split RoPE Transformer originate from:

> Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung,
> “Music Source Separation with Band-Split RoPE Transformer,” ICASSP 2024,
> pp. 481–485. [DOI: 10.1109/ICASSP48485.2024.10446843](https://doi.org/10.1109/ICASSP48485.2024.10446843),
> [arXiv:2309.02612](https://arxiv.org/abs/2309.02612).

## Runtime and scientific foundations

This project also depends on the work of the maintainers and contributors to:

- [Apple MLX](https://github.com/ml-explore/mlx) and
  [mlx-spectro](https://github.com/ssmall256/mlx-spectro) for native Apple
  Silicon tensor and spectral computation
- [PyTorch](https://github.com/pytorch/pytorch) and
  [ONNX Runtime](https://github.com/microsoft/onnxruntime) for model inference
- [FFmpeg](https://ffmpeg.org/), [NumPy](https://numpy.org/),
  [SciPy](https://scipy.org/), and [python-soundfile](https://github.com/bastibe/python-soundfile)
  for media decoding and numerical/audio processing

## Licensing note

The linked software projects retain their own copyrights and licenses; many are
MIT-licensed. Model checkpoints may have separate terms set by their trainers or
distributors. This repository does not bundle upstream source trees or model
weights: dependencies are installed from their projects, and checkpoints are
downloaded into the user's cache. Consult each linked project or model card for
the terms that apply to redistribution and use.
