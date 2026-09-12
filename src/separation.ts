import { createVocalsGpuRuntime } from "./vendor/musetric/ai/runtime/vocals/vocalsRuntime";

export const SAMPLE_RATE = 44_100;
const CHUNK_SAMPLES = 352_800;
const STEP = CHUNK_SAMPLES / 2;
const FADE = CHUNK_SAMPLES / 10;
const BORDER = CHUNK_SAMPLES - STEP;

const MODEL_ROOT = new URL(`${import.meta.env.BASE_URL}models/`, window.location.href).href;
export const MODEL_GRAPH_URL = `${MODEL_ROOT}kim_vocals_core_t801_webgpu.onnx`;
export const MODEL_DATA_URL = `${MODEL_ROOT}kim_vocals_core_t801_webgpu.onnx.data`;
export const MODEL_DATA_PATH = "kim_vocals_core_t801_webgpu.onnx.data";

export type StereoAudio = {
  left: Float32Array<ArrayBuffer>;
  right: Float32Array<ArrayBuffer>;
  sampleRate: number;
};

export type SeparationProgress = {
  phase: "model" | "separating" | "finishing";
  fraction: number;
  detail: string;
  etaSeconds?: number;
  processedSeconds?: number;
  totalSeconds?: number;
};

const reflectPad = (
  source: Float32Array<ArrayBuffer>,
  left: number,
  right: number,
): Float32Array<ArrayBuffer> => {
  const output = new Float32Array(left + source.length + right);
  output.set(source, left);
  for (let i = 0; i < left; i += 1) output[left - 1 - i] = source[i + 1];
  for (let i = 0; i < right; i += 1) output[left + source.length + i] = source[source.length - 2 - i];
  return output;
};

const makeChunk = (
  left: Float32Array<ArrayBuffer>,
  right: Float32Array<ArrayBuffer>,
  position: number,
): { data: Float32Array<ArrayBuffer>; length: number } => {
  const length = Math.min(CHUNK_SAMPLES, left.length - position);
  const data = new Float32Array(CHUNK_SAMPLES * 2);
  data.set(left.subarray(position, position + length), 0);
  data.set(right.subarray(position, position + length), CHUNK_SAMPLES);
  if (length < CHUNK_SAMPLES && length > CHUNK_SAMPLES / 2 + 1) {
    for (let i = length; i < CHUNK_SAMPLES; i += 1) {
      const reflected = 2 * length - 2 - i;
      data[i] = data[reflected];
      data[CHUNK_SAMPLES + i] = data[CHUNK_SAMPLES + reflected];
    }
  }
  return { data, length };
};

const fadeAt = (index: number, first: boolean, last: boolean): number => {
  if (!first && index < FADE) return index / (FADE - 1);
  if (!last && index >= CHUNK_SAMPLES - FADE) return (CHUNK_SAMPLES - 1 - index) / (FADE - 1);
  return 1;
};

const finite = (value: number): number => (Number.isFinite(value) ? value : 0);

export async function separateVocals(
  source: StereoAudio,
  onProgress: (progress: SeparationProgress) => void,
): Promise<{ vocals: StereoAudio; instrumental: StereoAudio }> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable. Use a current Chrome or Edge browser.");
  if (source.sampleRate !== SAMPLE_RATE) throw new Error(`Expected ${SAMPLE_RATE} Hz audio.`);

  onProgress({ phase: "model", fraction: 0.02, detail: "Loading the Kimberley Jensen model…" });
  const runtime = await createVocalsGpuRuntime({
    graph: {
      nFft: 2048,
      hop: 441,
      frames: 801,
      channels: 2,
      chunkSamples: CHUNK_SAMPLES,
      inputName: "stft_repr",
      outputName: "masks",
      minStorageBuffersPerShaderStage: 9,
    },
    modelUrl: MODEL_GRAPH_URL,
    modelDataUrl: MODEL_DATA_URL,
    modelDataPath: MODEL_DATA_PATH,
  });

  const shouldBorder = source.left.length > BORDER * 2;
  const left = shouldBorder ? reflectPad(source.left, BORDER, BORDER) : source.left;
  const right = shouldBorder ? reflectPad(source.right, BORDER, BORDER) : source.right;
  const resultL = new Float32Array(left.length);
  const resultR = new Float32Array(right.length);
  const counter = new Float32Array(left.length);
  const chunks = Math.ceil(left.length / STEP);
  const estimate = new Float32Array(CHUNK_SAMPLES * 2);
  const separationStarted = performance.now();
  const totalSeconds = source.left.length / SAMPLE_RATE;

  try {
    let chunkIndex = 0;
    for (let position = 0; position < left.length; position += STEP) {
      const { data, length } = makeChunk(left, right, position);
      onProgress({
        phase: "separating",
        fraction: chunkIndex / chunks,
        detail: chunkIndex === 0 ? "Listening closely and finding the voice…" : "Separating the voice from the music…",
        totalSeconds,
      });
      await runtime.processChunk({ input: data, output: estimate });
      const first = position === 0;
      const last = position + CHUNK_SAMPLES >= left.length;
      for (let i = 0; i < length; i += 1) {
        const weight = fadeAt(i, first, last);
        const target = position + i;
        resultL[target] += finite(estimate[i]) * weight;
        resultR[target] += finite(estimate[CHUNK_SAMPLES + i]) * weight;
        counter[target] += weight;
      }
      chunkIndex += 1;
      const elapsedSeconds = (performance.now() - separationStarted) / 1000;
      const completedFraction = chunkIndex / chunks;
      onProgress({
        phase: "separating",
        fraction: completedFraction,
        detail: "Separating the voice from the music…",
        etaSeconds: (elapsedSeconds / chunkIndex) * (chunks - chunkIndex),
        processedSeconds: totalSeconds * completedFraction,
        totalSeconds,
      });
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
  } finally {
    await runtime.release();
  }

  onProgress({ phase: "finishing", fraction: 0.98, detail: "Assembling lossless stems…" });
  const cropStart = shouldBorder ? BORDER : 0;
  const length = source.left.length;
  const vocalsL = new Float32Array(length);
  const vocalsR = new Float32Array(length);
  const instrumentalL = new Float32Array(length);
  const instrumentalR = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const index = cropStart + i;
    const weight = counter[index] || 1;
    vocalsL[i] = finite(resultL[index] / weight);
    vocalsR[i] = finite(resultR[index] / weight);
    instrumentalL[i] = source.left[i] - vocalsL[i];
    instrumentalR[i] = source.right[i] - vocalsR[i];
  }
  const common = { sampleRate: SAMPLE_RATE };
  return {
    vocals: { ...common, left: vocalsL, right: vocalsR },
    instrumental: { ...common, left: instrumentalL, right: instrumentalR },
  };
}
