import { createVocalsGpuRuntime } from "./vendor/musetric/ai/runtime/vocals/vocalsRuntime";

export const SAMPLE_RATE = 44_100;
const CHUNK_SAMPLES = 352_800;
// Two seconds of overlap is still comfortably wider than the 0.8 second fades,
// while cutting roughly one third of the model passes compared with 50% overlap.
const STEP = 264_600;
const FADE = CHUNK_SAMPLES / 10;
const BORDER = CHUNK_SAMPLES - STEP;
const MODEL_CACHE = "subtract-model-v1";
const SPEED_KEY = "subtract-seconds-per-section-v1";

const MODEL_ROOT = new URL(`${import.meta.env.BASE_URL}models/`, window.location.href).href;
export const MODEL_GRAPH_URL = `${MODEL_ROOT}kim_vocals_core_t801_webgpu.onnx`;
export const MODEL_DATA_URL = `${MODEL_ROOT}kim_vocals_core_t801_webgpu.onnx.data`;
export const MODEL_DATA_PATH = "kim_vocals_core_t801_webgpu.onnx.data";
const MODEL_FILES = [
  { url: MODEL_GRAPH_URL, bytes: 5_100_916 },
  { url: MODEL_DATA_URL, bytes: 458_075_020 },
] as const;

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

export type PartialSeparation = {
  startSample: number;
  endSample: number;
  vocalsLeft: Float32Array<ArrayBuffer>;
  vocalsRight: Float32Array<ArrayBuffer>;
  instrumentalLeft: Float32Array<ArrayBuffer>;
  instrumentalRight: Float32Array<ArrayBuffer>;
};

const estimatedSectionSeconds = (): number => {
  try {
    const saved = Number(localStorage.getItem(SPEED_KEY));
    if (Number.isFinite(saved) && saved > 0.1 && saved < 120) return saved;
  } catch { /* storage can be disabled */ }
  return 5;
};

const saveSectionSeconds = (seconds: number): void => {
  try { localStorage.setItem(SPEED_KEY, String(seconds)); } catch { /* storage can be disabled */ }
};

const ensureServiceWorkerControl = async (): Promise<void> => {
  if (!("serviceWorker" in navigator)) return;
  await navigator.serviceWorker.register(new URL(`${import.meta.env.BASE_URL}sw.js`, window.location.href));
  await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller) return;
  await new Promise<void>((resolve) => {
    const timeout = window.setTimeout(resolve, 2_000);
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
  });
};

const prepareModelFiles = async (
  computeSeconds: number,
  onProgress: (progress: SeparationProgress) => void,
): Promise<void> => {
  if (!("caches" in window)) return;
  await ensureServiceWorkerControl();
  const cache = await caches.open(MODEL_CACHE);
  const availability = await Promise.all(MODEL_FILES.map(async (file) => ({
    ...file,
    cached: Boolean(await cache.match(file.url)),
  })));
  const totalBytes = availability.reduce((sum, file) => sum + file.bytes, 0);
  const cachedBytes = availability.filter((file) => file.cached).reduce((sum, file) => sum + file.bytes, 0);
  const transferTotal = totalBytes - cachedBytes;
  if (transferTotal === 0) {
    onProgress({ phase: "model", fraction: 1, detail: "Starting the song separator…", etaSeconds: computeSeconds });
    return;
  }

  let transferred = 0;
  const started = performance.now();
  for (const file of availability) {
    if (file.cached) continue;
    const response = await fetch(file.url, {
      cache: "no-store",
      headers: { "x-subtract-prime": "1" },
    });
    if (!response.ok || !response.body) throw new Error("The song separator could not be downloaded. Please try again.");
    const stream = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        transferred += chunk.byteLength;
        const seconds = Math.max(0.1, (performance.now() - started) / 1000);
        const bytesPerSecond = transferred / seconds;
        const downloadEta = (transferTotal - transferred) / bytesPerSecond;
        const fraction = Math.min(1, (cachedBytes + transferred) / totalBytes);
        onProgress({
          phase: "model",
          fraction,
          detail: `Getting the song separator ready… ${Math.round(fraction * 100)}%`,
          etaSeconds: downloadEta + computeSeconds,
        });
        controller.enqueue(chunk);
      },
    }));
    await cache.put(file.url, new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    }));
  }
  onProgress({ phase: "model", fraction: 1, detail: "Starting the song separator…", etaSeconds: computeSeconds });
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
  onPartial?: (partial: PartialSeparation) => void,
): Promise<{ vocals: StereoAudio; instrumental: StereoAudio }> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable. Use a current Chrome or Edge browser.");
  if (source.sampleRate !== SAMPLE_RATE) throw new Error(`Expected ${SAMPLE_RATE} Hz audio.`);

  const shouldBorder = source.left.length > BORDER * 2;
  const paddedLength = source.left.length + (shouldBorder ? BORDER * 2 : 0);
  const chunks = Math.ceil(paddedLength / STEP);
  const priorSectionSeconds = estimatedSectionSeconds();
  onProgress({
    phase: "model",
    fraction: 0,
    detail: "Checking the song separator…",
    etaSeconds: priorSectionSeconds * chunks,
    processedSeconds: 0,
    totalSeconds: source.left.length / SAMPLE_RATE,
  });
  await prepareModelFiles(priorSectionSeconds * chunks, onProgress);
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

  const left = shouldBorder ? reflectPad(source.left, BORDER, BORDER) : source.left;
  const right = shouldBorder ? reflectPad(source.right, BORDER, BORDER) : source.right;
  const resultL = new Float32Array(left.length);
  const resultR = new Float32Array(right.length);
  const counter = new Float32Array(left.length);
  const estimate = new Float32Array(CHUNK_SAMPLES * 2);
  const totalSeconds = source.left.length / SAMPLE_RATE;
  const chunkDurations: number[] = [];
  let emittedUntil = 0;

  try {
    let chunkIndex = 0;
    for (let position = 0; position < left.length; position += STEP) {
      const chunkStarted = performance.now();
      const { data, length } = makeChunk(left, right, position);
      onProgress({
        phase: "separating",
        fraction: chunkIndex / chunks,
        detail: chunkIndex === 0 ? "Listening closely and finding the voice…" : "Separating the voice from the music…",
        etaSeconds: priorSectionSeconds * (chunks - chunkIndex),
        processedSeconds: totalSeconds * (chunkIndex / chunks),
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
      chunkDurations.push((performance.now() - chunkStarted) / 1000);
      const completedFraction = chunkIndex / chunks;
      const stableDurations = chunkDurations.length > 1 ? chunkDurations.slice(1).slice(-3) : [];
      const secondsPerChunk = stableDurations.length
        ? stableDurations.reduce((sum, value) => sum + value, 0) / stableDurations.length
        : priorSectionSeconds;
      const finalizedPaddedEnd = position + CHUNK_SAMPLES >= left.length ? left.length : position + STEP;
      const finalizedSourceEnd = Math.min(source.left.length, Math.max(0, finalizedPaddedEnd - (shouldBorder ? BORDER : 0)));
      if (onPartial && finalizedSourceEnd > emittedUntil) {
        const partialLength = finalizedSourceEnd - emittedUntil;
        const vocalsLeft = new Float32Array(partialLength);
        const vocalsRight = new Float32Array(partialLength);
        const instrumentalLeft = new Float32Array(partialLength);
        const instrumentalRight = new Float32Array(partialLength);
        for (let i = 0; i < partialLength; i += 1) {
          const sourceIndex = emittedUntil + i;
          const paddedIndex = sourceIndex + (shouldBorder ? BORDER : 0);
          const weight = counter[paddedIndex] || 1;
          vocalsLeft[i] = finite(resultL[paddedIndex] / weight);
          vocalsRight[i] = finite(resultR[paddedIndex] / weight);
          instrumentalLeft[i] = source.left[sourceIndex] - vocalsLeft[i];
          instrumentalRight[i] = source.right[sourceIndex] - vocalsRight[i];
        }
        onPartial({
          startSample: emittedUntil,
          endSample: finalizedSourceEnd,
          vocalsLeft,
          vocalsRight,
          instrumentalLeft,
          instrumentalRight,
        });
        emittedUntil = finalizedSourceEnd;
      }
      onProgress({
        phase: "separating",
        fraction: completedFraction,
        detail: "Separating the voice from the music…",
        etaSeconds: secondsPerChunk * (chunks - chunkIndex),
        processedSeconds: finalizedSourceEnd / SAMPLE_RATE,
        totalSeconds,
      });
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }
  } finally {
    await runtime.release();
  }

  const reusableDurations = chunkDurations.length > 1 ? chunkDurations.slice(1) : chunkDurations;
  if (reusableDurations.length) {
    saveSectionSeconds(reusableDurations.reduce((sum, value) => sum + value, 0) / reusableDurations.length);
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
