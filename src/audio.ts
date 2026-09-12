import type { StereoAudio } from "./separation";
import { SAMPLE_RATE } from "./separation";

export async function decodeAudio(file: File): Promise<StereoAudio> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    let buffer = decoded;
    if (decoded.sampleRate !== SAMPLE_RATE) {
      const frames = Math.ceil(decoded.duration * SAMPLE_RATE);
      const offline = new OfflineAudioContext(2, frames, SAMPLE_RATE);
      const source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      buffer = await offline.startRendering();
    }
    const left = new Float32Array(buffer.getChannelData(0));
    const right = new Float32Array(buffer.getChannelData(Math.min(1, buffer.numberOfChannels - 1)));
    return { left, right, sampleRate: SAMPLE_RATE };
  } finally {
    await context.close();
  }
}

export function encodeFloatWav(audio: StereoAudio): Blob {
  const samples = audio.left.length;
  const bytes = 44 + samples * 2 * 4;
  const buffer = new ArrayBuffer(bytes);
  const view = new DataView(buffer);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, bytes - 8, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * 8, true);
  view.setUint16(32, 8, true);
  view.setUint16(34, 32, true);
  text(36, "data");
  view.setUint32(40, samples * 8, true);
  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    view.setFloat32(offset, audio.left[i], true);
    view.setFloat32(offset + 4, audio.right[i], true);
    offset += 8;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export const durationLabel = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
};
