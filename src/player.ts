import type { StereoAudio } from "./separation";

export type TrackId = "original" | "vocals" | "instrumental";

export class StemPlayer {
  private context = new AudioContext();
  private tracks = new Map<TrackId, StereoAudio>();
  private sources: AudioBufferSourceNode[] = [];
  private gains = new Map<TrackId, GainNode>();
  private startedAt = 0;
  private offset = 0;
  private playing = false;
  private active: TrackId = "original";
  onState?: (playing: boolean) => void;

  constructor() {
    for (const id of ["original", "vocals", "instrumental"] as TrackId[]) {
      const gain = this.context.createGain();
      gain.connect(this.context.destination);
      this.gains.set(id, gain);
    }
    this.applyGains();
  }

  setTracks(tracks: Partial<Record<TrackId, StereoAudio>>): void {
    this.pause();
    this.tracks = new Map(Object.entries(tracks) as [TrackId, StereoAudio][]);
    this.offset = 0;
  }

  setActive(id: TrackId): void {
    if (!this.tracks.has(id)) return;
    this.active = id;
    this.applyGains();
  }

  getActive(): TrackId { return this.active; }
  getDuration(): number { return (this.tracks.get("original")?.left.length ?? 0) / 44_100; }
  getPosition(): number {
    if (!this.playing) return this.offset;
    return Math.min(this.getDuration(), this.offset + this.context.currentTime - this.startedAt);
  }
  isPlaying(): boolean { return this.playing; }

  async toggle(): Promise<void> {
    if (this.playing) this.pause(); else await this.play();
  }

  async play(): Promise<void> {
    if (!this.tracks.size) return;
    if (this.offset >= this.getDuration()) this.offset = 0;
    await this.context.resume();
    this.sources = [];
    for (const [id, track] of this.tracks) {
      const buffer = this.context.createBuffer(2, track.left.length, track.sampleRate);
      buffer.copyToChannel(track.left, 0);
      buffer.copyToChannel(track.right, 1);
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.gains.get(id)!);
      source.start(0, this.offset);
      this.sources.push(source);
    }
    this.startedAt = this.context.currentTime;
    this.playing = true;
    this.sources[0]?.addEventListener("ended", () => {
      if (this.getPosition() >= this.getDuration() - 0.05) {
        this.playing = false;
        this.offset = 0;
        this.onState?.(false);
      }
    }, { once: true });
    this.onState?.(true);
  }

  pause(): void {
    if (this.playing) this.offset = this.getPosition();
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already stopped */ }
    }
    this.sources = [];
    this.playing = false;
    this.onState?.(false);
  }

  seek(seconds: number): void {
    const resume = this.playing;
    this.pause();
    this.offset = Math.max(0, Math.min(this.getDuration(), seconds));
    if (resume) void this.play();
  }

  private applyGains(): void {
    for (const [id, gain] of this.gains) gain.gain.value = id === this.active ? 1 : 0;
  }
}

export function waveformPeaks(audio: StereoAudio, bins = 1200): Float32Array {
  const output = new Float32Array(bins);
  const stride = Math.max(1, Math.floor(audio.left.length / bins));
  for (let x = 0; x < bins; x += 1) {
    let peak = 0;
    const end = Math.min(audio.left.length, (x + 1) * stride);
    for (let i = x * stride; i < end; i += 1) peak = Math.max(peak, Math.abs(audio.left[i]), Math.abs(audio.right[i]));
    output[x] = peak;
  }
  return output;
}

export function drawWaveform(canvas: HTMLCanvasElement, peaks: Float32Array, progress: number, color: string): void {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const mid = height / 2;
  const max = Math.max(...peaks, 0.001);
  const bars = Math.floor(width / 3);
  for (let x = 0; x < bars; x += 1) {
    const sample = peaks[Math.floor((x / bars) * peaks.length)] / max;
    const h = Math.max(1.5, sample * (height - 8));
    const position = x / bars;
    ctx.globalAlpha = position <= progress ? 1 : 0.22;
    ctx.fillStyle = color;
    ctx.fillRect(x * 3, mid - h / 2, 1.5, h);
  }
  ctx.globalAlpha = 1;
}
