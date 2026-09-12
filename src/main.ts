import "./style.css";
import { decodeAudio, durationLabel, encodeFloatWav } from "./audio";
import {
  StemPlayer,
  addSegmentPeaks,
  drawBuildingWaveform,
  drawWaveform,
  waveformPeaks,
  type TrackId,
} from "./player";
import { separateVocals, type StereoAudio } from "./separation";

const $ = <T extends HTMLElement>(selector: string): T => document.querySelector(selector)!;
const fileInput = $("#file-input") as HTMLInputElement;
const dropCard = $("#drop-card");
const progressCard = $("#progress-card");
const studio = $("#studio");
const errorCard = $("#error-card");
const player = new StemPlayer();
let tracks: Partial<Record<TrackId, StereoAudio>> = {};
let peaks: Partial<Record<TrackId, Float32Array>> = {};
let baseName = "track";
let liveVocalsPeaks = new Float32Array(1200);
let liveInstrumentalPeaks = new Float32Array(1200);
let liveCompletion = 0;

const setView = (view: "drop" | "progress" | "studio" | "error") => {
  dropCard.hidden = view !== "drop";
  progressCard.hidden = view !== "progress";
  studio.hidden = view !== "studio";
  errorCard.hidden = view !== "error";
};

const etaLabel = (seconds?: number): string => {
  if (seconds === undefined || !Number.isFinite(seconds)) return "Calculating…";
  if (seconds < 8) return "Less than 10 seconds left";
  if (seconds < 60) return `${Math.ceil(seconds / 5) * 5} seconds left`;
  const minutes = Math.ceil(seconds / 60);
  return `About ${minutes} minute${minutes === 1 ? "" : "s"} left`;
};

const updateProgress = (
  title: string,
  detail: string,
  fraction: number,
  etaSeconds?: number,
  processedSeconds?: number,
  totalSeconds?: number,
) => {
  $("#progress-title").textContent = title;
  $("#progress-detail").textContent = detail;
  $("#progress-percent").textContent = `${Math.round(fraction * 100)}%`;
  ($("#progress-fill") as HTMLElement).style.width = `${Math.max(2, fraction * 100)}%`;
  $("#progress-eta").textContent = etaLabel(etaSeconds);
  if (totalSeconds !== undefined) {
    $("#progress-processed").textContent = `${durationLabel(processedSeconds ?? 0)} of ${durationLabel(totalSeconds)}`;
  }
};

const updateDevice = async () => {
  const pill = $("#device-pill");
  if (!navigator.gpu) {
    pill.classList.add("unsupported");
    $("#device-label").textContent = "WebGPU unavailable";
    $("#gpu-name").textContent = "Not available in this browser";
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  pill.classList.toggle("unsupported", !adapter);
  $("#device-label").textContent = adapter ? "WebGPU ready" : "WebGPU unavailable";
  if (!adapter) {
    $("#gpu-name").textContent = "No compatible graphics adapter found";
    return;
  }
  const info = adapter.info;
  const graphicsName = info.description || info.device || info.architecture || info.vendor || "Available graphics adapter";
  const bytes = Number(adapter.limits.maxBufferSize);
  $("#gpu-name").textContent = graphicsName;
  $("#gpu-buffer").textContent = bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
    : `${Math.round(bytes / 1024 ** 2)} MB`;
  $("#gpu-storage").textContent = String(adapter.limits.maxStorageBuffersPerShaderStage);
  $("#gpu-workgroup").textContent = `${adapter.limits.maxComputeWorkgroupSizeX} × ${adapter.limits.maxComputeWorkgroupSizeY}`;
};

const downloadTrack = (id: "vocals" | "instrumental") => {
  const audio = tracks[id];
  if (!audio) return;
  const url = URL.createObjectURL(encodeFloatWav(audio));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${baseName}_${id}.wav`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
};

const drawAll = () => {
  if (!progressCard.hidden) {
    drawBuildingWaveform($("#live-vocals-wave") as HTMLCanvasElement, liveVocalsPeaks, liveCompletion, "#ff5c35");
    drawBuildingWaveform($("#live-instrumental-wave") as HTMLCanvasElement, liveInstrumentalPeaks, liveCompletion, "#39d4a0");
  }
  const progress = player.getDuration() ? player.getPosition() / player.getDuration() : 0;
  const activePeaks = peaks[player.getActive()] ?? peaks.vocals;
  if (activePeaks) drawWaveform($("#master-wave") as HTMLCanvasElement, activePeaks, progress, "#f5f2ea");
  const colors: Record<TrackId, string> = { original: "#c9c7c1", vocals: "#ff5c35", instrumental: "#39d4a0" };
  for (const id of ["original", "vocals", "instrumental"] as TrackId[]) {
    const canvas = document.querySelector(`[data-wave="${id}"]`) as HTMLCanvasElement;
    if (canvas && peaks[id]) drawWaveform(canvas, peaks[id]!, progress, colors[id]);
  }
  $("#current-time").textContent = durationLabel(player.getPosition());
  ($("#playhead") as HTMLElement).style.left = `${progress * 100}%`;
  requestAnimationFrame(drawAll);
};

const activateTrack = (id: TrackId) => {
  player.setActive(id);
  document.querySelectorAll<HTMLElement>(".stem").forEach((row) => {
    const selected = row.dataset.track === id;
    row.classList.toggle("active", selected);
    row.querySelector("[role=radio]")?.setAttribute("aria-checked", String(selected));
  });
};

const processFile = async (file: File) => {
  if (!navigator.gpu) {
    $("#error-message").textContent = "Please open this page in a recent version of Chrome or Edge on a newer computer.";
    setView("error");
    return;
  }
  baseName = file.name.replace(/\.[^.]+$/, "") || "track";
  liveVocalsPeaks = new Float32Array(1200);
  liveInstrumentalPeaks = new Float32Array(1200);
  liveCompletion = 0;
  setView("progress");
  $("#progress-processed").textContent = "Waiting to begin";
  updateProgress("Opening your song", "Reading the audio on this device…", 0.01);
  try {
    const original = await decodeAudio(file);
    tracks = { original };
    peaks = { original: waveformPeaks(original) };
    updateProgress("Getting ready", "Preparing the song separator. This takes longer the first time…", 0.03);
    const separated = await separateVocals(original, ({ phase, fraction, detail, etaSeconds, processedSeconds, totalSeconds }) => {
      const title = phase === "model" ? "Getting ready" : phase === "separating" ? "Splitting your song" : "Almost done";
      const scaled = phase === "model" ? fraction * 0.08 : phase === "separating" ? 0.08 + fraction * 0.9 : 0.99;
      updateProgress(title, detail, scaled, etaSeconds, processedSeconds, totalSeconds);
    }, ({ startSample, endSample, vocalsLeft, vocalsRight, instrumentalLeft, instrumentalRight }) => {
      addSegmentPeaks(liveVocalsPeaks, vocalsLeft, vocalsRight, startSample, original.left.length);
      addSegmentPeaks(liveInstrumentalPeaks, instrumentalLeft, instrumentalRight, startSample, original.left.length);
      liveCompletion = endSample / original.left.length;
    });
    tracks = { original, ...separated };
    peaks = {
      original: peaks.original,
      vocals: waveformPeaks(separated.vocals),
      instrumental: waveformPeaks(separated.instrumental),
    };
    player.setTracks(tracks);
    activateTrack("vocals");
    $("#track-name").textContent = baseName;
    $("#track-info").textContent = `${durationLabel(original.left.length / original.sampleRate)} · Two tracks ready to play`;
    $("#total-time").textContent = durationLabel(original.left.length / original.sampleRate);
    setView("studio");
  } catch (error) {
    console.error(error);
    $("#error-message").textContent = error instanceof Error ? error.message : String(error);
    setView("error");
  }
};

$("#drop-target").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files?.[0] && void processFile(fileInput.files[0]));
for (const event of ["dragenter", "dragover"] as const) {
  document.addEventListener(event, (e) => { e.preventDefault(); dropCard.classList.add("dragging"); });
}
for (const event of ["dragleave", "drop"] as const) {
  document.addEventListener(event, (e) => { e.preventDefault(); dropCard.classList.remove("dragging"); });
}
document.addEventListener("drop", (e) => e.dataTransfer?.files[0] && void processFile(e.dataTransfer.files[0]));

$("#play").addEventListener("click", () => void player.toggle());
player.onState = (playing) => { $("#play-icon").textContent = playing ? "Ⅱ" : "▶"; };
$("#rewind").addEventListener("click", () => player.seek(player.getPosition() - 10));
$("#forward").addEventListener("click", () => player.seek(player.getPosition() + 10));
$("#master-wave").addEventListener("click", (event) => {
  const canvas = event.currentTarget as HTMLCanvasElement;
  player.seek(((event as MouseEvent).offsetX / canvas.clientWidth) * player.getDuration());
});
document.querySelectorAll<HTMLElement>(".stem-select").forEach((button) => button.addEventListener("click", () => {
  const row = button.closest<HTMLElement>(".stem");
  if (row) activateTrack(row.dataset.track as TrackId);
}));
document.querySelectorAll<HTMLElement>("[data-download]").forEach((button) => button.addEventListener("click", () => downloadTrack(button.dataset.download as "vocals" | "instrumental")));
for (const selector of ["#new-track", "#retry"]) $(selector).addEventListener("click", () => { player.pause(); fileInput.value = ""; setView("drop"); });

if ("serviceWorker" in navigator) void navigator.serviceWorker.register("./sw.js");
void updateDevice();
requestAnimationFrame(drawAll);
