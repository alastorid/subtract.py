import "./style.css";
import { decodeAudio, durationLabel, encodeFloatWav } from "./audio";
import {
  clearHistoryRecords,
  deleteHistoryRecord,
  getHistoryRecord,
  listHistoryRecords,
  saveHistoryRecord,
  type HistoryRecord,
  type HistorySummary,
} from "./history";
import {
  StemPlayer,
  addSegmentPeaks,
  drawBuildingWaveform,
  drawWaveform,
  waveformPeaks,
  type TrackId,
} from "./player";
import { separateVocals, type StereoAudio } from "./separation";

type AudioTracks = Record<TrackId, StereoAudio>;
type QueueJob = { id: string; file: File; name: string };

const $ = <T extends HTMLElement>(selector: string): T => document.querySelector(selector)!;
const fileInput = $("#file-input") as HTMLInputElement;
const dropCard = $("#drop-card");
const dropOverlay = $("#drop-overlay");
const queuePanel = $("#queue-panel");
const progressCard = $("#progress-card");
const studio = $("#studio");
const errorCard = $("#error-card");
const historyToggle = $("#history-toggle") as HTMLInputElement;
const player = new StemPlayer();

let tracks: Partial<Record<TrackId, StereoAudio>> = {};
let peaks: Partial<Record<TrackId, Float32Array>> = {};
let baseName = "track";
let hasPlayback = false;
let currentResultId: string | undefined;
let liveVocalsPeaks = new Float32Array(1200);
let liveInstrumentalPeaks = new Float32Array(1200);
let liveCompletion = 0;
let activeJob: QueueJob | undefined;
let queue: QueueJob[] = [];
let queueRunning = false;
let persistentHistory: HistorySummary[] = [];
const sessionHistory = new Map<string, HistoryRecord>();
let historyLoadToken = 0;
let historyEnabled = localStorage.getItem("subtract-history-enabled") !== "false";

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
};

const displayName = (name: string): string => name.replace(/\.[^.]+$/, "") || "track";

const refreshLayout = () => {
  const busy = Boolean(activeJob) || queue.length > 0;
  dropCard.hidden = hasPlayback || busy;
  queuePanel.hidden = !busy;
  progressCard.hidden = !activeJob;
  studio.hidden = !hasPlayback;
};

const showError = (message: string) => {
  $("#error-message").textContent = message;
  errorCard.hidden = false;
};

const hideError = () => { errorCard.hidden = true; };

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

const showTracks = (name: string, audio: AudioTracks, resultId: string) => {
  tracks = audio;
  peaks = {
    original: waveformPeaks(audio.original),
    vocals: waveformPeaks(audio.vocals),
    instrumental: waveformPeaks(audio.instrumental),
  };
  baseName = displayName(name);
  currentResultId = resultId;
  player.setTracks(tracks);
  activateTrack("vocals");
  $("#track-name").textContent = baseName;
  $("#track-info").textContent = `${durationLabel(audio.original.left.length / audio.original.sampleRate)} · Two tracks ready to play`;
  $("#total-time").textContent = durationLabel(audio.original.left.length / audio.original.sampleRate);
  hasPlayback = true;
  refreshLayout();
  renderHistory();
};

const reconstructOriginal = (vocals: StereoAudio, instrumental: StereoAudio): StereoAudio => {
  const length = Math.min(vocals.left.length, instrumental.left.length);
  const left = new Float32Array(length);
  const right = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    left[i] = vocals.left[i] + instrumental.left[i];
    right[i] = vocals.right[i] + instrumental.right[i];
  }
  return { left, right, sampleRate: vocals.sampleRate };
};

const historyItems = (): Array<HistorySummary & { saved: boolean }> => {
  const saved = persistentHistory.map((item) => ({ ...item, saved: true }));
  const session = [...sessionHistory.values()].map(({ vocals: _v, instrumental: _i, ...item }) => ({ ...item, saved: false }));
  return [...saved, ...session].sort((a, b) => b.createdAt - a.createdAt);
};

const updateStorageSummary = async () => {
  const savedBytes = persistentHistory.reduce((sum, item) => sum + item.bytes, 0);
  try {
    const estimate = await navigator.storage.estimate();
    if (estimate.usage !== undefined && estimate.quota !== undefined) {
      $("#history-storage").textContent = `History ${formatBytes(savedBytes)} · This site ${formatBytes(estimate.usage)} of ${formatBytes(estimate.quota)}`;
      return;
    }
  } catch { /* storage estimate is optional */ }
  $("#history-storage").textContent = `History uses ${formatBytes(savedBytes)}`;
};

const loadHistoryItem = async (id: string) => {
  const token = ++historyLoadToken;
  $("#history-message").textContent = "Opening your saved tracks…";
  try {
    const record = sessionHistory.get(id) ?? await getHistoryRecord(id);
    if (!record || token !== historyLoadToken) return;
    const [vocals, instrumental] = await Promise.all([decodeAudio(record.vocals), decodeAudio(record.instrumental)]);
    if (token !== historyLoadToken) return;
    const original = reconstructOriginal(vocals, instrumental);
    showTracks(record.name, { original, vocals, instrumental }, record.id);
    $("#history-message").textContent = "";
    studio.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
    $("#history-message").textContent = "This result could not be opened.";
  }
};

const removeHistoryItem = async (id: string, saved: boolean) => {
  if (saved) await deleteHistoryRecord(id);
  else sessionHistory.delete(id);
  if (currentResultId === id) currentResultId = undefined;
  await refreshHistory();
};

function renderHistory(): void {
  const list = $("#history-list");
  const items = historyItems();
  list.replaceChildren();
  $("#history-empty").hidden = items.length > 0;
  $("#clear-history").hidden = items.length === 0;
  $("#history-count").textContent = `${items.length} result${items.length === 1 ? "" : "s"}`;

  for (const item of items) {
    const article = document.createElement("article");
    article.className = `history-item${item.id === currentResultId ? " current" : ""}`;

    const open = document.createElement("button");
    open.className = "history-open";
    open.type = "button";
    open.addEventListener("click", () => void loadHistoryItem(item.id));
    const title = document.createElement("strong");
    title.textContent = displayName(item.name);
    const meta = document.createElement("span");
    const date = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(item.createdAt);
    meta.textContent = `${durationLabel(item.duration)} · ${date}`;
    open.append(title, meta);

    const size = document.createElement("span");
    size.className = "history-size";
    size.textContent = `${item.saved ? "Saved" : "This session"} · ${formatBytes(item.bytes)}`;

    const remove = document.createElement("button");
    remove.className = "history-delete";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Delete this result";
    remove.setAttribute("aria-label", `Delete ${displayName(item.name)}`);
    remove.addEventListener("click", () => void removeHistoryItem(item.id, item.saved));
    article.append(open, size, remove);
    list.append(article);
  }
}

const refreshHistory = async () => {
  try {
    persistentHistory = await listHistoryRecords();
    renderHistory();
    await updateStorageSummary();
  } catch (error) {
    console.error(error);
    $("#history-message").textContent = "Saved history is unavailable in this browser.";
  }
};

const renderQueue = () => {
  const jobs = activeJob ? [activeJob, ...queue] : queue;
  $("#queue-count").textContent = activeJob ? `${queue.length} waiting` : `${queue.length} waiting to start`;
  const list = $("#queue-list");
  list.replaceChildren();
  jobs.forEach((job, index) => {
    const row = document.createElement("div");
    row.className = `queue-item${job === activeJob ? " active" : ""}`;
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("strong");
    name.textContent = displayName(job.name);
    const status = document.createElement("small");
    status.textContent = job === activeJob ? "Processing now" : "Waiting";
    row.append(number, name, status);
    list.append(row);
  });
  refreshLayout();
};

const processJob = async (job: QueueJob) => {
  liveVocalsPeaks = new Float32Array(1200);
  liveInstrumentalPeaks = new Float32Array(1200);
  liveCompletion = 0;
  $("#processing-name").textContent = displayName(job.name);
  $("#progress-processed").textContent = "Waiting to begin";
  updateProgress("Opening your song", "Reading the audio on this device…", 0.01);

  const original = await decodeAudio(job.file);
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

  const resultId = crypto.randomUUID();
  if (!hasPlayback) showTracks(job.name, { original, ...separated }, resultId);

  const vocals = encodeFloatWav(separated.vocals);
  const instrumental = encodeFloatWav(separated.instrumental);
  const record: HistoryRecord = {
    id: resultId,
    name: job.name,
    createdAt: Date.now(),
    duration: original.left.length / original.sampleRate,
    bytes: vocals.size + instrumental.size,
    vocals,
    instrumental,
  };

  if (historyEnabled) {
    try {
      await saveHistoryRecord(record);
    } catch (error) {
      console.error(error);
      sessionHistory.set(record.id, record);
      $("#history-message").textContent = "Storage is full, so the newest result will last only for this session.";
    }
  } else {
    sessionHistory.set(record.id, record);
  }
  await refreshHistory();
};

const runQueue = async () => {
  if (queueRunning) return;
  queueRunning = true;
  hideError();
  while (queue.length) {
    activeJob = queue.shift()!;
    renderQueue();
    try {
      await processJob(activeJob);
    } catch (error) {
      console.error(error);
      showError(`${displayName(activeJob.name)}: ${error instanceof Error ? error.message : String(error)}`);
    }
    activeJob = undefined;
    renderQueue();
  }
  queueRunning = false;
  refreshLayout();
};

const enqueueFiles = (files: Iterable<File>) => {
  const additions = [...files];
  if (!additions.length) return;
  if (!navigator.gpu) {
    showError("Please open this page in a recent version of Chrome or Edge on a newer computer.");
    return;
  }
  hideError();
  queue.push(...additions.map((file) => ({ id: crypto.randomUUID(), file, name: file.name })));
  renderQueue();
  void runQueue();
};

historyToggle.checked = historyEnabled;
$("#history-mode").textContent = historyEnabled ? "New results are saved on this device" : "New results last for this session only";
historyToggle.addEventListener("change", () => {
  historyEnabled = historyToggle.checked;
  localStorage.setItem("subtract-history-enabled", String(historyEnabled));
  $("#history-mode").textContent = historyEnabled ? "New results are saved on this device" : "New results last for this session only";
  if (historyEnabled) void navigator.storage.persist?.();
});

$("#clear-history").addEventListener("click", () => {
  if (!window.confirm("Delete every result from history?")) return;
  void (async () => {
    await clearHistoryRecords();
    sessionHistory.clear();
    currentResultId = undefined;
    await refreshHistory();
  })();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files) enqueueFiles(fileInput.files);
  fileInput.value = "";
});
for (const selector of ["#drop-target", "#new-track", "#retry"]) {
  $(selector).addEventListener("click", () => fileInput.click());
}

let dragDepth = 0;
document.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  dropOverlay.hidden = false;
});
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropOverlay.hidden = true;
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropOverlay.hidden = true;
  if (event.dataTransfer?.files) enqueueFiles(event.dataTransfer.files);
});

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

if (historyEnabled) void navigator.storage.persist?.();
if ("serviceWorker" in navigator) void navigator.serviceWorker.register("./sw.js");
void updateDevice();
void refreshHistory();
refreshLayout();
requestAnimationFrame(drawAll);
