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
type QueueJob = { id: string; file: File; name: string; workspace: boolean };

const $ = <T extends HTMLElement>(selector: string): T => document.querySelector(selector)!;
const fileInput = $("#file-input") as HTMLInputElement;
const dropCard = $("#drop-card");
const dropOverlay = $("#drop-overlay");
const workbench = $("#workbench");
const songSidebar = $("#song-sidebar");
const workspacePlaceholder = $("#workspace-placeholder");
const progressCard = $("#progress-card");
const studio = $("#studio");
const errorCard = $("#error-card");
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
let workspaceMode = localStorage.getItem("subtract-workspace-mode") === "true";
let selectedSongId: string | undefined;
let activeProgress = 0;
let noticeTimer: number | undefined;

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
};

const displayName = (name: string): string => name.replace(/\.[^.]+$/, "") || "track";

const refreshLayout = () => {
  const busy = Boolean(activeJob) || queue.length > 0;
  dropCard.hidden = workspaceMode || hasPlayback || busy;
  workbench.hidden = !workspaceMode && !hasPlayback && !busy;
  songSidebar.hidden = !workspaceMode;

  if (!workspaceMode) {
    progressCard.hidden = !activeJob;
    studio.hidden = !hasPlayback;
    workspacePlaceholder.hidden = true;
    return;
  }

  const showingProgress = Boolean(activeJob && selectedSongId === activeJob.id);
  const showingPlayer = Boolean(!showingProgress && hasPlayback && selectedSongId === currentResultId);
  progressCard.hidden = !showingProgress;
  studio.hidden = !showingPlayer;
  workspacePlaceholder.hidden = showingProgress || showingPlayer;
};

const setPlaceholder = (title: string, copy: string) => {
  $("#workspace-placeholder-title").textContent = title;
  $("#workspace-placeholder-copy").textContent = copy;
};

const hideError = () => {
  if (noticeTimer !== undefined) window.clearTimeout(noticeTimer);
  noticeTimer = undefined;
  errorCard.hidden = true;
};

const showError = (message: string, title = "We couldn’t split this song", temporary = false) => {
  if (noticeTimer !== undefined) window.clearTimeout(noticeTimer);
  $("#error-title").textContent = title;
  $("#error-message").textContent = message;
  $("#retry").hidden = temporary;
  errorCard.classList.toggle("notice", temporary);
  errorCard.hidden = false;
  noticeTimer = temporary ? window.setTimeout(hideError, 5_000) : undefined;
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
  activeProgress = fraction;
  const miniFill = activeJob ? document.querySelector<HTMLElement>(`[data-song-id="${activeJob.id}"] .song-mini-fill`) : null;
  if (miniFill) miniFill.style.width = `${Math.max(2, fraction * 100)}%`;
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
  selectedSongId = resultId;
  player.setTracks(tracks);
  activateTrack("vocals");
  $("#track-name").textContent = baseName;
  $("#track-info").textContent = `${durationLabel(audio.original.left.length / audio.original.sampleRate)} · Two tracks ready to play`;
  $("#total-time").textContent = durationLabel(audio.original.left.length / audio.original.sampleRate);
  hasPlayback = true;
  refreshLayout();
  renderSongList();
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

const completedItems = (): Array<HistorySummary & { saved: boolean }> => {
  const unfinishedIds = new Set([activeJob?.id, ...queue.map((job) => job.id)].filter(Boolean));
  const saved = persistentHistory.map((item) => ({ ...item, saved: true }));
  const session = [...sessionHistory.values()].map(({ vocals: _v, instrumental: _i, ...item }) => ({ ...item, saved: false }));
  return [...saved, ...session].filter((item) => !unfinishedIds.has(item.id)).sort((a, b) => b.createdAt - a.createdAt);
};

const updateStorageSummary = async () => {
  const savedBytes = persistentHistory.reduce((sum, item) => sum + item.bytes, 0);
  try {
    const estimate = await navigator.storage.estimate();
    if (estimate.usage !== undefined && estimate.quota !== undefined) {
      $("#song-storage").textContent = `Saved ${formatBytes(savedBytes)} · Site ${formatBytes(estimate.usage)} of ${formatBytes(estimate.quota)}`;
      return;
    }
  } catch { /* storage estimate is optional */ }
  $("#song-storage").textContent = `Saved songs use ${formatBytes(savedBytes)}`;
};

const loadHistoryItem = async (id: string) => {
  const token = ++historyLoadToken;
  selectedSongId = id;
  setPlaceholder("Opening this song", "Loading its voice and music tracks from this device…");
  renderSongList();
  try {
    const record = sessionHistory.get(id) ?? await getHistoryRecord(id);
    if (!record || token !== historyLoadToken) return;
    const [vocals, instrumental] = await Promise.all([decodeAudio(record.vocals), decodeAudio(record.instrumental)]);
    if (token !== historyLoadToken) return;
    const original = reconstructOriginal(vocals, instrumental);
    showTracks(record.name, { original, vocals, instrumental }, record.id);
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
    setPlaceholder("This song could not be opened", "You can delete it from the list and choose another song.");
  }
};

const removeHistoryItem = async (id: string, saved: boolean) => {
  if (saved) await deleteHistoryRecord(id);
  else sessionHistory.delete(id);
  if (currentResultId === id) {
    currentResultId = undefined;
    hasPlayback = false;
    player.pause();
  }
  if (selectedSongId === id) selectedSongId = undefined;
  await refreshHistory();
};

const selectSong = (id: string) => {
  historyLoadToken += 1;
  selectedSongId = id;
  if (activeJob?.id === id) {
    refreshLayout();
    renderSongList();
    return;
  }
  const waiting = queue.find((job) => job.id === id);
  if (waiting) {
    setPlaceholder("Waiting in queue", `${displayName(waiting.name)} will begin automatically.`);
    refreshLayout();
    renderSongList();
    return;
  }
  if (currentResultId === id && hasPlayback) {
    refreshLayout();
    renderSongList();
    return;
  }
  void loadHistoryItem(id);
};

function renderSongList(): void {
  const list = $("#song-list");
  const finished = completedItems();
  const jobs = activeJob ? [activeJob, ...queue] : queue;
  const total = jobs.length + finished.length;
  list.replaceChildren();
  $("#song-list-empty").hidden = total > 0;
  $("#clear-history").hidden = finished.length === 0;
  $("#song-count").textContent = `${total} song${total === 1 ? "" : "s"}`;

  for (const job of jobs) {
    const article = document.createElement("article");
    article.className = `song-item${job.id === selectedSongId ? " selected" : ""}`;
    article.dataset.songId = job.id;

    const open = document.createElement("button");
    open.className = "song-open";
    open.type = "button";
    open.addEventListener("click", () => selectSong(job.id));
    const title = document.createElement("strong");
    title.textContent = displayName(job.name);
    open.append(title);

    const state = document.createElement("span");
    state.className = `song-state ${job === activeJob ? "processing" : "queued"}`;
    state.textContent = job === activeJob ? "processing" : "queue";

    if (job === activeJob) {
      const miniTrack = document.createElement("div");
      miniTrack.className = "song-mini-track";
      const miniFill = document.createElement("i");
      miniFill.className = "song-mini-fill";
      miniFill.style.width = `${Math.max(2, activeProgress * 100)}%`;
      miniTrack.append(miniFill);
      open.append(miniTrack);
    }

    article.append(open, state);
    if (job !== activeJob) {
      const remove = document.createElement("button");
      remove.className = "song-delete";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${displayName(job.name)} from queue`);
      remove.addEventListener("click", () => {
        queue = queue.filter((queued) => queued.id !== job.id);
        if (selectedSongId === job.id) selectedSongId = activeJob?.id ?? currentResultId ?? completedItems()[0]?.id;
        renderSongList();
      });
      article.append(remove);
    }
    list.append(article);
  }

  for (const item of finished) {
    const article = document.createElement("article");
    article.className = `song-item${item.id === selectedSongId ? " selected" : ""}`;
    article.dataset.songId = item.id;

    const open = document.createElement("button");
    open.className = "song-open";
    open.type = "button";
    open.addEventListener("click", () => selectSong(item.id));
    const title = document.createElement("strong");
    title.textContent = displayName(item.name);
    const meta = document.createElement("span");
    meta.textContent = `${durationLabel(item.duration)} · ${formatBytes(item.bytes)}`;
    open.append(title, meta);

    const remove = document.createElement("button");
    remove.className = "song-delete";
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Delete ${displayName(item.name)}`);
    remove.addEventListener("click", () => void removeHistoryItem(item.id, item.saved));
    article.append(open, remove);
    list.append(article);
  }
  refreshLayout();
}

const refreshHistory = async () => {
  try {
    persistentHistory = await listHistoryRecords();
    if (!selectedSongId) selectedSongId = activeJob?.id ?? queue[0]?.id ?? currentResultId ?? completedItems()[0]?.id;
    renderSongList();
    await updateStorageSummary();
  } catch (error) {
    console.error(error);
    setPlaceholder("Saved songs are unavailable", "This browser could not open its local song library.");
  }
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

  const resultId = job.id;
  const shouldShowResult = !job.workspace || selectedSongId === job.id;
  if (shouldShowResult) showTracks(job.name, { original, ...separated }, resultId);

  // Clean mode is deliberately a one-song, one-result experience. The richer
  // workspace owns queueing and persistent history.
  if (!job.workspace) return;

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

  try {
    await saveHistoryRecord(record);
  } catch (error) {
    console.error(error);
    sessionHistory.set(record.id, record);
    showError("This browser could not save the newest song permanently. It will remain available until this tab is closed.");
  }
  await refreshHistory();
};

const runQueue = async () => {
  if (queueRunning) return;
  queueRunning = true;
  hideError();
  while (queue.length) {
    activeJob = queue.shift()!;
    activeProgress = 0;
    if (!selectedSongId) selectedSongId = activeJob.id;
    renderSongList();
    try {
      await processJob(activeJob);
    } catch (error) {
      console.error(error);
      showError(`${displayName(activeJob.name)}: ${error instanceof Error ? error.message : String(error)}`);
      if (selectedSongId === activeJob.id) selectedSongId = queue[0]?.id ?? currentResultId ?? completedItems()[0]?.id;
    }
    activeJob = undefined;
    activeProgress = 0;
    renderSongList();
  }
  queueRunning = false;
  refreshLayout();
};

const enqueueFiles = (files: Iterable<File>) => {
  let additions = [...files];
  if (!additions.length) return;
  if (!navigator.gpu) {
    showError("Please open this page in a recent version of Chrome or Edge on a newer computer.");
    return;
  }
  hideError();
  if (!workspaceMode) {
    if (activeJob || queueRunning) {
      showError("This song is still being separated. Wait for it to finish before choosing another one.");
      return;
    }
    additions = additions.slice(0, 1);
    player.pause();
    hasPlayback = false;
    currentResultId = undefined;
  }
  const jobs = additions.map((file) => ({
    id: crypto.randomUUID(),
    file,
    name: file.name,
    workspace: workspaceMode,
  }));
  queue.push(...jobs);
  if (!selectedSongId) selectedSongId = jobs[0]?.id;
  renderSongList();
  void runQueue();
};

const setWorkspaceMode = (enabled: boolean) => {
  workspaceMode = enabled;
  localStorage.setItem("subtract-workspace-mode", String(enabled));
  document.body.classList.toggle("workspace-mode", enabled);
  fileInput.multiple = enabled;
  refreshLayout();
  if (enabled) {
    void navigator.storage.persist?.();
    void refreshHistory();
  }
};

$("#mode-switch").addEventListener("click", () => {
  if (workspaceMode && (activeJob || queue.length)) {
    showError(
      "Your songs are still being processed. You can keep listening here, then return to simple mode when the queue is finished.",
      "The workspace is still busy",
      true,
    );
    return;
  }
  hideError();
  setWorkspaceMode(!workspaceMode);
});

$("#clear-history").addEventListener("click", () => {
  if (!window.confirm("Delete every result from history?")) return;
  void (async () => {
    historyLoadToken += 1;
    const finishedIds = new Set(completedItems().map((item) => item.id));
    await clearHistoryRecords();
    sessionHistory.clear();
    if (currentResultId && finishedIds.has(currentResultId)) {
      currentResultId = undefined;
      hasPlayback = false;
      player.pause();
    }
    if (selectedSongId && finishedIds.has(selectedSongId)) selectedSongId = activeJob?.id ?? queue[0]?.id;
    await refreshHistory();
  })();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files) enqueueFiles(fileInput.files);
  fileInput.value = "";
});
for (const selector of ["#drop-target", "#new-track", "#retry", "#add-song"]) {
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

if ("serviceWorker" in navigator) void navigator.serviceWorker.register("./sw.js");
void updateDevice();
setWorkspaceMode(workspaceMode);
requestAnimationFrame(drawAll);
