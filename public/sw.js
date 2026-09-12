const CACHE = "subtract-model-v1";
const isModel = (url) => new URL(url).pathname.includes("/models/kim_vocals_core_t801_webgpu.onnx");
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (event) => {
  if (!isModel(event.request.url)) return;
  // The page streams first-time downloads into this same cache so it can show
  // byte-level progress. Avoid a second clone/write while that is happening.
  if (event.request.headers.get("x-subtract-prime") === "1") {
    event.respondWith(fetch(event.request));
    return;
  }
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(event.request);
    if (cached) return cached;
    const response = await fetch(event.request);
    if (response.ok) event.waitUntil(cache.put(event.request, response.clone()));
    return response;
  })());
});
