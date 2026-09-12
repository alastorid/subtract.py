import { createReadStream, existsSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";
import { join } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "@musetric/fft/gpu": fileURLToPath(
        new URL("./src/vendor/musetric/fft/index.ts", import.meta.url),
      ),
      "@musetric/utils": fileURLToPath(
        new URL("./src/vendor/musetric/resourceCell.ts", import.meta.url),
      ),
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
  plugins: [{
    name: "local-webgpu-model",
    configureServer(server) {
      server.middlewares.use("/models/", (request, response, next) => {
        const name = request.url?.replace(/^\//, "");
        if (!name || !/^kim_vocals_core_t801_webgpu\.onnx(?:\.data)?$/.test(name)) return next();
        const path = join(fileURLToPath(new URL("./model-build", import.meta.url)), name);
        if (!existsSync(path)) return next();
        response.setHeader("Content-Type", "application/octet-stream");
        createReadStream(path).pipe(response);
      });
    },
  }],
});
