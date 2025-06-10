import { URL, fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import svgLoader from "vite-svg-loader";

import { svgoConfig } from "@knime/styles/config/svgo.config";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), svgLoader({ svgoConfig })],
  base: "",
  build: {
    target: "esnext",
  },

  // TODO: remove this when we have builds fo that libs, without them the optimizer can break things
  optimizeDeps: {
    exclude: ["@knime/components", "@knime/utils"],
  },

  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
