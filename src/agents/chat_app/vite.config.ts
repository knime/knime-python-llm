import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import svgLoader from "vite-svg-loader";

import { svgoConfig } from "@knime/styles/config/svgo.config";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), svgLoader({ svgoConfig })],
  base: "",
  build: {
    outDir: '../../../',
  },
});
