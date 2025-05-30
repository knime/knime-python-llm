/// <reference types="@knime/utils/globals.d.ts" />
/// <reference types="@knime/styles/config/svg.d.ts" />
/// <reference types="vite/client" />
/// <reference types="vite-svg-loader" />

// For more info on these env variables see .env.example
interface ImportMetaEnv {
  readonly VITE_LOG_LEVEL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
