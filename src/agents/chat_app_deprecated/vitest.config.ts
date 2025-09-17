import { fileURLToPath } from "url";

import { defineConfig } from "vitest/config";

import viteCfg from "./vite.config";

export default defineConfig((env) => {
  return {
    ...viteCfg(env),

    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },

    test: {
      include: ["**/__tests__/*.test.{js,mjs,cjs,ts,mts,cts}"],
      setupFiles: ["src/test/setup"],
      environment: "jsdom",
      testTimeout: 30000,
      reporters: ["default", "junit"],

      coverage: {
        provider: "v8",
        all: true,
        reportsDirectory: "test-results",
        reporter: ["lcov"],
        exclude: [
          "test-results",
          "src/test",
          "buildtools",
          ".history",
          "src/main.ts",
          "coverage/**",
          "dist/**",
          "**/*.d.ts",
          "**/__tests__/**",
          "**/{vite,vitest}.config.{js,cjs,mjs,ts}",
          "**/.{eslint,prettier,stylelint}rc.{js,cjs,yml}",
          "**/{eslint,lint-staged,postcss}.config.{js,cjs,mjs}",
        ],
      },
      outputFile: {
        // needed for Bitbucket Pipeline
        // see https://support.atlassian.com/bitbucket-cloud/docs/test-reporting-in-pipelines/
        junit: "test-results/junit.xml",
      },
    },
  };
});
