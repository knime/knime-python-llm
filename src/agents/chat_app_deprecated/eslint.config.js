import globals from "globals";

import knimeVitestConfig from "@knime/eslint-config/vitest.js";
import knimeVue3TSConfig from "@knime/eslint-config/vue3-typescript.js";

export default [
  ...knimeVue3TSConfig,
  ...knimeVitestConfig,
  {
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.browser,
        consola: true,
      },
    },
    settings: {
      "import-x/resolver": {
        "eslint-import-resolver-custom-alias": {
          alias: {
            "@": "./src",
          },
        },
      },
    },
  },
];
