const { preset } = require("@knime/styles/config/postcss.config.cjs"); // eslint-disable-line @typescript-eslint/no-require-imports

module.exports = {
  plugins: {
    "postcss-mixins": {},
    "postcss-preset-env": preset,
  },
};
