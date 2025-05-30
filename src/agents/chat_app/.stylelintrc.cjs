module.exports = {
  extends: ["@knime/eslint-config/stylelint/vue"],
  rules: {
    "media-feature-range-notation": null,
    "at-rule-no-unknown": [
      true,
      {
        ignoreAtRules: ["define-mixin", "mixin"],
      },
    ],
  },
};
