import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setupLogger } from "../logger";

describe("logger", () => {
  let originalConsola;

  beforeEach(() => {
    originalConsola = window.consola;
    delete window.consola;
  });

  afterEach(() => {
    window.consola = originalConsola;
  });

  it("defines a logger", () => {
    setupLogger();
    [
      "debug",
      "error",
      "fatal",
      "info",
      "log",
      "ready",
      "silent",
      "start",
      "success",
      "trace",
      "verbose",
      "warn",
    ].forEach((method) => {
      expect(window.consola[method]).toBeInstanceOf(Function);
    });
  });
});
