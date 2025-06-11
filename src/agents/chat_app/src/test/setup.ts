import { vi } from "vitest";

import { setupLogger } from "@/plugins/logger";

setupLogger();

vi.mock("@knime/components", async (importActual) => {
  const actual = await importActual();

  return {
    // @ts-expect-error
    ...actual,
    setupHints: vi.fn(),
    useHint: () => ({ createHint: vi.fn(), isCompleted: vi.fn(() => true) }),
  };
});

vi.mock("@/plugins/toasts", () => {
  const show = vi.fn();
  const remove = vi.fn();
  const removeBy = vi.fn();

  return {
    getToastsProvider: () => {
      return { show, remove, removeBy };
    },
  };
});
