import { afterEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { flushPromises } from "@vue/test-utils";

import { withSetup } from "@/test/utils/withSetup";
import { useScrollToBottom } from "../useScrollToBottom";

const scrollHeight = 1000;
const clientHeight = 200;

describe("useScrollToBottom", () => {
  const createChatInterface = () => {
    const container = document.createElement("div");
    const containerRef = ref<HTMLElement | null>(container);

    document.body.appendChild(container);

    Object.defineProperty(container, "scrollHeight", { value: scrollHeight });
    Object.defineProperty(container, "clientHeight", { value: clientHeight });

    const [result, app] = withSetup(() => useScrollToBottom(containerRef));

    return { container, result, app };
  };

  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("scrolls to bottom if container children change", async () => {
    const { container } = createChatInterface();

    container.appendChild(document.createElement("div"));
    await flushPromises();

    expect(container.scrollTop).toBe(scrollHeight);
  });
});
