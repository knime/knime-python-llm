import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import { useChatStore } from "@/stores/chat";
import ExpandableTimeline from "../ExpandableTimeline.vue";

describe("ExpandableTimeline", () => {
  beforeEach(() => {
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );
  });

  const createWrapper = (props = {}) => {
    return mount(ExpandableTimeline, {
      props: {
        id: "1",
        label: "Executing tools",
        status: "completed",
        type: "timeline",
        items: [],
        ...props,
      },
      global: {
        stubs: {
          WrenchIcon: true,
          ArrowNextIcon: true,
          AnimatedEllipsis: true,
          TimelineItem: true,
        },
      },
    });
  };

  it("renders the label from props when not interrupted", () => {
    const wrapper = createWrapper({
      label: "Original Label",
      status: "active",
    });
    const chatStore = useChatStore();
    chatStore.isInterrupted = false;

    expect(wrapper.text()).toContain("Original Label");
  });

  it("renders 'Cancelling' when active and interrupted", async () => {
    const wrapper = createWrapper({
      label: "Original Label",
      status: "active",
    });
    const chatStore = useChatStore();

    chatStore.isInterrupted = true;
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain("Cancelling");
    expect(wrapper.text()).not.toContain("Original Label");
  });

  it("renders the original label when not active even if interrupted", async () => {
    const wrapper = createWrapper({
      label: "Original Label",
      status: "completed",
    });
    const chatStore = useChatStore();

    chatStore.isInterrupted = true;
    await wrapper.vm.$nextTick();

    expect(wrapper.text()).toContain("Original Label");
    expect(wrapper.text()).not.toContain("Cancelling");
  });
});
