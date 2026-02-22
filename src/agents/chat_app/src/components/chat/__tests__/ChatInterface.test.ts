import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import { useChatStore } from "@/stores/chat";
import {
  createAiMessage,
  createToolMessage,
  createUserMessage,
} from "@/test/factories/messages";
import type {
  AiMessage,
  ErrorMessage,
  HumanMessage,
  Timeline,
  ToolMessage,
} from "@/types";
import ChatInterface from "../ChatInterface.vue";

vi.mock("@/composables/useScrollToBottom", () => ({
  useScrollToBottom: vi.fn(),
}));

describe("ChatInterface", () => {
  const originalHash = window.location.hash;

  beforeEach(() => {
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );
    window.location.hash = "";
  });

  afterEach(() => {
    window.location.hash = originalHash;
    vi.restoreAllMocks();
  });

  const createWrapper = () => {
    return mount(ChatInterface, {
      global: {
        stubs: {
          AiMessage: { template: "<div class='ai-message'>AI Message</div>" },
          HumanMessage: {
            template: "<div class='human-message'>Human Message</div>",
          },
          ErrorMessage: {
            template: "<div class='error-message'>Error Message</div>",
          },
          NodeViewMessage: {
            template: "<div class='view-message'>View Message</div>",
          },
          ExpandableTimeline: {
            template: "<div class='timeline'>Timeline</div>",
          },
          StatusIndicator: {
            props: ["label"],
            template: "<div class='status-indicator'>{{ label }}</div>",
          },
          MessageInput: {
            template: "<div class='message-input'>Message Input</div>",
          },
        },
      },
    });
  };

  const createNavigationWrapper = () => {
    return mount(ChatInterface, {
      attachTo: document.body,
      global: {
        stubs: {
          AiMessage: {
            props: ["id", "content"],
            template: "<div :id='id' class='ai-message' v-html='content || \"AI Message\"' />",
          },
          HumanMessage: {
            props: ["id", "content"],
            template:
              "<div :id='id' class='human-message'>{{ content || 'Human Message' }}</div>",
          },
          ErrorMessage: {
            props: ["id"],
            template: "<div :id='id' class='error-message'>Error Message</div>",
          },
          NodeViewMessage: {
            props: ["id"],
            template: "<div :id='id' class='view-message'>View Message</div>",
          },
          ExpandableTimeline: {
            template: "<div class='timeline'>Timeline</div>",
          },
          StatusIndicator: {
            props: ["label"],
            template: "<div class='status-indicator'>{{ label }}</div>",
          },
          MessageInput: {
            template: "<div class='message-input'>Message Input</div>",
          },
        },
      },
    });
  };

  it("renders the basic chat interface structure", () => {
    const wrapper = createWrapper();

    expect(wrapper.find(".chat-interface").exists()).toBe(true);
    expect(wrapper.find(".scrollable-container").exists()).toBe(true);
    expect(wrapper.find(".message-list").exists()).toBe(true);
    expect(wrapper.find(".message-input").exists()).toBe(true);
  });

  it("renders chat items from the store", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    const humanMessage: HumanMessage = {
      id: "1",
      type: "human",
      content: "Hello",
    };

    const aiMessage: AiMessage = {
      id: "2",
      type: "ai",
      content: "Hi there!",
    };

    chatStore.chatItems = [humanMessage, aiMessage];
    await wrapper.vm.$nextTick();

    expect(wrapper.find(".human-message").exists()).toBe(true);
    expect(wrapper.find(".ai-message").exists()).toBe(true);
  });

  it("renders different message types correctly", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    const errorMessage: ErrorMessage = {
      id: "1",
      type: "error",
      content: "Something went wrong",
    };

    const timeline: Timeline = {
      id: "2",
      type: "timeline",
      label: "Using tools",
      status: "active",
      items: [],
    };

    chatStore.chatItems = [errorMessage, timeline];
    await wrapper.vm.$nextTick();

    expect(wrapper.find(".error-message").exists()).toBe(true);
    expect(wrapper.find(".timeline").exists()).toBe(true);
  });

  it("shows 'Using tools' indicator when store indicates tools are being used", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    // Mock the getter to return true
    chatStore.isLoading = true;
    chatStore.lastMessage = {
      type: "tool",
      content: "",
      toolCallId: "123",
      id: "tool1",
    };
    chatStore.config = {
      show_tool_calls_and_results: false,
      reexecution_trigger: "NONE",
    };
    await wrapper.vm.$nextTick();

    const statusIndicator = wrapper.find(".status-indicator");
    expect(statusIndicator.exists()).toBe(true);
    expect(statusIndicator.text()).toBe("Using tools");
  });

  it("shows 'Cancelling' in status indicator when interrupted", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.isLoading = true;
    chatStore.lastMessage = {
      type: "tool",
      content: "",
      toolCallId: "123",
      id: "tool1",
    };
    chatStore.config = {
      show_tool_calls_and_results: false,
      reexecution_trigger: "NONE",
    };
    chatStore.isInterrupted = true;
    await wrapper.vm.$nextTick();

    const statusIndicator = wrapper.find(".status-indicator");
    expect(statusIndicator.exists()).toBe(true);
    expect(statusIndicator.text()).toBe("Cancelling");
  });

  it("does not show status indicator when store indicates tools are not being used", () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.isLoading = false;
    chatStore.lastMessage = createAiMessage("AI response");

    expect(wrapper.find(".status-indicator").exists()).toBe(false);
  });

  it("shows generic loading indicator when store indicates generic loading", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.isLoading = true;
    chatStore.lastMessage = createUserMessage("User message");
    await wrapper.vm.$nextTick();

    expect(wrapper.find(".skeleton-item").exists()).toBe(true);
  });

  it("does not show generic loading indicator when not loading", () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.isLoading = false;
    chatStore.lastMessage = createAiMessage("AI response");

    expect(wrapper.find(".skeleton-item").exists()).toBe(false);
  });

  it("renders empty chat interface when no chat items exist", () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.chatItems = [];
    chatStore.isLoading = false;
    chatStore.lastMessage = createAiMessage("AI response");

    const messageList = wrapper.find(".message-list");
    expect(messageList.exists()).toBe(true);

    // Should only contain MessageInput, no messages or indicators
    expect(wrapper.find(".ai-message").exists()).toBe(false);
    expect(wrapper.find(".human-message").exists()).toBe(false);
    expect(wrapper.find(".status-indicator").exists()).toBe(false);
    expect(wrapper.find(".skeleton-item").exists()).toBe(false);
  });

  it("handles mixed chat states correctly", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    const humanMessage: HumanMessage = {
      id: "1",
      type: "human",
      content: "Hello",
    };

    const toolMessage: ToolMessage = createToolMessage(
      "Tool response",
      "tool1",
    );

    chatStore.chatItems = [humanMessage];
    chatStore.lastMessage = toolMessage;
    chatStore.isLoading = true;
    chatStore.config = {
      show_tool_calls_and_results: false,
      reexecution_trigger: "NONE",
    };
    await wrapper.vm.$nextTick();

    // Should show both the message and the status indicator
    expect(wrapper.find(".human-message").exists()).toBe(true);
    expect(wrapper.find(".status-indicator").exists()).toBe(true);
    expect(wrapper.find(".skeleton-item").exists()).toBe(false);
  });

  it("always includes MessageInput component", () => {
    const wrapper = createWrapper();

    expect(wrapper.find(".message-input").exists()).toBe(true);
  });

  it("finishes loading when the last message is an AI message without tool calls", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();

    chatStore.isLoading = true;
    const humanMessage: HumanMessage = createUserMessage("Hello");
    chatStore.addMessages([humanMessage], false);
    await wrapper.vm.$nextTick();
    expect(wrapper.find(".skeleton-item").exists()).toBe(true);

    const aiMessage: AiMessage = createAiMessage("AI response");
    chatStore.addMessages([aiMessage], false);
    await wrapper.vm.$nextTick();
    expect(chatStore.isLoading).toBe(false);
    expect(wrapper.find(".skeleton-item").exists()).toBe(false);
  });

  it("navigates to a user message when clicking a hash link", async () => {
    const wrapper = createNavigationWrapper();
    const chatStore = useChatStore();

    const humanMessage: HumanMessage = {
      id: "msg-0002",
      type: "human",
      content: "Hello",
    };

    chatStore.chatItems = [humanMessage];
    await wrapper.vm.$nextTick();

    const target = wrapper.find("#msg-0002");
    expect(target.exists()).toBe(true);

    const scrollIntoViewMock = vi.fn();
    (target.element as HTMLElement).scrollIntoView = scrollIntoViewMock;

    const link = document.createElement("a");
    link.setAttribute("href", "#msg-0002");
    target.element.appendChild(link);

    link.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await wrapper.vm.$nextTick();

    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "start",
    });
    expect(window.location.hash).toBe("#msg-0002");
    expect(target.classes()).toContain("link-target-highlight");

    wrapper.unmount();
  });

  it("shows a hover preview for hash links to previous messages", async () => {
    const wrapper = createNavigationWrapper();
    const chatStore = useChatStore();

    chatStore.chatItems = [
      {
        id: "msg-0001",
        type: "ai",
        content: '<a href="#msg-0002">See previous</a>',
      } as AiMessage,
      {
        id: "msg-0002",
        type: "human",
        content: "This is the referenced user message content.",
      } as HumanMessage,
    ];
    await wrapper.vm.$nextTick();

    const link = wrapper.find("#msg-0001 a");
    expect(link.exists()).toBe(true);

    link.element.dispatchEvent(
      new MouseEvent("mouseover", {
        bubbles: true,
        clientX: 120,
        clientY: 80,
      }),
    );
    await wrapper.vm.$nextTick();

    const preview = wrapper.find('[data-testid="reference-preview"]');
    expect(preview.exists()).toBe(true);
    expect(preview.text()).toContain("msg-0002");
    expect(preview.text()).toContain("referenced user message content");

    link.element.dispatchEvent(new MouseEvent("mouseout", { bubbles: true }));
    await wrapper.vm.$nextTick();
    expect(wrapper.find('[data-testid="reference-preview"]').exists()).toBe(
      false,
    );

    wrapper.unmount();
  });

  it("shows backlinks for selected message and navigates to backlink source", async () => {
    const wrapper = createNavigationWrapper();
    const chatStore = useChatStore();

    chatStore.chatItems = [
      {
        id: "msg-0001",
        type: "ai",
        content:
          'Reasoning with a reference to <a href="#msg-0002">the user message</a>.',
      } as AiMessage,
      {
        id: "msg-0002",
        type: "human",
        content: "User question content",
      } as HumanMessage,
    ];
    await wrapper.vm.$nextTick();

    const selectedTarget = wrapper.find("#msg-0002");
    expect(selectedTarget.exists()).toBe(true);
    selectedTarget.element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await wrapper.vm.$nextTick();

    const panel = wrapper.find('[data-testid="backlink-panel"]');
    expect(panel.exists()).toBe(true);
    expect(panel.text()).toContain("Backlinks to msg-0002");
    expect(panel.text()).toContain("msg-0001");

    const source = wrapper.find("#msg-0001");
    const scrollIntoViewMock = vi.fn();
    (source.element as HTMLElement).scrollIntoView = scrollIntoViewMock;

    await wrapper.find(".backlink-item").trigger("click");

    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "start",
    });
    expect(window.location.hash).toBe("#msg-0001");
    expect(source.classes()).toContain("link-target-highlight");

    wrapper.unmount();
  });
});
