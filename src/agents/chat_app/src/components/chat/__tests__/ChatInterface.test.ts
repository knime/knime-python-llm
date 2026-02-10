import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import type {
  AiMessage,
  ErrorMessage,
  HumanMessage,
  Timeline,
  ToolMessage,
} from "@/types";
import { useChatStore } from "@/stores/chat";
import ChatInterface from "../ChatInterface.vue";
import {
  createAiMessage,
  createToolMessage,
  createUserMessage,
} from "@/test/factories/messages";
import { nextTick } from "process";

vi.mock("@/composables/useScrollToBottom", () => ({
  useScrollToBottom: vi.fn(),
}));

describe("ChatInterface", () => {
  beforeEach(() => {
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );
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
});
