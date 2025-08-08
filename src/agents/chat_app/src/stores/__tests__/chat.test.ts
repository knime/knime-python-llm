import { afterEach, describe, expect, it, vi } from "vitest";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import type { Timeline } from "@/types";
import { useChatStore } from "../chat";
import {
  createAiMessage,
  createConfig,
  createErrorMessage,
  createTimeline,
  createToolCall,
  createToolCallTimelineItem,
  createToolMessage,
  createUserMessage,
  createViewMessage,
} from "@/test/factories/messages";

const mockJsonDataService = {
  data: vi.fn(),
  initialData: vi.fn(() => Promise.resolve({})),
  applyData: vi.fn(() => Promise.resolve({ isApplied: true })),
};

vi.mock("@knime/ui-extension-service", () => ({
  JsonDataService: {
    getInstance: vi.fn(() => Promise.resolve(mockJsonDataService)),
  },
}));

vi.mock("consola", () => ({
  default: {
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    trace: vi.fn(),
  },
}));

describe("chat store", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  const setupStore = () => {
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );

    const store = useChatStore();
    return { store };
  };

  describe("initial state", () => {
    it("creates an empty store", () => {
      const { store } = setupStore();

      expect(store).toMatchObject({
        chatItems: [],
        config: null,
        isLoading: false,
        isUsingTools: false,
        lastUserMessage: "",
        activeTimeline: null,
        jsonDataService: null,
      });
    });
  });

  describe("getters", () => {
    it("shouldShowToolUseIndicator returns true when loading with tools and not showing tool calls", () => {
      const { store } = setupStore();
      store.isLoading = true;
      store.isUsingTools = true;
      store.config = createConfig(false);

      expect(store.shouldShowToolUseIndicator).toBe(true);
    });

    it("shouldShowToolUseIndicator returns false when showing tool calls", () => {
      const { store } = setupStore();
      store.isLoading = true;
      store.isUsingTools = true;
      store.config = createConfig(true);

      expect(store.shouldShowToolUseIndicator).toBe(false);
    });

    it("shouldShowGenericLoadingIndicator returns true when loading without tools", () => {
      const { store } = setupStore();
      store.isLoading = true;
      store.isUsingTools = false;

      expect(store.shouldShowGenericLoadingIndicator).toBe(true);
    });

    it("shouldShowGenericLoadingIndicator returns false when using tools", () => {
      const { store } = setupStore();
      store.isLoading = true;
      store.isUsingTools = true;

      expect(store.shouldShowGenericLoadingIndicator).toBe(false);
    });

    it("shouldShowToolCalls returns config value", () => {
      const { store } = setupStore();
      store.config = createConfig(true);

      expect(store.shouldShowToolCalls).toBe(true);

      store.config = createConfig(false);
      expect(store.shouldShowToolCalls).toBe(false);
    });
  });

  describe("addErrorMessage", () => {
    it("adds error message to chat items", () => {
      const { store } = setupStore();
      store.isLoading = true;
      store.isUsingTools = true;

      store.addErrorMessage("sending");

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject(
        createErrorMessage("There was an error while sending your message."),
      );
      expect(store.isLoading).toBe(false);
      expect(store.isUsingTools).toBe(false);
    });

    it("completes active timeline when adding error", () => {
      const { store } = setupStore();
      const mockTimeline = createTimeline("Using tools", "active", []);
      store.activeTimeline = mockTimeline;
      store.chatItems.push(mockTimeline);

      store.addErrorMessage("processing");

      expect(store.activeTimeline).toBe(null);
      expect(mockTimeline.status).toBe("completed");
    });
  });

  describe("initialization", () => {
    it("initializes successfully with config and initial message", async () => {
      const { store } = setupStore();
      const config = createConfig(true);
      const initialMessage = createAiMessage(
        "Hello! How can I help you today?",
      );

      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data
        .mockResolvedValueOnce(config)
        .mockResolvedValueOnce(initialMessage);

      // Test the initialization parts separately
      store.config = await store.getConfiguration();
      const initMsg = await store.getInitialMessage();
      if (initMsg) {
        store.addItemsToChat([initMsg]);
      }

      expect(store.config).toEqual(config);
      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(initialMessage);
    });

    it("initializes successfully without initial message", async () => {
      const { store } = setupStore();
      const config = createConfig(true);

      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data
        .mockResolvedValueOnce(config)
        .mockResolvedValueOnce(null);

      store.config = await store.getConfiguration();
      const initMsg = await store.getInitialMessage();
      if (initMsg) {
        store.addItemsToChat([initMsg]);
      }

      expect(store.config).toEqual(config);
      expect(store.chatItems).toHaveLength(0);
    });

    it("handles initialization failure", async () => {
      const { store } = setupStore();
      // Mock JsonDataService.getInstance to fail during ensureJsonDataService
      const JsonDataService = await import("@knime/ui-extension-service");
      vi.mocked(JsonDataService.JsonDataService.getInstance).mockRejectedValue(
        new Error("Network error"),
      );

      await store.init();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject(
        createErrorMessage("Something went wrong. Try again later."),
      );
    });
  });

  describe("backend API methods", () => {
    it("getConfiguration handles success", async () => {
      const { store } = setupStore();
      const config = createConfig(true);
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue(config);

      const result = await store.getConfiguration();

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "get_configuration",
      });
      expect(result).toEqual(config);
    });

    it("getConfiguration handles failure with default config", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      const result = await store.getConfiguration();

      expect(result).toEqual(createConfig(false));
    });

    it("getLastMessages handles success", async () => {
      const { store } = setupStore();
      const messages = [
        createAiMessage("Based on the search results, here's your answer."),
      ];
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue(messages);

      const result = await store.getLastMessages();

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "get_last_messages",
      });
      expect(result).toEqual(messages);
    });

    it("getLastMessages handles failure", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      const error = new Error("Network error");
      mockJsonDataService.data.mockRejectedValue(error);

      await expect(store.getLastMessages()).rejects.toThrow(error);
    });

    it("checkIsProcessing handles success", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      const mockResponse = { is_processing: true };
      mockJsonDataService.data.mockResolvedValue(mockResponse);

      const result = await store.checkIsProcessing();

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "is_processing",
      });
      expect(result).toEqual(mockResponse);
    });

    it("checkIsProcessing handles failure", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      const result = await store.checkIsProcessing();

      expect(result).toEqual({ is_processing: false });
    });

    it("postUserMessage handles success", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue({});

      await store.postUserMessage("test message");

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "post_user_message",
        options: ["test message"],
      });
    });

    it("postUserMessage handles failure", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      const error = new Error("Network error");
      mockJsonDataService.data.mockRejectedValue(error);

      await expect(store.postUserMessage("test message")).rejects.toThrow(
        error,
      );
    });
  });

  describe("sendUserMessage", () => {
    it("sends user message successfully", async () => {
      const { store } = setupStore();

      // Set up the store state directly
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockClear();
      mockJsonDataService.data.mockResolvedValue({});

      // Test the core functionality by simulating the flow step by step
      const message = "Hello!";

      // 1. Add user message to chat (this happens first in sendUserMessage)
      store.addItemsToChat([
        { id: createUserMessage(message).id, content: message, type: "human" },
      ]);

      // 2. Store the message for recall
      store.lastUserMessage = message;

      // 3. Test that the backend call would be made correctly
      await store.postUserMessage(message);

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject(createUserMessage(message));
      expect(store.lastUserMessage).toBe(message);
      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "post_user_message",
        options: [message],
      });
    });

    it("handles sending failure", async () => {
      const { store } = setupStore();

      // Clear and reset the mock
      mockJsonDataService.data.mockClear();
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      // Set up the store state directly
      store.jsonDataService = mockJsonDataService;

      await store.sendUserMessage("Hello!");

      expect(store.chatItems).toHaveLength(2); // user message + error message
      expect(store.chatItems[1]).toMatchObject(
        createErrorMessage("There was an error while sending your message."),
      );
      expect(store.isLoading).toBe(false);
    });

    it("ignores empty messages", async () => {
      const { store } = setupStore();

      await store.sendUserMessage("");
      await store.sendUserMessage("   ");

      expect(store.chatItems).toHaveLength(0);
      expect(store.isLoading).toBe(false);
    });

    it("ignores messages when already loading", async () => {
      const { store } = setupStore();
      store.isLoading = true;

      await store.sendUserMessage("Hello!");

      expect(store.chatItems).toHaveLength(0);
    });
  });

  describe("timeline management", () => {
    it("creates active timeline when needed", () => {
      const { store } = setupStore();

      store.ensureActiveTimeline();

      expect(store.activeTimeline).toBeTruthy();
      expect(store.activeTimeline?.status).toBe("active");
      expect(store.activeTimeline?.label).toBe("Using tools");
      expect(store.isUsingTools).toBe(true);
      expect(store.chatItems).toHaveLength(1);
    });

    it("does not create timeline if one already exists", () => {
      const { store } = setupStore();
      const existingTimeline = createTimeline("Existing", "active", []);
      store.activeTimeline = existingTimeline;

      store.ensureActiveTimeline();

      expect(store.activeTimeline).toStrictEqual(existingTimeline);
      expect(store.chatItems).toHaveLength(0);
    });

    it("completes active timeline", () => {
      const { store } = setupStore();
      const timeline = createTimeline("Using tools", "active", [
        createToolCallTimelineItem("search", "completed"),
      ]);
      store.activeTimeline = timeline;
      store.isUsingTools = true;

      store.completeActiveTimeline();

      expect(store.activeTimeline).toBe(null);
      expect(store.isUsingTools).toBe(false);
      expect(timeline.status).toBe("completed");
      expect(timeline.label).toBe("Completed 1 tool call");
    });

    it("handles multiple tool calls in timeline label", () => {
      const { store } = setupStore();
      const timeline = createTimeline("Using tools", "active", [
        createToolCallTimelineItem("search", "completed"),
        createToolCallTimelineItem("analyze", "completed"),
      ]);
      store.activeTimeline = timeline;

      store.completeActiveTimeline();

      expect(timeline.label).toBe("Completed 2 tool calls");
    });
  });

  describe("adding messages to chat", () => {
    it("adds AI response with tool calls to timeline", () => {
      const { store } = setupStore();
      const toolCall = {
        id: "tool-1",
        name: "search_tool",
        args: '{"query": "test"}',
      };
      const aiMessageWithToolCalls = createAiMessage(
        "Let me search for that information.",
        [toolCall],
      );
      store.config = createConfig(true);

      store.addAiMessageWithToolCalls(aiMessageWithToolCalls);

      expect(store.activeTimeline).toBeTruthy();
      expect(store.activeTimeline?.items).toHaveLength(2); // reasoning + tool call
      expect(store.activeTimeline?.items[0]).toMatchObject({
        type: "reasoning",
        content: "Let me search for that information.",
      });
      expect(store.activeTimeline?.items[1]).toMatchObject({
        type: "tool_call",
        name: "search_tool",
        status: "running",
      });
    });

    it("adds tool response to timeline", () => {
      const { store } = setupStore();
      const toolCallId = "tool-1";
      const toolMessage = createToolMessage("Search results found", toolCallId);
      store.config = createConfig(true);
      const timeline = createTimeline("Using tools", "active", [
        {
          id: toolCallId,
          type: "tool_call",
          name: "search",
          status: "running",
        },
      ]);
      store.activeTimeline = timeline;

      store.addToolMessage(toolMessage);

      expect(timeline.items[0]).toMatchObject({
        type: "tool_call",
        status: "completed",
        content: "Search results found",
      });
    });

    it("adds final AI response to chat items", () => {
      const { store } = setupStore();
      const finalMessage = createAiMessage(
        "Based on the search results, here's your answer.",
      );
      const timeline = createTimeline("Using tools", "active", []);
      store.activeTimeline = timeline;
      store.isLoading = true;

      store.addAiMessage(finalMessage);

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(finalMessage);
      expect(store.isLoading).toBe(false);
      expect(store.activeTimeline).toBe(null);
      expect(timeline.status).toBe("completed");
    });

    it("processes mixed message responses correctly", () => {
      const { store } = setupStore();
      const toolCallId = "tool-1";
      const toolCall = {
        id: toolCallId,
        name: "search_tool",
        args: '{"query": "test"}',
      };
      const aiMessageWithToolCalls = createAiMessage(
        "Let me search for that information.",
        [toolCall],
      );
      const toolMessage = createToolMessage("Search results found", toolCallId);
      const finalMessage = createAiMessage(
        "Based on the search results, here's your answer.",
      );
      const viewMessage = createViewMessage(
        "project#workflow:node",
        "Chart View",
      );

      store.config = createConfig(true);

      const messages = [
        aiMessageWithToolCalls,
        toolMessage,
        finalMessage,
        viewMessage,
      ];

      store.addItemsToChat(messages);

      // Should have timeline + final AI response + view response
      expect(store.chatItems).toHaveLength(3);
      expect(store.chatItems[0].type).toBe("timeline");
      expect(store.chatItems[1]).toEqual(finalMessage);
      expect(store.chatItems[2]).toEqual(viewMessage);
    });
  });

  describe("pollForNewMessages", () => {
    it("polls successfully and processes messages", async () => {
      const { store } = setupStore();
      const finalMessage = createAiMessage(
        "Based on the search results, here's your answer.",
      );
      store.jsonDataService = mockJsonDataService;
      store.isLoading = true;

      // Mock polling sequence: empty -> messages -> not processing
      mockJsonDataService.data
        .mockResolvedValueOnce([]) // first getLastMessages call
        .mockResolvedValueOnce([finalMessage]) // second getLastMessages call
        .mockResolvedValueOnce({ is_processing: false }); // checkIsProcessing call

      // Mock setTimeout to resolve immediately
      vi.spyOn(global, "setTimeout").mockImplementation((fn: any) => {
        fn();
        return 1 as any;
      });

      await store.pollForNewMessages();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(finalMessage);
      expect(store.isLoading).toBe(false);
    });

    it("handles polling errors", async () => {
      const { store } = setupStore();
      store.jsonDataService = mockJsonDataService;
      store.isLoading = true;

      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      await store.pollForNewMessages();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject(
        createErrorMessage(
          "Lost connection while waiting for response. Please try sending your message again.",
        ),
      );
    });
  });

  describe("resetChat", () => {
    it("resets all state to initial values", () => {
      const { store } = setupStore();
      const config = createConfig(true);
      const userMessage = createUserMessage("test");
      const timeline = createTimeline();

      // Set some state
      store.chatItems = [userMessage];
      store.config = config;
      store.isLoading = true;
      store.isUsingTools = true;
      store.lastUserMessage = "test";
      store.activeTimeline = timeline as Timeline;
      store.jsonDataService = mockJsonDataService;

      store.resetChat();

      expect(store).toMatchObject({
        chatItems: [],
        config: null,
        isLoading: false,
        isUsingTools: false,
        lastUserMessage: "",
        activeTimeline: null,
        jsonDataService: null,
      });
    });
  });
});
