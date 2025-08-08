import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import type {
  AiMessage,
  Config,
  ErrorMessage,
  HumanMessage,
  Timeline,
  ToolMessage,
  ViewMessage,
} from "@/types";
import { useChatStore } from "../chat";

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

const mockConfig: Config = {
  show_tool_calls_and_results: true,
};

const mockInitialAiMessage: AiMessage = {
  id: "initial-1",
  type: "ai",
  content: "Hello! How can I help you today?",
};

const mockHumanMessage: HumanMessage = {
  id: "human-1",
  type: "human",
  content: "Test user message",
};

const mockAiMessageWithToolCalls: AiMessage = {
  id: "ai-1",
  type: "ai",
  content: "Let me search for that information.",
  toolCalls: [
    {
      id: "tool-1",
      name: "search_tool",
      args: '{"query": "test"}',
    },
  ],
};

const mockToolMessage: ToolMessage = {
  id: "tool-response-1",
  type: "tool",
  content: "Search results found",
  toolCallId: "tool-1",
};

const mockAiMessageFinal: AiMessage = {
  id: "ai-2",
  type: "ai",
  content: "Based on the search results, here's your answer.",
};

const mockViewMessage: ViewMessage = {
  id: "view-1",
  type: "view",
  content: "project#workflow:node",
  name: "Chart View",
};

const mockErrorMessage: ErrorMessage = {
  id: "error-1",
  type: "error",
  content: "Something went wrong",
};

describe("chat store", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );
  });

  const initializeStore = () => {
    const store = useChatStore();
    return { store };
  };

  describe("initial state", () => {
    it("creates an empty store", () => {
      const { store } = initializeStore();

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
      const { store } = initializeStore();
      store.isLoading = true;
      store.isUsingTools = true;
      store.config = { show_tool_calls_and_results: false };

      expect(store.shouldShowToolUseIndicator).toBe(true);
    });

    it("shouldShowToolUseIndicator returns false when showing tool calls", () => {
      const { store } = initializeStore();
      store.isLoading = true;
      store.isUsingTools = true;
      store.config = { show_tool_calls_and_results: true };

      expect(store.shouldShowToolUseIndicator).toBe(false);
    });

    it("shouldShowGenericLoadingIndicator returns true when loading without tools", () => {
      const { store } = initializeStore();
      store.isLoading = true;
      store.isUsingTools = false;

      expect(store.shouldShowGenericLoadingIndicator).toBe(true);
    });

    it("shouldShowGenericLoadingIndicator returns false when using tools", () => {
      const { store } = initializeStore();
      store.isLoading = true;
      store.isUsingTools = true;

      expect(store.shouldShowGenericLoadingIndicator).toBe(false);
    });

    it("shouldShowToolCalls returns config value", () => {
      const { store } = initializeStore();
      store.config = { show_tool_calls_and_results: true };

      expect(store.shouldShowToolCalls).toBe(true);

      store.config = { show_tool_calls_and_results: false };
      expect(store.shouldShowToolCalls).toBe(false);
    });
  });

  describe("addErrorMessage", () => {
    it("adds error message to chat items", () => {
      const { store } = initializeStore();
      store.isLoading = true;
      store.isUsingTools = true;

      store.addErrorMessage("sending");

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject({
        type: "error",
        content: "There was an error while sending your message.",
      });
      expect(store.isLoading).toBe(false);
      expect(store.isUsingTools).toBe(false);
    });

    it("completes active timeline when adding error", () => {
      const { store } = initializeStore();
      const mockTimeline: Timeline = {
        id: "timeline-1",
        type: "timeline",
        label: "Using tools",
        status: "active",
        items: [],
      };
      store.activeTimeline = mockTimeline;
      store.chatItems.push(mockTimeline);

      store.addErrorMessage("processing");

      expect(store.activeTimeline).toBe(null);
      expect(mockTimeline.status).toBe("completed");
    });
  });

  describe("initialization", () => {
    it("initializes successfully with config and initial message", async () => {
      const { store } = initializeStore();
      mockJsonDataService.data
        .mockResolvedValueOnce(mockConfig)
        .mockResolvedValueOnce(mockInitialAiMessage);

      await store.init();

      expect(store.config).toEqual(mockConfig);
      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(mockInitialAiMessage);
    });

    it("initializes successfully without initial message", async () => {
      const { store } = initializeStore();
      mockJsonDataService.data
        .mockResolvedValueOnce(mockConfig)
        .mockResolvedValueOnce(null);

      await store.init();

      expect(store.config).toEqual(mockConfig);
      expect(store.chatItems).toHaveLength(0);
    });

    it("handles initialization failure", async () => {
      const { store } = initializeStore();
      // Mock JsonDataService.getInstance to fail during ensureJsonDataService
      const JsonDataService = await import("@knime/ui-extension-service");
      vi.mocked(JsonDataService.JsonDataService.getInstance).mockRejectedValue(
        new Error("Network error"),
      );

      await store.init();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject({
        type: "error",
        content: "Something went wrong. Try again later.",
      });
    });
  });

  describe("backend API methods", () => {
    beforeEach(() => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
    });

    it("getConfiguration handles success", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue(mockConfig);

      const result = await store.getConfiguration();

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "get_configuration",
      });
      expect(result).toEqual(mockConfig);
    });

    it("getConfiguration handles failure with default config", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      const result = await store.getConfiguration();

      expect(result).toEqual({ show_tool_calls_and_results: false });
    });

    it("getLastMessages handles success", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      const mockMessages = [mockAiMessageFinal];
      mockJsonDataService.data.mockResolvedValue(mockMessages);

      const result = await store.getLastMessages();

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "get_last_messages",
      });
      expect(result).toEqual(mockMessages);
    });

    it("getLastMessages handles failure", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      const error = new Error("Network error");
      mockJsonDataService.data.mockRejectedValue(error);

      await expect(store.getLastMessages()).rejects.toThrow(error);
    });

    it("checkIsProcessing handles success", async () => {
      const { store } = initializeStore();
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
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      const result = await store.checkIsProcessing();

      expect(result).toEqual({ is_processing: false });
    });

    it("postUserMessage handles success", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue({});

      await store.postUserMessage("test message");

      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "post_user_message",
        options: ["test message"],
      });
    });

    it("postUserMessage handles failure", async () => {
      const { store } = initializeStore();
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
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockResolvedValue({});
      const pollingSpy = vi
        .spyOn(store, "pollForNewMessages")
        .mockImplementation(() => Promise.resolve());

      await store.sendUserMessage("Hello!");

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject({
        type: "human",
        content: "Hello!",
      });
      expect(store.lastUserMessage).toBe("Hello!");
      expect(store.isLoading).toBe(true);
      expect(mockJsonDataService.data).toHaveBeenCalledWith({
        method: "post_user_message",
        options: ["Hello!"],
      });
      expect(pollingSpy).toHaveBeenCalled();
    });

    it("handles sending failure", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      await store.sendUserMessage("Hello!");

      expect(store.chatItems).toHaveLength(2); // user message + error message
      expect(store.chatItems[1]).toMatchObject({
        type: "error",
        content: "There was an error while sending your message.",
      });
      expect(store.isLoading).toBe(false);
    });

    it("ignores empty messages", async () => {
      const { store } = initializeStore();

      await store.sendUserMessage("");
      await store.sendUserMessage("   ");

      expect(store.chatItems).toHaveLength(0);
      expect(store.isLoading).toBe(false);
    });

    it("ignores messages when already loading", async () => {
      const { store } = initializeStore();
      store.isLoading = true;

      await store.sendUserMessage("Hello!");

      expect(store.chatItems).toHaveLength(0);
    });
  });

  describe("timeline management", () => {
    it("creates active timeline when needed", () => {
      const { store } = initializeStore();

      store.ensureActiveTimeline();

      expect(store.activeTimeline).toBeTruthy();
      expect(store.activeTimeline?.status).toBe("active");
      expect(store.activeTimeline?.label).toBe("Using tools");
      expect(store.isUsingTools).toBe(true);
      expect(store.chatItems).toHaveLength(1);
    });

    it("does not create timeline if one already exists", () => {
      const { store } = initializeStore();
      const existingTimeline: Timeline = {
        id: "existing",
        type: "timeline",
        label: "Existing",
        status: "active",
        items: [],
      };
      store.activeTimeline = existingTimeline;

      store.ensureActiveTimeline();

      expect(store.activeTimeline).toStrictEqual(existingTimeline);
      expect(store.chatItems).toHaveLength(0);
    });

    it("completes active timeline", () => {
      const { store } = initializeStore();
      const timeline: Timeline = {
        id: "timeline-1",
        type: "timeline",
        label: "Using tools",
        status: "active",
        items: [
          {
            id: "tool-1",
            type: "tool_call",
            name: "search",
            status: "completed",
          },
        ],
      };
      store.activeTimeline = timeline;
      store.isUsingTools = true;

      store.completeActiveTimeline();

      expect(store.activeTimeline).toBe(null);
      expect(store.isUsingTools).toBe(false);
      expect(timeline.status).toBe("completed");
      expect(timeline.label).toBe("Completed 1 tool call");
    });

    it("handles multiple tool calls in timeline label", () => {
      const { store } = initializeStore();
      const timeline: Timeline = {
        id: "timeline-1",
        type: "timeline",
        label: "Using tools",
        status: "active",
        items: [
          {
            id: "tool-1",
            type: "tool_call",
            name: "search",
            status: "completed",
          },
          {
            id: "tool-2",
            type: "tool_call",
            name: "analyze",
            status: "completed",
          },
        ],
      };
      store.activeTimeline = timeline;

      store.completeActiveTimeline();

      expect(timeline.label).toBe("Completed 2 tool calls");
    });
  });

  describe("adding messages to chat", () => {
    it("adds AI response with tool calls to timeline", () => {
      const { store } = initializeStore();
      store.config = { show_tool_calls_and_results: true };

      store.addAiMessageWithToolCalls(mockAiMessageWithToolCalls);

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
      const { store } = initializeStore();
      store.config = { show_tool_calls_and_results: true };
      const timeline: Timeline = {
        id: "timeline-1",
        type: "timeline",
        label: "Using tools",
        status: "active",
        items: [
          {
            id: "tool-1",
            type: "tool_call",
            name: "search",
            status: "running",
          },
        ],
      };
      store.activeTimeline = timeline;

      store.addToolMessage(mockToolMessage);

      expect(timeline.items[0]).toMatchObject({
        type: "tool_call",
        status: "completed",
        content: "Search results found",
      });
    });

    it("adds final AI response to chat items", () => {
      const { store } = initializeStore();
      const timeline: Timeline = {
        id: "timeline-1",
        type: "timeline",
        label: "Using tools",
        status: "active",
        items: [],
      };
      store.activeTimeline = timeline;
      store.isLoading = true;

      store.addAiMessage(mockAiMessageFinal);

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(mockAiMessageFinal);
      expect(store.isLoading).toBe(false);
      expect(store.activeTimeline).toBe(null);
      expect(timeline.status).toBe("completed");
    });

    it("processes mixed message responses correctly", () => {
      const { store } = initializeStore();
      store.config = { show_tool_calls_and_results: true };

      const messages = [
        mockAiMessageWithToolCalls,
        mockToolMessage,
        mockAiMessageFinal,
        mockViewMessage,
      ];

      store.addItemsToChat(messages);

      // Should have timeline + final AI response + view response
      expect(store.chatItems).toHaveLength(3);
      expect(store.chatItems[0].type).toBe("timeline");
      expect(store.chatItems[1]).toEqual(mockAiMessageFinal);
      expect(store.chatItems[2]).toEqual(mockViewMessage);
    });
  });

  describe("pollForNewMessages", () => {
    it("polls successfully and processes messages", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      store.isLoading = true;

      // Mock polling sequence: empty -> messages -> not processing
      mockJsonDataService.data
        .mockResolvedValueOnce([]) // first getLastMessages call
        .mockResolvedValueOnce([mockAiMessageFinal]) // second getLastMessages call
        .mockResolvedValueOnce({ is_processing: false }); // checkIsProcessing call

      // Mock setTimeout to resolve immediately
      vi.spyOn(global, "setTimeout").mockImplementation((fn: any) => {
        fn();
        return 1 as any;
      });

      await store.pollForNewMessages();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toEqual(mockAiMessageFinal);
      expect(store.isLoading).toBe(false);
    });

    it("handles polling errors", async () => {
      const { store } = initializeStore();
      store.jsonDataService = mockJsonDataService;
      store.isLoading = true;

      mockJsonDataService.data.mockRejectedValue(new Error("Network error"));

      await store.pollForNewMessages();

      expect(store.chatItems).toHaveLength(1);
      expect(store.chatItems[0]).toMatchObject({
        type: "error",
        content:
          "Lost connection while waiting for response. Please try sending your message again.",
      });
    });
  });

  describe("resetChat", () => {
    it("resets all state to initial values", () => {
      const { store } = initializeStore();
      // Set some state
      store.chatItems = [mockHumanMessage];
      store.config = mockConfig;
      store.isLoading = true;
      store.isUsingTools = true;
      store.lastUserMessage = "test";
      store.activeTimeline = {} as Timeline;
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
