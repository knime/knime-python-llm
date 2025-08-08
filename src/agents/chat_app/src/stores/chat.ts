import { defineStore } from "pinia";

import { JsonDataService } from "@knime/ui-extension-service";

import type {
  AiMessage,
  ChatItem,
  Config,
  ErrorMessage,
  Message,
  Timeline,
  ToolCallTimelineItem,
  ToolMessage,
} from "@/types";
import { createId } from "@/utils/utils";

const ERROR_MESSAGES = {
  init: "Something went wrong. Try again later.",
  sending: "There was an error while sending your message.",
  processing: "There was an error while processing your request.",
  connection: "Failed to connect to the service. Please try again.",
  configuration:
    "Failed to load configuration. Some features may not work properly.",
  polling:
    "Lost connection while waiting for response. Please try sending your message again.",
} as const;

export const useChatStore = defineStore("chat", {
  state: () => {
    return {
      chatItems: [] as ChatItem[],
      config: null as Config | null,
      isLoading: false,
      isUsingTools: false,
      lastUserMessage: "",
      activeTimeline: null as Timeline | null,
      jsonDataService: null as JsonDataService | null,
    };
  },
  getters: {
    shouldShowToolUseIndicator: (state) =>
      state.isLoading &&
      state.isUsingTools &&
      !state.config?.show_tool_calls_and_results,
    shouldShowGenericLoadingIndicator: (state) =>
      state.isLoading && !state.isUsingTools,
    shouldShowToolCalls: (state) => state.config?.show_tool_calls_and_results,
  },
  actions: {
    addErrorMessage(errorType: keyof typeof ERROR_MESSAGES) {
      const errorMessage: ErrorMessage = {
        id: createId(),
        type: "error",
        content: ERROR_MESSAGES[errorType],
      };
      this.chatItems.push(errorMessage);

      this.isLoading = false;
      this.isUsingTools = false;
      this.completeActiveTimeline();
    },
    async init() {
      try {
        await this.ensureJsonDataService();

        const [config, initialAiMessage] = await Promise.all([
          this.getConfiguration(),
          this.getInitialMessage(),
        ]);

        this.config = config;
        if (initialAiMessage) {
          this.addItemsToChat([initialAiMessage]);
        }
      } catch (error) {
        consola.error("Chat Store: Error during initialization:", error);
        this.addErrorMessage("init");
      }
    },
    ensureActiveTimeline() {
      if (this.activeTimeline) {
        return;
      }

      const newTimeline: Timeline = {
        id: `timeline-${createId()}`,
        label: "Using tools",
        items: [],
        status: "active",
        type: "timeline",
      };
      this.chatItems.push(newTimeline);
      this.activeTimeline = newTimeline;
      this.isUsingTools = true;
    },
    completeActiveTimeline() {
      if (!this.activeTimeline) {
        return;
      }

      this.isUsingTools = false;

      const toolCallCount = this.activeTimeline.items.filter(
        (item) => item.type === "tool_call",
      ).length;
      this.activeTimeline.label = `Completed ${toolCallCount} tool call${toolCallCount === 1 ? "" : "s"}`;
      this.activeTimeline.status = "completed";
      this.activeTimeline = null;
    },
    addAiReasoningToTimeline(msg: AiMessage) {
      if (msg.content?.trim()) {
        this.activeTimeline?.items.push({
          id: msg.id,
          type: "reasoning",
          content: msg.content,
        });
      }
    },
    addToolCallsToTimeline(msg: AiMessage) {
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          this.activeTimeline?.items.push({
            id: tc.id,
            type: "tool_call",
            name: tc.name,
            args: tc.args,
            status: "running",
          });
        }
      }
    },
    addAiMessageWithToolCalls(msg: AiMessage) {
      if (!this.shouldShowToolCalls) {
        this.isUsingTools = true;
        return;
      }
      // this either kicks off the agentic loop (create a new timeline)
      // or contributes to the active timeline
      this.ensureActiveTimeline();

      // extract and add reasoning to timeline
      this.addAiReasoningToTimeline(msg);

      // extract and add tool calls to timeline
      this.addToolCallsToTimeline(msg);
    },
    addToolMessage(msg: ToolMessage) {
      if (!this.activeTimeline) {
        return;
      }

      if (!this.shouldShowToolCalls) {
        return;
      }

      const index = this.activeTimeline.items.findIndex(
        (item) => item.id === msg.toolCallId,
      );

      if (index === -1) {
        consola.error(
          `Could not find tool call with ID ${msg.toolCallId} in the active timeline.`,
        );
      } else {
        const originalItem = this.activeTimeline.items[
          index
        ] as ToolCallTimelineItem;

        const content = msg.content ?? "";
        const status = content.toLowerCase().startsWith("error")
          ? "failed"
          : "completed";

        const updatedItem: ToolCallTimelineItem = {
          ...originalItem,
          content,
          status,
        };

        this.activeTimeline.items[index] = updatedItem;
      }
    },
    addAiMessage(msg: AiMessage) {
      // agent either finished the agentic loop, or simply replied without tool use
      this.completeActiveTimeline();
      this.isLoading = false;

      this.chatItems.push(msg);
    },
    addItemsToChat(messages: Message[]) {
      if (!messages.length) {
        return;
      }

      for (const msg of messages) {
        if (msg.type === "ai" && msg.toolCalls?.length) {
          this.addAiMessageWithToolCalls(msg);
          continue;
        }

        if (msg.type === "tool") {
          this.addToolMessage(msg);
          continue;
        }

        if (msg.type === "ai" && !msg.toolCalls?.length) {
          this.addAiMessage(msg);
          continue;
        }

        // otherwise add message directly to the conversation
        this.chatItems.push(msg);
      }
    },

    // ==== BACKEND API ACTIONS ====
    async ensureJsonDataService() {
      if (this.jsonDataService) {
        return;
      }

      try {
        this.jsonDataService = await JsonDataService.getInstance();
      } catch (error) {
        consola.error(
          "Chat Store: Failed to get JsonDataService instance:",
          error,
        );
        throw new Error("Failed to connect to the service");
      }
    },
    async getConfiguration() {
      try {
        const response = await this.jsonDataService?.data({
          method: "get_configuration",
        });
        return response;
      } catch (error) {
        consola.error("Chat Store: Failed to get configuration:", error);
        return { show_tool_calls_and_results: false };
      }
    },
    async getInitialMessage() {
      try {
        const response = await this.jsonDataService?.data({
          method: "get_initial_message",
        });
        return response;
      } catch (error) {
        consola.error("Chat Store: Failed to get initial message:", error);
        return null;
      }
    },
    async getLastMessages() {
      try {
        const response = await this.jsonDataService?.data({
          method: "get_last_messages",
        });
        return response || [];
      } catch (error) {
        consola.error("Chat Store: Failed to get last messages:", error);
        throw error;
      }
    },
    async checkIsProcessing() {
      try {
        const response = await this.jsonDataService?.data({
          method: "is_processing",
        });
        return response;
      } catch (error) {
        consola.error("Chat Store: Failed to check processing status:", error);
        return { is_processing: false };
      }
    },
    async postUserMessage(msg: string) {
      try {
        await this.jsonDataService?.data({
          method: "post_user_message",
          options: [msg],
        });
      } catch (error) {
        consola.error("Chat Store: Failed to post user message:", error);
        throw error;
      }
    },

    // ==== HIGHER LEVEL CHAT CONTROLS ====
    async sendUserMessage(msg: string) {
      if (!msg.trim() || this.isLoading) {
        return;
      }

      this.isLoading = true;

      // 1. render user message in the chat
      this.addItemsToChat([{ id: createId(), content: msg, type: "human" }]);

      // 2. store for arrow up recall
      this.lastUserMessage = msg;

      try {
        // 3. send to backend and start polling
        await this.postUserMessage(msg);
        this.pollForNewMessages();
      } catch (error) {
        consola.error("Chat Store: Error sending user message:", error);
        this.addErrorMessage("sending");
      }
    },
    async pollForNewMessages() {
      let consecutiveEmptyPolls = 0;
      const maxEmptyPolls = 10;

      try {
        while (this.isLoading) {
          try {
            const msgs = await this.getLastMessages();

            if (msgs.length > 0) {
              this.addItemsToChat(msgs);
              consecutiveEmptyPolls = 0;
            } else {
              consecutiveEmptyPolls++;

              if (consecutiveEmptyPolls >= maxEmptyPolls) {
                const response = await this.checkIsProcessing();

                if (!response?.is_processing) {
                  this.isLoading = false;
                  this.isUsingTools = false;
                  break;
                }

                // still processing, give agent and tools more time
                consecutiveEmptyPolls = 0;
              }
            }
          } catch (error) {
            consola.error("Chat Store: Error during message polling:", error);
            this.addErrorMessage("polling");
            break;
          }
        }
      } catch (error) {
        consola.error("Chat Store: Fatal error in polling loop:", error);
        this.addErrorMessage("processing");
      }
    },
    resetChat() {
      this.chatItems = [];
      this.config = null;
      this.isLoading = false;
      this.isUsingTools = false;
      this.lastUserMessage = "";
      this.activeTimeline = null;
      this.jsonDataService = null;
    },
  },
});
