import { defineStore } from "pinia";

import { JsonDataService } from "@knime/ui-extension-service";

import type {
  AiResponse,
  ChatItem,
  Config,
  MessageResponse,
  Timeline,
  ToolCallTimelineItem,
  ToolResponse,
} from "@/types";
import { createId } from "@/utils/utils";

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
    async init() {
      await this.ensureJsonDataService();
      this.config = await this.getConfiguration();

      const initialAiMessage = await this.getInitialMessage();
      if (initialAiMessage) {
        this.addItemsToChat([initialAiMessage]);
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
    addAiReasoningToTimeline(msg: AiResponse) {
      if (msg.content?.trim()) {
        this.activeTimeline?.items.push({
          id: msg.id,
          type: "reasoning",
          content: msg.content,
        });
      }
    },
    addToolCallsToTimeline(msg: AiResponse) {
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
    addAiResponseWithToolCalls(msg: AiResponse) {
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
    addToolResponse(msg: ToolResponse) {
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
        consola.warn(
          `Could not find tool call with ID ${msg.id} in the active timeline.`,
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
    addAiResponse(msg: AiResponse) {
      // agent either finished the agentic loop, or simply replied without tool use
      this.completeActiveTimeline();
      this.isLoading = false;

      this.chatItems.push(msg);
    },
    addItemsToChat(responses: MessageResponse[]) {
      if (!responses.length) {
        return;
      }

      for (const msg of responses) {
        if (msg.type === "ai" && msg.toolCalls?.length) {
          this.addAiResponseWithToolCalls(msg);
          continue;
        }

        if (msg.type === "tool") {
          this.addToolResponse(msg);
          continue;
        }

        if (msg.type === "ai" && !msg.toolCalls?.length) {
          this.addAiResponse(msg);
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

      // TODO error message
      this.jsonDataService = await JsonDataService.getInstance();
    },
    async getConfiguration() {
      await this.ensureJsonDataService();

      // TODO error message
      const response = await this.jsonDataService?.data({
        method: "get_configuration",
      });

      return response;
    },
    async getInitialMessage() {
      await this.ensureJsonDataService();

      // TODO error message
      const response = await this.jsonDataService?.data({
        method: "get_initial_message",
      });

      return response;
    },
    async getLastMessages() {
      await this.ensureJsonDataService();

      // TODO error message
      const response = await this.jsonDataService?.data({
        method: "get_last_messages",
      });

      return response;
    },
    async checkIsProcessing() {
      await this.ensureJsonDataService();

      // TODO error message
      const response = await this.jsonDataService?.data({
        method: "is_processing",
      });

      return response;
    },
    async postUserMessage(msg: string) {
      await this.ensureJsonDataService();

      // TODO error message
      await this.jsonDataService?.data({
        method: "post_user_message",
        options: [msg],
      });
    },

    // ==== HIGHER LEVEL CHAT THINGIES ====
    async sendUserMessage(msg: string) {
      if (!msg.trim() || this.isLoading) {
        return;
      }

      this.isLoading = true; // gets set to false in addAiResponse() -> completeActiveTimeline()

      // 1. render user message in the chat
      this.addItemsToChat([{ id: createId(), content: msg, type: "human" }]);

      // 2. store for arrow up recall
      this.lastUserMessage = msg;

      // 3. send to backend and start polling
      await this.postUserMessage(msg);
      this.pollForNewMessages();
    },
    async pollForNewMessages() {
      let consecutiveEmptyPolls = 0;
      const maxEmptyPolls = 10;

      while (this.isLoading) {
        const msgs = await this.getLastMessages();

        if (msgs.length > 0) {
          this.addItemsToChat(msgs);
          consecutiveEmptyPolls = 0;

          // TODO check isLoading or isUsingTools here?
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

        // TODO error message
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
