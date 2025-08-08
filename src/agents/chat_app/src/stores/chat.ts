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
import { computed, ref } from "vue";

const ERROR_MESSAGES = {
  init: "Something went wrong. Try again later.",
  sending: "There was an error while sending your message.",
  processing: "There was an error while processing your request.",
  connection: "Failed to connect to the service. Please try again.",
  configuration:
    "Failed to load configuration. Some features may not work properly.",
  polling:
    "Lost connection while waiting for response. Please try sending your message again.",
};

export const useChatStore = defineStore("chat", () => {
  // state
  const chatItems = ref<ChatItem[]>([]);
  const config = ref<Config | null>(null);
  const isLoading = ref(false);
  const isUsingTools = ref(false);
  const lastUserMessage = ref("");
  const activeTimeline = ref<Timeline | null>(null);
  const jsonDataService = ref<JsonDataService | null>(null);

  // getters
  const shouldShowToolUseIndicator = computed(() => {
    return (
      isLoading.value &&
      isUsingTools.value &&
      !config.value?.show_tool_calls_and_results
    );
  });
  const shouldShowGenericLoadingIndicator = computed(
    () => isLoading.value && !isUsingTools.value,
  );
  const shouldShowToolCalls = computed(
    () => config.value?.show_tool_calls_and_results,
  );

  // actions
  function addErrorMessage(errorType: keyof typeof ERROR_MESSAGES) {
    const errorMessage: ErrorMessage = {
      id: createId(),
      type: "error",
      content: ERROR_MESSAGES[errorType],
    };
    chatItems.value.push(errorMessage);

    isLoading.value = false;
    isUsingTools.value = false;
    completeActiveTimeline();
  }

  async function init() {
    try {
      await ensureJsonDataService();

      const [configResponse, initialAiMessage] = await Promise.all([
        getConfiguration(),
        getInitialMessage(),
      ]);

      config.value = configResponse;
      if (initialAiMessage) {
        addItemsToChat([initialAiMessage]);
      }
    } catch (error) {
      consola.error("Chat Store: Error during initialization:", error);
      addErrorMessage("init");
    }
  }

  function ensureActiveTimeline() {
    if (activeTimeline.value) {
      return;
    }

    const newTimeline: Timeline = {
      id: `timeline-${createId()}`,
      label: "Using tools",
      items: [],
      status: "active",
      type: "timeline",
    };
    chatItems.value.push(newTimeline);
    activeTimeline.value = newTimeline;
    isUsingTools.value = true;
  }

  function completeActiveTimeline() {
    if (!activeTimeline.value) {
      return;
    }

    isUsingTools.value = false;

    const toolCallCount = activeTimeline.value.items.filter(
      (item) => item.type === "tool_call",
    ).length;
    activeTimeline.value.label = `Completed ${toolCallCount} tool call${toolCallCount === 1 ? "" : "s"}`;
    activeTimeline.value.status = "completed";
    activeTimeline.value = null;
  }

  function addAiReasoningToTimeline(msg: AiMessage) {
    if (msg.content?.trim()) {
      activeTimeline.value?.items.push({
        id: msg.id,
        type: "reasoning",
        content: msg.content,
      });
    }
  }

  function addToolCallsToTimeline(msg: AiMessage) {
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        activeTimeline.value?.items.push({
          id: tc.id,
          type: "tool_call",
          name: tc.name,
          args: tc.args,
          status: "running",
        });
      }
    }
  }

  function addAiMessageWithToolCalls(msg: AiMessage) {
    if (!shouldShowToolCalls.value) {
      isUsingTools.value = true;
      return;
    }
    // this either kicks off the agentic loop (create a new timeline)
    // or contributes to the active timeline
    ensureActiveTimeline();

    // extract and add reasoning to timeline
    addAiReasoningToTimeline(msg);

    // extract and add tool calls to timeline
    addToolCallsToTimeline(msg);
  }

  function addToolMessage(msg: ToolMessage) {
    if (!activeTimeline.value) {
      return;
    }

    if (!shouldShowToolCalls.value) {
      return;
    }

    const index = activeTimeline.value.items.findIndex(
      (item) => item.id === msg.toolCallId,
    );

    if (index === -1) {
      consola.error(
        `Could not find tool call with ID ${msg.toolCallId} in the active timeline.`,
      );
    } else {
      const originalItem = activeTimeline.value.items[
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

      activeTimeline.value.items[index] = updatedItem;
    }
  }

  function addAiMessage(msg: AiMessage) {
    // agent either finished the agentic loop, or simply replied without tool use
    completeActiveTimeline();
    isLoading.value = false;

    chatItems.value.push(msg);
  }

  function addItemsToChat(messages: Message[]) {
    if (!messages.length) {
      return;
    }

    for (const msg of messages) {
      if (msg.type === "ai" && msg.toolCalls?.length) {
        addAiMessageWithToolCalls(msg);
        continue;
      }

      if (msg.type === "tool") {
        addToolMessage(msg);
        continue;
      }

      if (msg.type === "ai" && !msg.toolCalls?.length) {
        addAiMessage(msg);
        continue;
      }

      // otherwise add message directly to the conversation
      chatItems.value.push(msg);
    }
  }

  // ==== BACKEND API ACTIONS ====
  async function ensureJsonDataService() {
    if (jsonDataService.value) {
      return;
    }

    try {
      jsonDataService.value = await JsonDataService.getInstance();
    } catch (error) {
      consola.error(
        "Chat Store: Failed to get JsonDataService instance:",
        error,
      );
      throw new Error("Failed to connect to the service");
    }
  }

  async function getConfiguration() {
    try {
      const response = await jsonDataService.value?.data({
        method: "get_configuration",
      });
      return response;
    } catch (error) {
      consola.error("Chat Store: Failed to get configuration:", error);
      return { show_tool_calls_and_results: false };
    }
  }

  async function getInitialMessage() {
    try {
      const response = await jsonDataService.value?.data({
        method: "get_initial_message",
      });
      return response;
    } catch (error) {
      consola.error("Chat Store: Failed to get initial message:", error);
      return null;
    }
  }

  async function getLastMessages() {
    try {
      const response = await jsonDataService.value?.data({
        method: "get_last_messages",
      });
      return response || [];
    } catch (error) {
      consola.error("Chat Store: Failed to get last messages:", error);
      throw error;
    }
  }

  async function checkIsProcessing() {
    try {
      const response = await jsonDataService.value?.data({
        method: "is_processing",
      });
      return response;
    } catch (error) {
      consola.error("Chat Store: Failed to check processing status:", error);
      return { is_processing: false };
    }
  }

  async function postUserMessage(msg: string) {
    try {
      await jsonDataService.value?.data({
        method: "post_user_message",
        options: [msg],
      });
    } catch (error) {
      consola.error("Chat Store: Failed to post user message:", error);
      throw error;
    }
  }

  // ==== HIGHER LEVEL CHAT CONTROLS ====
  async function sendUserMessage(msg: string) {
    if (!msg.trim() || isLoading.value) {
      return;
    }

    isLoading.value = true;

    // 1. render user message in the chat
    addItemsToChat([{ id: createId(), content: msg, type: "human" }]);

    // 2. store for arrow up recall
    lastUserMessage.value = msg;

    try {
      // 3. send to backend and start polling
      await postUserMessage(msg);
      pollForNewMessages();
    } catch (error) {
      consola.error("Chat Store: Error sending user message:", error);
      addErrorMessage("sending");
    }
  }

  async function pollForNewMessages() {
    let consecutiveEmptyPolls = 0;
    const maxEmptyPolls = 10;

    try {
      while (isLoading.value) {
        try {
          const msgs = await getLastMessages();

          if (msgs.length > 0) {
            addItemsToChat(msgs);
            consecutiveEmptyPolls = 0;
          } else {
            consecutiveEmptyPolls++;

            if (consecutiveEmptyPolls >= maxEmptyPolls) {
              const response = await checkIsProcessing();

              if (!response?.is_processing) {
                isLoading.value = false;
                isUsingTools.value = false;
                break;
              }

              // still processing, give agent and tools more time
              consecutiveEmptyPolls = 0;
            }
          }
        } catch (error) {
          consola.error("Chat Store: Error during message polling:", error);
          addErrorMessage("polling");
          break;
        }
      }
    } catch (error) {
      consola.error("Chat Store: Fatal error in polling loop:", error);
      addErrorMessage("processing");
    }
  }

  function resetChat() {
    chatItems.value = [];
    config.value = null;
    isLoading.value = false;
    isUsingTools.value = false;
    lastUserMessage.value = "";
    activeTimeline.value = null;
    jsonDataService.value = null;
  }

  return {
    //state
    chatItems,
    config,
    isLoading,
    isUsingTools,
    lastUserMessage,
    activeTimeline,
    jsonDataService,

    // getters
    shouldShowToolUseIndicator,
    shouldShowGenericLoadingIndicator,
    shouldShowToolCalls,

    // actions
    addErrorMessage,
    init,
    ensureActiveTimeline,
    completeActiveTimeline,
    addAiReasoningToTimeline,
    addToolCallsToTimeline,
    addAiMessageWithToolCalls,
    addToolMessage,
    addAiMessage,
    addItemsToChat,
    ensureJsonDataService,
    getConfiguration,
    getInitialMessage,
    getLastMessages,
    checkIsProcessing,
    postUserMessage,
    sendUserMessage,
    pollForNewMessages,
    resetChat,
  };
});
