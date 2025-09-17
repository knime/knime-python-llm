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
  InitializationState,
} from "@/types";
import { computed, ref, shallowRef, useId, watch } from "vue";

// static utilities
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

function completeActiveTimeline(timeline: Timeline) {
  const toolCallCount = timeline.items.filter(
    (item) => item.type === "tool_call",
  ).length;
  timeline.label = `Completed ${toolCallCount} tool call${toolCallCount === 1 ? "" : "s"}`;
  timeline.status = "completed";
}

function addAiReasoningToTimeline(msg: AiMessage, timeline: Timeline) {
  if (msg.content?.trim()) {
    timeline.items.push({
      id: msg.id,
      type: "reasoning",
      content: msg.content,
    });
  }
}

function addToolCallsToTimeline(msg: AiMessage, timeline: Timeline) {
  if (msg.toolCalls) {
    for (const tc of msg.toolCalls) {
      timeline.items.push({
        id: tc.id,
        type: "tool_call",
        name: tc.name,
        args: tc.args,
        status: "running",
      });
    }
  }
}

function addAiMessageWithToolCallsToTimeline(
  msg: AiMessage,
  timelineParam?: Timeline,
) {
  let timeline;
  // this either kicks off the agentic loop (create a new timeline)
  // or contributes to the active timeline
  if (!timelineParam) {
    timeline = {
      id: `timeline-${useId()}`,
      label: "Using tools",
      items: [],
      status: "active",
      type: "timeline",
    };
  } else {
    timeline = timelineParam;
  }

  // extract and add reasoning to timeline
  addAiReasoningToTimeline(msg, timeline);

  // extract and add tool calls to timeline
  addToolCallsToTimeline(msg, timeline);

  return timeline;
}

function addToolMessageToTimeLine(msg: ToolMessage, timeline: Timeline) {
  const index = timeline.items.findIndex((item) => item.id === msg.toolCallId);

  if (index === -1) {
    consola.error(
      `Could not find tool call with ID ${msg.toolCallId} in the active timeline.`,
    );
  } else {
    const originalItem = timeline.items[index] as ToolCallTimelineItem;

    const content = msg.content ?? "";
    const status = content.toLowerCase().startsWith("error")
      ? "failed"
      : "completed";

    const updatedItem: ToolCallTimelineItem = {
      ...originalItem,
      content,
      status,
    };

    timeline.items[index] = updatedItem;
  }
}

function addMessagesToChatItems(
  messages: Message[],
  chatItems: ChatItem[],
  shouldShowToolCalls: boolean,
  activeTimeline?: Timeline,
) {
  for (const msg of messages) {
    if (shouldShowToolCalls) {
      if (msg.type === "ai" && msg.toolCalls?.length) {
        const updatedTimeline = addAiMessageWithToolCallsToTimeline(
          msg,
          activeTimeline,
        );
        if (!activeTimeline) {
          activeTimeline = updatedTimeline;
          chatItems.push(updatedTimeline);
        }
        continue;
      }

      if (msg.type === "tool" && activeTimeline) {
        addToolMessageToTimeLine(msg, activeTimeline);
        continue;
      }

      if (msg.type === "ai" && !msg.toolCalls?.length && activeTimeline) {
        completeActiveTimeline(activeTimeline);
        chatItems.push(msg);
        continue;
      }
    } else if (msg.type === "ai" && msg.toolCalls?.length) {
      // ignore ai tool-call message if tool calls are not shown
      continue;
    }

    // TODO error-type message

    // otherwise add message directly to the conversation
    chatItems.push(msg);
  }
}

export const useChatStore = defineStore("chat", () => {
  // state
  let messages: Message[] = [];
  const lastMessages = shallowRef<Message[]>([]);
  const config = ref<Config | null>(null); // node settings that affect rendering
  const isLoading = ref(false); // true if agent is currently responding to user message
  const lastUserMessage = ref("");
  const jsonDataService = ref<JsonDataService | null>(null);
  const initState = ref<InitializationState>("idle");
  const requestQueue: Array<() => void> = []; // populated with messages sent before data service is ready

  // watchers
  watch(lastMessages, (newMessages, oldMessages) => {
    const lastMessage = newMessages.at(-1);
    if (lastMessage?.type === "ai" && !lastMessage.toolCalls?.length) {
      loadingFinished();
    }
  });

  // getters
  const isUsingTools = computed(() => {
    // last message is either an 'ai' message with tool calls or a 'tool' message
    const lastMessage = lastMessages.value.at(-1);
    return (
      (lastMessage?.type === "ai" &&
        (lastMessage.toolCalls?.length ?? 0) > 0) ||
      lastMessage?.type === "tool"
    );
  });

  const shouldShowToolUseIndicator = computed(() => {
    return (
      isLoading.value &&
      isUsingTools &&
      !config.value?.show_tool_calls_and_results
    );
  });

  const shouldShowGenericLoadingIndicator = computed(
    () => isLoading.value && !isUsingTools.value,
  );

  const shouldShowToolCalls = computed(() =>
    config.value ? config.value.show_tool_calls_and_results : true,
  );

  const chatItems = computed((previous: ChatItem[] | undefined): ChatItem[] => {
    if (lastMessages.value.length === 0) {
      return previous || [];
    }

    const chatItems =
      previous === undefined || previous.length === 0 ? [] : [...previous];
    const activeTimeline = chatItems.findLast(
      (item) => item.type === "timeline" && item.status === "active",
    ) as Timeline | undefined;
    addMessagesToChatItems(
      lastMessages.value,
      chatItems,
      shouldShowToolCalls.value,
      activeTimeline,
    );
    return chatItems;
  });

  // actions
  function addErrorMessage(errorType: keyof typeof ERROR_MESSAGES) {
    const errorMessage: ErrorMessage = {
      id: useId(),
      type: "error",
      content: ERROR_MESSAGES[errorType],
    };
    // TODO
    // chatItems.value.push(errorMessage);

    // TODO apply?
    isLoading.value = false;
    // TODO
    // completeActiveTimeline();
  }

  async function init() {
    if (initState.value !== "idle") {
      return;
    }

    try {
      await ensureJsonDataService();

      const initialConversation = await getInitialConversation();
      messages = initialConversation || [];
      lastMessages.value = initialConversation || [];

      // TODO
      const [configResponse, initialAiMessage] = await Promise.all([
        getConfiguration(),
        getInitialMessage(),
      ]);

      config.value = configResponse;
      // TODO
      if (initialAiMessage && !initialConversation) {
        lastMessages.value = [initialAiMessage];
        messages.push(initialAiMessage);
      }

      initState.value = "ready";
      flushRequestQueue();
    } catch (error) {
      consola.error("Chat Store: Error during initialization:", error);
      addErrorMessage("init");
      initState.value = "error";
    }
  }

  function flushRequestQueue() {
    while (requestQueue.length > 0) {
      const request = requestQueue.shift();
      request?.();
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

  async function getInitialConversation() {
    var initialData = await jsonDataService.value?.initialData();
    if (initialData) {
      return initialData;
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

  async function sendMessageToBackend(msg: string) {
    // store for arrow up recall
    lastUserMessage.value = msg;

    isLoading.value = true;

    try {
      await postUserMessage(msg);
      pollForNewMessages();
    } catch (error) {
      consola.error("Chat Store: Error sending user message:", error);
      addErrorMessage("sending");
    }
  }

  // ==== HIGHER LEVEL CHAT ACTIONS ====
  async function sendUserMessage(msg: string) {
    if (!msg.trim() || isLoading.value) {
      return;
    }

    const send = async () => {
      // 1. render user message
      const message: Message = { id: useId(), content: msg, type: "human" };
      messages.push(message);
      lastMessages.value = [message];
      // 2. send to backend
      await sendMessageToBackend(msg);
    };

    if (initState.value === "ready") {
      await send();
    } else if (initState.value === "error") {
      consola.warn("Chat Store: Initialization failed. Cannot send message.");
    } else {
      requestQueue.push(send);
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
            messages.push(...msgs);
            lastMessages.value = msgs;
            consecutiveEmptyPolls = 0;
          } else {
            consecutiveEmptyPolls++;

            if (consecutiveEmptyPolls >= maxEmptyPolls) {
              const response = await checkIsProcessing();

              if (!response?.is_processing) {
                loadingFinished();
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

  // TODO naming
  function loadingFinished() {
    isLoading.value = false;
    // TODO make sure 'isUsingTools' is false
    // TODO apply conversation
  }

  function applyConversation() {
    jsonDataService.value?.applyData(messages);
  }

  function resetChat() {
    messages = [];
    lastMessages.value = [];
    config.value = null;
    isLoading.value = false;
    lastUserMessage.value = "";
    jsonDataService.value = null;
    initState.value = "idle";
    requestQueue.length = 0;
  }

  return {
    //state
    config,
    isLoading,
    lastUserMessage,
    jsonDataService,
    initState,

    // getters
    shouldShowToolUseIndicator,
    shouldShowGenericLoadingIndicator,
    shouldShowToolCalls,
    chatItems,

    // actions
    addErrorMessage,
    init,
    ensureJsonDataService,
    getConfiguration,
    getInitialMessage,
    getInitialConversation,
    getLastMessages,
    checkIsProcessing,
    postUserMessage,
    sendUserMessage,
    sendMessageToBackend,
    flushRequestQueue,
    pollForNewMessages,
    applyConversation,
    resetChat,
  };
});
