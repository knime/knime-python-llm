import { defineStore } from "pinia";

import {
  JsonDataService,
  SharedDataService,
} from "@knime/ui-extension-service";

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
  ViewData,
  WorkflowInfo,
} from "@/types";
import { computed, ref, shallowRef, toRaw, useId, watch } from "vue";

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
  // this either kicks off the agentic loop (create a new timeline)
  // or contributes to the active timeline
  const timeline = timelineParam ?? {
    id: `timeline-${useId()}`,
    label: "Using tools",
    items: [],
    status: "active",
    type: "timeline",
  };

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
  timeline?: Timeline,
) {
  let activeTimeline = timeline;
  for (const msg of messages) {
    if (isAiMessageWithToolCalls(msg)) {
      const updatedTimeline = addAiMessageWithToolCallsToTimeline(
        msg as AiMessage,
        activeTimeline,
      );
      if (!activeTimeline) {
        activeTimeline = updatedTimeline;
        chatItems.push(updatedTimeline);
      }
      continue;
    }

    if (activeTimeline) {
      if (isToolMessage(msg)) {
        addToolMessageToTimeLine(msg as ToolMessage, activeTimeline);
        continue;
      }
      if (isAiMessageWithoutToolCalls(msg)) {
        completeActiveTimeline(activeTimeline);
        activeTimeline = undefined;
        chatItems.push(msg);
        continue;
      }
      if (msg.type === "error") {
        completeActiveTimeline(activeTimeline!);
        activeTimeline = undefined;
        chatItems.push(msg);
        continue;
      }
    }

    // otherwise add message directly to the conversation
    chatItems.push(msg);
  }
}

function isToolMessage(msg?: Message): boolean {
  if (!msg) {
    return false;
  }
  return msg.type === "tool";
}

function isErrorMessage(msg?: Message): boolean {
  if (!msg) {
    return false;
  }
  return msg.type === "error";
}

function isAiMessageWithToolCalls(msg?: Message): boolean {
  if (!msg) {
    return false;
  }
  return msg.type === "ai" && (msg.toolCalls?.length ?? 0) > 0;
}

function isAiMessageWithoutToolCalls(msg?: Message): boolean {
  if (!msg) {
    return false;
  }
  return msg.type === "ai" && !msg.toolCalls?.length;
}

export const useChatStore = defineStore("chat", () => {
  // state
  let messagesToPersist: Message[] = [];
  const lastMessage = shallowRef<Message | undefined>();
  const config = ref<Config | null>(null); // node settings that affect rendering
  const isLoading = ref(false); // true if agent is currently responding to user message
  const isInterrupted = ref(false); // true if agent is currently responding to cancellation
  const chatItems = ref<ChatItem[]>([]);
  const lastUserMessage = ref("");
  const jsonDataService = ref<JsonDataService | null>(null);
  const sharedDataService = ref<SharedDataService | null>(null);
  const initState = ref<InitializationState>("idle");
  const requestQueue: Array<() => void> = []; // populated with messages sent before data service is ready

  // getters
  const isUsingTools = computed(() => {
    if (!isLoading.value || !lastMessage.value) {
      return false;
    }
    return (
      isAiMessageWithToolCalls(lastMessage.value) ||
      isToolMessage(lastMessage.value)
    );
  });

  const shouldShowStatusIndicator = computed(
    () => isLoading.value && isUsingTools.value && !shouldShowToolCalls.value,
  );

  const shouldShowGenericLoadingIndicator = computed(
    () => isLoading.value && !isUsingTools.value,
  );

  const shouldShowToolCalls = computed(
    () => config.value?.show_tool_calls_and_results,
  );

  // actions
  function addErrorMessage(errorType: keyof typeof ERROR_MESSAGES) {
    const errorMessage: ErrorMessage = {
      id: useId(),
      type: "error",
      content: ERROR_MESSAGES[errorType],
    };

    const wasLoading = isLoading.value;
    addMessages([errorMessage], true, false);

    // addMessages calls finishLoading if isLoading is true
    if (!wasLoading && config.value) {
      finishLoading(false);
    }
  }

  async function init() {
    if (initState.value !== "idle") {
      return;
    }

    try {
      await ensureServices();

      const viewData = await getInitialViewData();
      if (viewData) {
        addMessages(
          viewData.conversation,
          viewData.config.show_tool_calls_and_results,
        );
        config.value = viewData.config;
        sharedDataService.value?.shareData(viewData);
      } else {
        const [configResponse, initialAiMessage] = await Promise.all([
          getConfiguration(),
          getInitialMessage(),
        ]);
        config.value = configResponse;
        if (initialAiMessage) {
          addMessages([initialAiMessage], true);
        }
        sharedDataService.value?.shareData({
          conversation: messagesToPersist,
          config: toRaw(config.value),
        });
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
  async function ensureServices() {
    if (jsonDataService.value && sharedDataService.value) {
      return;
    }

    try {
      jsonDataService.value = await JsonDataService.getInstance();
      sharedDataService.value = await SharedDataService.getInstance();
    } catch (error) {
      consola.error("Chat Store: Failed to get service instances:", error);
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
      return {
        show_tool_calls_and_results: false,
        reexecution_trigger: "NONE",
      };
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

  async function getInitialViewData(): Promise<ViewData | undefined> {
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

  async function getCombinedToolsWorkflowInfo(): Promise<WorkflowInfo> {
    const response = await jsonDataService.value?.data({
      method: "get_combined_tools_workflow_info",
    });
    if (response) {
      return {
        projectId: response.project_id,
        workflowId: response.workflow_id,
      };
    }
    throw new Error("Failed to get combined tools workflow info");
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

  async function cancelAgent() {
    try {
      isInterrupted.value = true;
      await jsonDataService.value?.data({
        method: "cancel_agent",
      });
    } catch (error) {
      isInterrupted.value = false;
      consola.error("Chat Store: Failed to cancel agent:", error);
      throw error;
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
      addMessages([message], shouldShowToolCalls.value || false);
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
            addMessages(msgs, shouldShowToolCalls.value || false);
            consecutiveEmptyPolls = 0;
          } else {
            consecutiveEmptyPolls++;

            if (consecutiveEmptyPolls >= maxEmptyPolls) {
              const response = await checkIsProcessing();

              if (!response?.is_processing) {
                finishLoading(true);
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

  function addMessages(
    msgs: Message[],
    showToolCallsResults: boolean,
    shallApplyOnFinish: boolean = true,
  ) {
    messagesToPersist.push(...msgs);
    lastMessage.value = msgs.at(-1);

    const isInteractionFinished =
      isAiMessageWithoutToolCalls(lastMessage.value) ||
      isErrorMessage(lastMessage.value);

    if (isLoading.value && isInteractionFinished) {
      finishLoading(shallApplyOnFinish && initState.value !== "idle"); // does not apply view data on initial load
    }

    // update chatItems
    const lastMessagesToDisplay = showToolCallsResults
      ? msgs
      : msgs.filter(
          (msg) => !isToolMessage(msg) && !isAiMessageWithToolCalls(msg),
        );
    const activeTimeline = chatItems.value.findLast(
      (item) => item.type === "timeline" && item.status === "active",
    ) as Timeline | undefined;
    addMessagesToChatItems(
      lastMessagesToDisplay,
      chatItems.value,
      activeTimeline,
    );
  }

  function finishLoading(shallApplyViewData: boolean) {
    isLoading.value = false;
    isInterrupted.value = false;
    const viewData: ViewData = {
      conversation: messagesToPersist,
      config: toRaw(config.value!),
    };

    // share view data
    sharedDataService.value?.shareData(viewData);

    // optionally apply view data
    if (
      shallApplyViewData &&
      config.value?.reexecution_trigger === "INTERACTION"
    ) {
      jsonDataService.value?.applyData(viewData);
    }
  }

  function resetChat() {
    messagesToPersist = [];
    lastMessage.value = undefined;
    config.value = null;
    chatItems.value = [];
    isLoading.value = false;
    isInterrupted.value = false;
    lastUserMessage.value = "";
    jsonDataService.value = null;
    initState.value = "idle";
    requestQueue.length = 0;
  }

  return {
    //state
    config,
    chatItems,
    lastMessage,
    isLoading,
    isInterrupted,
    lastUserMessage,
    jsonDataService,
    initState,

    // getters
    shouldShowStatusIndicator,
    shouldShowGenericLoadingIndicator,
    shouldShowToolCalls,
    isUsingTools,

    // actions
    addErrorMessage,
    addMessages,
    init,
    getConfiguration,
    getCombinedToolsWorkflowInfo,
    getInitialMessage,
    getLastMessages,
    checkIsProcessing,
    postUserMessage,
    sendUserMessage,
    sendMessageToBackend,
    flushRequestQueue,
    pollForNewMessages,
    resetChat,
    cancelAgent,
  };
});
