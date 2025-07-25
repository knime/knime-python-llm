import { onMounted, ref } from "vue";

import { JsonDataService } from "@knime/ui-extension-service";

import NodeViewMessage from "@/components/NodeViewMessage.vue";
import type { MessageComponentMap, MessageResponse } from "@/types";
import AiMessage from "../components/AiMessage.vue";
import HumanMessage from "../components/HumanMessage.vue";
import ToolMessage from "../components/ToolMessage.vue";
import ErrorMessage from "../components/chat/ErrorMessage.vue";

export const initError = "Something went wrong. Try again later.";
export const sendingError = "There was an error while sending your message.";
export const processingError =
  "There was an error while processing your request.";

const createId = () => (Date.now() + 1).toString();

const messageComponents: MessageComponentMap = {
  ai: AiMessage,
  tool: ToolMessage,
  view: NodeViewMessage,
  human: HumanMessage,
  error: ErrorMessage,
};

type InitState = "idle" | "ready" | "error";
const initState = ref<InitState>("idle");
const requestQueue: Array<() => void> = [];

const messages = ref<MessageResponse[]>([]);
const isLoading = ref(false);

let jsonDataService: JsonDataService;

const displayNewMessages = (responses: MessageResponse[]) => {
  messages.value.push(...responses);
};

const pollForNewMessages = async () => {
  try {
    const response = await jsonDataService.data({
      method: "get_last_messages",
    });
    if (response?.length > 0) {
      displayNewMessages(response);
    } else {
      await pollForNewMessages();
    }
  } catch (error) {
    consola.error("Chat Agent: Error generating message response:", error);
    displayNewMessages([
      { id: createId(), content: processingError, type: "error" },
    ]);
  }
};

const postMessage = async (message: string) => {
  try {
    await jsonDataService.data({
      method: "post_user_message",
      options: [message],
    });
    await pollForNewMessages();
  } catch (error) {
    consola.error("Chat Agent: Error posting message:", error);
    displayNewMessages([
      { id: createId(), content: sendingError, type: "error" },
    ]);
  }
};

const sendMessage = (message: string) => {
  if (!message.trim()) {
    return;
  }

  const send = async () => {
    displayNewMessages([{ id: createId(), content: message, type: "human" }]);
    await postMessage(message);
    isLoading.value = false;
  };

  isLoading.value = true;

  if (initState.value === "ready") {
    send();
  } else if (initState.value === "error") {
    consola.warn(
      "Chat Agent: JsonDataService failed to initialize. Cannot send message.",
    );
  } else {
    requestQueue.push(send);
  }
};

const flushRequestQueue = () => {
  while (requestQueue.length > 0) {
    requestQueue.shift()?.();
  }
};

const init = async () => {
  try {
    jsonDataService = await JsonDataService.getInstance();
  } catch (error) {
    consola.error("Chat Agent: Error initializing JsonDataService:", error);
    displayNewMessages([{ id: createId(), content: initError, type: "error" }]);
    initState.value = "error";
    return;
  }

  let initialMessage: MessageResponse | undefined;
  try {
    initialMessage = await jsonDataService.data({
      method: "get_initial_message",
    });
    initState.value = "ready";
    flushRequestQueue();
  } catch (error) {
    consola.error("Chat Agent: Error fetching initial message:", error);
    initState.value = "error";
  }

  if (initialMessage) {
    displayNewMessages([initialMessage]);
  }
};

const resetChat = () => {
  initState.value = "idle";
  messages.value = [];
  isLoading.value = false;
};

export const useChat = () => {
  onMounted(init);
  return { messageComponents, messages, isLoading, sendMessage, resetChat };
};
