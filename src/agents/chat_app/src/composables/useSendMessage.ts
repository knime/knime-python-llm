import { ref } from "vue";

import { JsonDataService } from "@knime/ui-extension-service";

import type { Message, MessageResponse } from "@/types";

const sendingError = "There was an error while sending your message.";
const processingError = "There was an error while processing your request.";

// TODO: Maybe split messages into `useMessages` composable
const messages = ref<Message[]>([]);
const isLoading = ref(false);

// TODO: Refactor for new styling
const createNewMessage = (response: MessageResponse): Message => {
  const toolCalls = response.toolCalls
    ? `;tool calls: ${response.toolCalls}`
    : "";
  return {
    id: (Date.now() + 1).toString(),
    content: `${response.role}: ${response.content}${toolCalls}`,
    role: response.role,
  };
};

const displayNewMessages = (responses: MessageResponse[]) => {
  messages.value.push(...responses.map(createNewMessage));
};

const pollForNewMessages = async () => {
  try {
    const jsonDataService = await JsonDataService.getInstance();
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
    displayNewMessages([{ content: processingError, role: "system" }]);
  }
};

const postMessage = async (message: string) => {
  try {
    const jsonDataService = await JsonDataService.getInstance();
    await jsonDataService.data({
      method: "post_user_message",
      options: [message],
    });
    await pollForNewMessages();
  } catch (error) {
    consola.error("Chat Agent: Error posting message:", error);
    displayNewMessages([{ content: sendingError, role: "system" }]);
  }
};

const sendMessage = async (message: string) => {
  if (!message.trim()) {
    return;
  }
  isLoading.value = true;
  displayNewMessages([{ content: message, role: "user" }]);
  await postMessage(message);
  isLoading.value = false;
};

export const useSendMessage = () => ({ isLoading, messages, sendMessage });
