<script setup lang="ts">
import { onMounted, ref } from "vue";

import { JsonDataService } from "@knime/ui-extension-service";

import ChatInterface from "./components/ChatInterface.vue";
import { type Message } from "./types";

const messages = ref<Message[]>([]);

const userInput = ref("");
const isLoading = ref(false);

onMounted(async () => {
  // initializes the python backend (e.g. starting the python process)
  const jsonDataService = await JsonDataService.getInstance();
  jsonDataService.data({ method: "init" });
});

const sendMessageToBackend = async (message: string): Promise<any[]> => {
  const jsonDataService = await JsonDataService.getInstance();
  await jsonDataService.data({
    method: "post_user_message",
    options: [message],
  });

  // long polling for response
  // TODO cancel polling eventually?
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const response = await jsonDataService.data({
          method: "get_last_messages",
        });
        if (response && response.length > 0) {
          resolve(response);
        } else {
          poll();
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
};

const sendMessage = async () => {
  if (!userInput.value.trim()) {
    return;
  }

  const userMessage: Message = {
    id: Date.now().toString(),
    content: userInput.value,
    role: "user",
    timestamp: new Date(),
  };

  messages.value.push(userMessage);
  userInput.value = "";

  isLoading.value = true;

  try {
    const response = await sendMessageToBackend(userMessage.content);

    for (const message of response) {
      const toolCalls = message.toolCalls
        ? `;tool calls: ${message.toolCalls}`
        : "";
      messages.value.push({
        id: (Date.now() + 1).toString(),
        content: `${message.role}: ${message.content}${toolCalls}`,
        role: message.role,
        timestamp: new Date(),
      });
    }
  } catch (error) {
    // TODO: Real logging
    // eslint-disable-next-line no-console
    console.error("Error generating response:", error);
    messages.value.push({
      id: (Date.now() + 1).toString(),
      content: "There was an error while processing your request.",
      role: "system",
      timestamp: new Date(),
    });
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="app-container">
    <ChatInterface
      v-model:user-input="userInput"
      :messages="messages"
      :is-loading="isLoading"
      @send-message="sendMessage"
    />
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--knime-white);
  color: var(--knime-masala);
}
</style>
