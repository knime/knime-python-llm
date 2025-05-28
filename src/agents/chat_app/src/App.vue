<script setup lang="ts">
import { ref } from "vue";
import ChatInterface from "./components/ChatInterface.vue";
import { type Message } from "./types";
import { JsonDataService } from "@knime/ui-extension-service";

const messages = ref<Message[]>([]);

const userInput = ref("");
const isLoading = ref(false);

const sendMessageToBackend = async (message: string): Promise<string> => {
  const jsonDataService = await JsonDataService.getInstance();
  return jsonDataService.data({ options: [message] });
};

const sendMessage = async () => {
  if (!userInput.value.trim()) return;

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

    messages.value.push({
      id: (Date.now() + 1).toString(),
      content: response,
      role: "assistant",
      timestamp: new Date(),
    });
  } catch (error) {
    // TODO: Real logging
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
      :messages="messages"
      :is-loading="isLoading"
      @send-message="sendMessage"
      v-model:user-input="userInput"
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
