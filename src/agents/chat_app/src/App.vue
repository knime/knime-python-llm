<script setup lang="ts">
import { ref, onMounted } from "vue";
import ChatInterface from "./components/ChatInterface.vue";
import { type Message } from "./types";
import { JsonDataService } from "@knime/ui-extension-service";

const messages = ref<Message[]>([]);

const userInput = ref("");
const isLoading = ref(false);


onMounted(async () => {
  // initializes the python backend (e.g. starting the python process)
  const jsonDataService = await JsonDataService.getInstance();
  jsonDataService.data({ method: "init"});
});

const sendMessageToBackend = async (message: string): Promise<any[]> => {
  const jsonDataService = await JsonDataService.getInstance();
  return jsonDataService.data({ method: "post_user_message", options: [message] });
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

    for(const message of response) {
      const tool_calls = message.tool_calls ? `;tool calls: ${message.tool_calls}` : "";
      messages.value.push({
        id: (Date.now() + 1).toString(),
        content: `${message.role}: ${message.content}${tool_calls}`,
        role: message.role,
        timestamp: new Date(),
      });
    }
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
