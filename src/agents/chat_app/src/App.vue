<script setup lang="ts">
import ChatInterface from "./components/chat/ChatInterface.vue";
import { useChat } from "./composables/useChat";

const { messageComponents, messages, isLoading, sendMessage } = useChat();
</script>

<template>
  <div class="app-container">
    <ChatInterface :is-loading="isLoading" @send-message="sendMessage">
      <template v-for="message in messages" :key="message.id">
        <component :is="messageComponents[message.type]" v-bind="message" />
      </template>
    </ChatInterface>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: var(--knime-white);
  color: var(--knime-masala);
}
</style>
