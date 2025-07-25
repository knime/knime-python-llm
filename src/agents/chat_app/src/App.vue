<script setup lang="ts">
import { onMounted } from "vue";

import ChatInterface from "./components/chat/ChatInterface.vue";
import { useChatStore } from "./stores/chat";

const chatStore = useChatStore();

onMounted(async () => {
  await chatStore.init();
});
</script>

<template>
  <div class="app-container">
    <ChatInterface />
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
