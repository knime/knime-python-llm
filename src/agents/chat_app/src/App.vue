<script setup lang="ts">
import { onBeforeMount, ref } from "vue";

import ChatGraphView from "./components/chat/ChatGraphView.vue";
import ChatInterface from "./components/chat/ChatInterface.vue";
import { useChatStore } from "./stores/chat";

const chatStore = useChatStore();
const currentView = ref<"chat" | "graph">("chat");

onBeforeMount(async () => {
  await chatStore.init();
});

const openMessageInChat = (messageId: string) => {
  window.location.hash = `#${messageId}`;
  currentView.value = "chat";
};
</script>

<template>
  <div class="app-container">
    <header class="view-toggle">
      <button
        class="toggle-button"
        :class="{ active: currentView === 'chat' }"
        type="button"
        @click="currentView = 'chat'"
      >
        Chat
      </button>
      <button
        class="toggle-button"
        :class="{ active: currentView === 'graph' }"
        type="button"
        @click="currentView = 'graph'"
      >
        Graph
      </button>
    </header>
    <ChatInterface v-if="currentView === 'chat'" />
    <ChatGraphView v-else @open-message="openMessageInChat" />
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

.view-toggle {
  display: flex;
  gap: var(--space-8);
  padding: var(--space-12) var(--space-24) 0;
}

.toggle-button {
  border: 1px solid var(--knime-silver-sand);
  border-radius: var(--space-4);
  background: var(--knime-white);
  padding: var(--space-4) var(--space-12);
  cursor: pointer;
  font-size: 12px;
}

.toggle-button.active {
  background: var(--knime-porcelain);
  border-color: var(--knime-masala);
  font-weight: 600;
}
</style>
