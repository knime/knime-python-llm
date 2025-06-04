<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";

import { useSendMessage } from "@/composables/useSendMessage";

import MessageInput from "./MessageInput.vue";
import MessageList from "./MessageList.vue";

const messagesContainer = ref<HTMLElement | null>(null);
const isAtBottom = ref(true);
const showScrollToBottom = ref(false);

const { messages, isLoading, sendMessage } = useSendMessage();

const scrollToBottom = () => {
  if (!messagesContainer.value) {
    return;
  }

  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
      showScrollToBottom.value = false;
    }
  });
};

const handleScroll = (event: Event) => {
  const target = event.target as HTMLElement;
  const { scrollTop, scrollHeight, clientHeight } = target;

  const bottomThreshold = 100;
  isAtBottom.value = scrollHeight - scrollTop - clientHeight < bottomThreshold;

  showScrollToBottom.value = !isAtBottom.value;
};

watch(
  () => messages.value.length,
  () => {
    if (isAtBottom.value) {
      scrollToBottom();
    } else {
      showScrollToBottom.value = true;
    }
  },
);

onMounted(() => {
  scrollToBottom();
});
</script>

<template>
  <main class="chat-interface">
    <div
      ref="messagesContainer"
      class="messages-container"
      @scroll="handleScroll"
    >
      <MessageList :messages="messages" :is-loading="isLoading" />
    </div>
    <MessageInput :is-loading="isLoading" :send-message="sendMessage" />
  </main>
</template>

<style scoped>
@import url("@knime/styles/css/mixins");

.chat-interface {
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  position: relative;
  overflow: hidden;
}

.messages-container {
  flex-grow: 1;
  overflow-y: auto;
  padding: var(--space-24) 0;
  scroll-behavior: smooth;
}
</style>
