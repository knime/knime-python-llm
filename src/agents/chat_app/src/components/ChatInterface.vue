<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";

import { useSendMessage } from "@/composables/useSendMessage";

import MessageInput from "./MessageInput.vue";
import MessageList from "./MessageList.vue";

const messagesContainer = ref<HTMLElement | null>(null);
const isAtBottom = ref(true);
const showScrollToBottom = ref(false);

const { messages } = useSendMessage();

// TODO: Check if this functionality is necessary/useful
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
      <MessageList />
    </div>

    <button
      v-if="showScrollToBottom"
      class="scroll-to-bottom"
      aria-label="Scroll to bottom"
      @click="scrollToBottom"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </button>

    <MessageInput />
  </main>
</template>

<style scoped>
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
  padding: 24px 0;
  scroll-behavior: smooth;
}

.scroll-to-bottom {
  position: absolute;
  right: 24px;
  bottom: 80px;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: var(--color-primary);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  cursor: pointer;
  box-shadow: 0 2px 8px rgb(0 0 0 / 20%);
  opacity: 0;
  transform: translateY(20px);
  transition:
    opacity 0.3s ease,
    transform 0.3s ease;
  z-index: 5;
}

.scroll-to-bottom:hover {
  background-color: var(--color-primary-dark);
}

.scroll-to-bottom[v-if="showScrollToBottom"] {
  opacity: 1;
  transform: translateY(0);
}
</style>
