<script setup lang="ts">
import { ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";

import MessageBox from "./MessageBox.vue";
import MessageInput from "./MessageInput.vue";

defineProps<{ isLoading: boolean }>();
const emit = defineEmits<{ sendMessage: [message: string] }>();

const scrollableContainer = ref<HTMLElement | null>(null);
const messagesList = ref<HTMLElement | null>(null);

useScrollToBottom(scrollableContainer, messagesList);
</script>

<template>
  <main class="chat-interface">
    <div ref="scrollableContainer" class="scrollable-container">
      <div ref="messagesList" class="message-list">
        <slot />
        <MessageBox v-if="isLoading">
          <SkeletonItem height="24px" />
        </MessageBox>
      </div>
    </div>
    <MessageInput
      :is-loading="isLoading"
      @send-message="emit('sendMessage', $event)"
    />
  </main>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.chat-interface {
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  position: relative;
  overflow-y: hidden;
  padding: 0 0 var(--space-8) 0;
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
}

.scrollable-container {
  flex: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
}

.message-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-24);
  padding: var(--space-24) 0;
}
</style>
