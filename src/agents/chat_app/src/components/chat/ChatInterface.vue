<script setup lang="ts">
import { ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";

import MessageBox from "./MessageBox.vue";
import MessageInput from "./MessageInput.vue";

defineProps<{ isLoading: boolean }>();
const emit = defineEmits<{ sendMessage: [message: string] }>();

const messagesContainer = ref<HTMLElement | null>(null);

useScrollToBottom(messagesContainer);
</script>

<template>
  <main class="chat-interface">
    <div ref="messagesContainer" class="message-list">
      <slot />
      <MessageBox v-if="isLoading">
        <SkeletonItem height="24px" />
      </MessageBox>
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
  overflow: hidden;
  padding: var(--space-8) var(--space-4);
}

.message-list {
  flex-grow: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
  display: flex;
  flex-direction: column;
  gap: var(--space-24);
  padding: var(--space-24) var(--space-16);
}
</style>
