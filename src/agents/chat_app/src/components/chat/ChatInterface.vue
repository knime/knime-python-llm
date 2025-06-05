<script setup lang="ts">
import { ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";

import MessageBox from "./MessageBox.vue";
import MessageInput from "./MessageInput.vue";

defineProps<{
  isLoading: boolean;
}>();

const emit = defineEmits<{
  sendMessage: [message: string];
}>();

const messagesContainer = ref<HTMLElement | null>(null);

const { handleScroll } = useScrollToBottom(messagesContainer);
</script>

<template>
  <main class="chat-interface">
    <div ref="messagesContainer" class="message-list" @scroll="handleScroll">
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
  padding: var(--space-8) 3px;
}

.message-list {
  flex-grow: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
  display: flex;
  flex-direction: column;
  gap: var(--space-24);
  padding: var(--space-24) var(--space-16);

  & :deep(.icon svg) {
    @mixin svg-icon-size 16;
  }
}
</style>
