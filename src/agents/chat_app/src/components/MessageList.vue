<script setup lang="ts">
import type { MessageResponse } from "@/types";

import ErrorMessage from "./ErrorMessage.vue";
import MessageBox from "./MessageBox.vue";

defineProps<{
  messages: MessageResponse[];
  isLoading: boolean;
}>();
</script>

<template>
  <div class="message-list">
    <template v-for="message in messages" :key="message.id">
      <ErrorMessage v-if="message.type === 'error'" v-bind="message" />

      <!-- User or AI message -->
      <div v-else class="message-wrapper" :class="message.type">
        <div class="message-content">
          <MessageBox v-bind="message" />
        </div>
      </div>
    </template>

    <!-- Loading new messages -->
    <MessageBox v-if="isLoading" name="" content="" type="ai" />
  </div>
</template>

<style scoped>
@import url("@knime/styles/css/mixins");

.message-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-16);
  padding: 0 var(--space-16);
}

.message-wrapper {
  display: flex;
  margin-bottom: var(--space-8);
}

.message-wrapper.human {
  justify-content: flex-end;
}

.message-content {
  max-width: 80%;
}
</style>
