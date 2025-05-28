<script setup lang="ts">
import { defineProps } from "vue";
import MessageBox from "./MessageBox.vue";
import { type Message } from "../types";

defineProps<{
  messages: Message[];
  isLoading: boolean;
}>();
</script>

<template>
  <div class="message-list">
    <template v-for="message in messages" :key="message.id">
      <!-- System message -->
      <div v-if="message.role === 'system'" class="system-message">
        {{ message.content }}
      </div>

      <!-- User or AI message -->
      <div v-else class="message-wrapper" :class="message.role">
        <div class="message-content">
          <MessageBox :content="message.content" :role="message.role" />
        </div>
      </div>
    </template>

    <!-- TODO: Improve isLoading state (maybe approach like in K-AI?) -->
    <MessageBox v-if="isLoading" content="" role="assistant" />
  </div>
</template>

<style scoped>
.message-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 0 16px;
}

.message-wrapper {
  display: flex;
  margin-bottom: 8px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-wrapper.user {
  justify-content: flex-end;
}

.message-wrapper.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 80%;
}

@media (max-width: 768px) {
  .message-content {
    max-width: 90%;
  }
}

.system-message {
  text-align: center;
  font-size: 0.825rem;
  color: var(--color-text-secondary);
  margin: 8px 0;
  padding: 4px 12px;
  border-radius: 16px;
  background-color: var(--color-bg-secondary);
  align-self: center;
  animation: fadeIn 0.3s ease;
}

.date-separator {
  display: flex;
  align-items: center;
  margin: 16px 0;
  gap: 16px;
}

.date-line {
  flex-grow: 1;
  height: 1px;
  background-color: var(--color-border);
}

.date-text {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  white-space: nowrap;
  padding: 0 8px;
}
</style>
