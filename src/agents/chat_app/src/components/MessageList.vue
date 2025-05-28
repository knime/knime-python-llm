<script setup lang="ts">
import { ref, defineProps } from 'vue';
import MessageBubble from './MessageBubble.vue';
import TypingIndicator from './TypingIndicator.vue';
import { type Message, type Model } from '../types';

const props = defineProps<{
  messages: Message[];
  isLoading: boolean;
  selectedModel: Model;
}>();

const formatTime = (date: Date): string => {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: 'numeric',
    hour12: true,
  }).format(date);
};

const isNewDay = (current: Date, previous?: Date): boolean => {
  if (!previous) return true;
  
  const currentDate = new Date(current).setHours(0, 0, 0, 0);
  const previousDate = new Date(previous).setHours(0, 0, 0, 0);
  
  return currentDate !== previousDate;
};

const formatDate = (date: Date): string => {
  const today = new Date().setHours(0, 0, 0, 0);
  const messageDate = new Date(date).setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  if (messageDate === today) {
    return 'Today';
  } else if (messageDate === yesterday.setHours(0, 0, 0, 0)) {
    return 'Yesterday';
  } else {
    return new Intl.DateTimeFormat('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
    }).format(date);
  }
};
</script>

<template>
  <div class="message-list">
    <template v-for="(message, index) in messages" :key="message.id">
      <!-- Date separator -->
      <div 
        v-if="isNewDay(message.timestamp, index > 0 ? messages[index - 1].timestamp : undefined)"
        class="date-separator"
      >
        <div class="date-line"></div>
        <div class="date-text">{{ formatDate(message.timestamp) }}</div>
        <div class="date-line"></div>
      </div>
      
      <!-- System message -->
      <div v-if="message.role === 'system'" class="system-message">
        {{ message.content }}
      </div>
      
      <!-- User or AI message -->
      <div v-else class="message-wrapper" :class="message.role">
        <div class="message-content">
          <MessageBubble
            :content="message.content"
            :role="message.role"
            :timestamp="formatTime(message.timestamp)"
          />
        </div>
      </div>
    </template>
    
    <!-- Typing indicator -->
    <div v-if="isLoading" class="message-wrapper assistant">
      <div class="message-content">
        <TypingIndicator :model="selectedModel" />
      </div>
    </div>
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
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
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