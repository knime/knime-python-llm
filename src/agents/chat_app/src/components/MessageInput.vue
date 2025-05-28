<script setup lang="ts">
import { ref, defineEmits, defineProps, nextTick } from 'vue';

const props = defineProps<{
  value: string;
  isLoading: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:value', value: string): void;
  (e: 'send'): void;
}>();

const inputElement = ref<HTMLTextAreaElement | null>(null);

const handleInput = (event: Event) => {
  const target = event.target as HTMLTextAreaElement;
  emit('update:value', target.value);
  resizeTextarea();
};

const resizeTextarea = () => {
  if (!inputElement.value) return;
  
  inputElement.value.style.height = 'auto';
  inputElement.value.style.height = `${Math.min(inputElement.value.scrollHeight, 150)}px`;
};

const sendMessage = () => {
  if (props.isLoading || !props.value.trim()) return;
  emit('send');
  
  // Reset height after sending
  nextTick(() => {
    if (inputElement.value) {
      inputElement.value.style.height = 'auto';
    }
  });
};

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
};
</script>

<template>
  <div class="message-input-container">
    <div class="input-wrapper">
      <textarea
        ref="inputElement"
        class="message-input"
        placeholder="Type a message..."
        :value="value"
        @input="handleInput"
        @keydown="handleKeydown"
        :disabled="isLoading"
      ></textarea>
      
      <button 
        class="send-button" 
        :class="{ 'active': value.trim(), 'loading': isLoading }"
        :disabled="!value.trim() || isLoading"
        @click="sendMessage"
        aria-label="Send message"
      >
        <span v-if="isLoading" class="loading-spinner"></span>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.message-input-container {
  padding: 16px 24px;
  background-color: var(--color-bg-secondary);
  border-top: 1px solid var(--color-border);
}

.input-wrapper {
  display: flex;
  background-color: var(--color-bg-primary);
  border-radius: 24px;
  overflow: hidden;
  border: 1px solid var(--color-border);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.input-wrapper:focus-within {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.message-input {
  flex-grow: 1;
  padding: 12px 16px;
  min-height: 24px;
  max-height: 150px;
  border: none;
  background: transparent;
  color: var(--color-text-primary);
  font-family: inherit;
  font-size: 0.95rem;
  line-height: 1.5;
  resize: none;
  outline: none;
}

.message-input::placeholder {
  color: var(--color-text-secondary);
  opacity: 0.7;
}

.send-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  margin: 4px;
  border: none;
  border-radius: 50%;
  background-color: var(--color-gray-200);
  color: var(--color-gray-500);
  cursor: pointer;
  transition: all 0.2s ease;
}

.send-button.active {
  background-color: var(--color-primary);
  color: white;
}

.send-button.active:hover {
  background-color: var(--color-primary-dark);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>