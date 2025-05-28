<script setup lang="ts">
import { ref, nextTick, onMounted, watch, defineProps, defineEmits } from 'vue';
import MessageList from './MessageList.vue';
import MessageInput from './MessageInput.vue';
import { type Message } from '../types';

const props = defineProps<{
  messages: Message[];
  isLoading: boolean;
  userInput: string;
}>();

const emit = defineEmits<{
  (e: 'sendMessage'): void;
  (e: 'update:userInput', value: string): void;
}>();

const messagesContainer = ref<HTMLElement | null>(null);
const isAtBottom = ref(true);
const showScrollToBottom = ref(false);

const scrollToBottom = () => {
  if (!messagesContainer.value) return;
  
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

const updateUserInput = (value: string) => {
  emit('update:userInput', value);
};

watch(() => props.messages.length, () => {
  if (isAtBottom.value) {
    scrollToBottom();
  } else {
    showScrollToBottom.value = true;
  }
});

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
      <MessageList 
        :messages="messages" 
        :is-loading="isLoading"
      />
    </div>
    
    <button 
      v-if="showScrollToBottom" 
      class="scroll-to-bottom" 
      @click="scrollToBottom"
      aria-label="Scroll to bottom"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    </button>
    
    <MessageInput 
      :value="userInput"
      :is-loading="isLoading"
      @update:value="updateUserInput"
      @send="emit('sendMessage')"
    />
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
  padding: 16px 0;
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
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.3s ease, transform 0.3s ease;
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