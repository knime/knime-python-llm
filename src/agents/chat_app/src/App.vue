<script setup lang="ts">
import { ref } from 'vue';
import ChatHeader from './components/ChatHeader.vue';
import ChatInterface from './components/ChatInterface.vue';
import { type Message } from './types';
import { JsonDataService } from "@knime/ui-extension-service";
const messages = ref<Message[]>([
  {
    id: '1',
    content: 'Hello! I\'m your AI assistant. How can I help you today?',
    role: 'assistant',
    timestamp: new Date()
  }
]);

const userInput = ref('');
const isLoading = ref(false);

const sendMessageToBackend = async (message: string) => {
  const jsonDataService = await JsonDataService.getInstance();
  const res = jsonDataService.data({options: [ message]})

  console.log("res", res);
  alert(res);
}

const sendMessage = async () => {
  if (!userInput.value.trim()) return;
  
  const userMessage: Message = {
    id: Date.now().toString(),
    content: userInput.value,
    role: 'user',
    timestamp: new Date()
  };
  
  messages.value.push(userMessage);
  userInput.value = '';
  
  isLoading.value = true;
  
  try {
    // Simulate AI response delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    messages.value.push({
      id: (Date.now() + 1).toString(),
      content: 'I am an AI assistant. I understand you said: ' + userMessage.content,
      role: 'assistant',
      timestamp: new Date()
    });
  } catch (error) {
    console.error('Error generating response:', error);
    messages.value.push({
      id: (Date.now() + 1).toString(),
      content: 'Sorry, I encountered an error while processing your request.',
      role: 'assistant',
      timestamp: new Date()
    });
  } finally {
    isLoading.value = false;
  }

  sendMessageToBackend(userMessage.content);
};
</script>

<template>
  <div class="app-container">
    <ChatHeader />
    
    <ChatInterface 
      :messages="messages" 
      :is-loading="isLoading"
      @send-message="sendMessage"
      v-model:user-input="userInput"
    />
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}
</style>