<script setup lang="ts">
</script>

<template>
  <div class="typing-container">
    <div class="typing-model">AI is typing</div>
    <div class="typing-indicator">
      <span></span>
      <span></span>
      <span></span>
    </div>
  </div>
</template>

<style scoped>
.typing-container {
  padding: 12px 16px;
  background-color: var(--ai-bubble-bg);
  border-radius: 18px;
  border-bottom-left-radius: 4px;
  max-width: 100px;
}

.typing-model {
  font-size: 0.7rem;
  margin-bottom: 4px;
  color: var(--ai-bubble-text);
  opacity: 0.8;
}

.typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--ai-bubble-text);
  opacity: 0.6;
  display: inline-block;
}

.typing-indicator span:nth-child(1) {
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation: typing 1.4s infinite ease-in-out 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation: typing 1.4s infinite ease-in-out 0.4s;
}

@keyframes typing {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-5px);
  }
}
</style>