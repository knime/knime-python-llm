<script setup lang="ts">
// Simple header with no settings button
</script>

<template>
  <header class="chat-header">
    <div class="model-info">
      <div class="model-status">
        <div class="status-indicator"></div>
        <span class="status-text">Connected</span>
      </div>
      <h1 class="model-name">AI Assistant</h1>
    </div>
  </header>
</template>

<style scoped>
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background-color: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
  z-index: 10;
}

.model-info {
  display: flex;
  flex-direction: column;
}

.model-status {
  display: flex;
  align-items: center;
  margin-bottom: 4px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--color-success);
  margin-right: 8px;
  position: relative;
}

.status-indicator::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 50%;
  background-color: var(--color-success);
  opacity: 0.4;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 0.4;
  }
  50% {
    transform: scale(2);
    opacity: 0;
  }
  100% {
    transform: scale(1);
    opacity: 0;
  }
}

.status-text {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.model-name {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
}
</style>