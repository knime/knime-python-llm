<script setup lang="ts">
import { defineProps, defineEmits } from 'vue';

const props = defineProps<{
  isDarkMode: boolean;
}>();

const emit = defineEmits<{
  (e: 'toggleDarkMode'): void;
  (e: 'close'): void;
}>();
</script>

<template>
  <div class="settings-panel">
    <div class="settings-header">
      <h2 class="settings-title">Settings</h2>
      <button class="close-button" @click="emit('close')" aria-label="Close settings">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
    
    <div class="settings-content">
      <div class="setting-item">
        <div class="setting-label">
          <div class="setting-name">Dark Mode</div>
          <div class="setting-description">Switch between light and dark themes</div>
        </div>
        <button 
          class="toggle-button" 
          :class="{ active: isDarkMode }"
          @click="emit('toggleDarkMode')"
          aria-label="Toggle dark mode"
        >
          <span class="toggle-slider"></span>
        </button>
      </div>
      
      <slot></slot>
    </div>
  </div>
</template>

<style scoped>
.settings-panel {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 350px;
  background-color: var(--color-bg-secondary);
  border-left: 1px solid var(--color-border);
  box-shadow: -5px 0 25px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  z-index: 100;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--color-border);
}

.settings-title {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 0;
}

.close-button {
  background: transparent;
  border: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  padding: 8px;
  border-radius: 8px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-button:hover {
  background-color: rgba(0, 0, 0, 0.05);
  color: var(--color-text-primary);
}

.dark-mode .close-button:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

.settings-content {
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

.setting-label {
  display: flex;
  flex-direction: column;
}

.setting-name {
  font-weight: 500;
  margin-bottom: 4px;
}

.setting-description {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
}

.toggle-button {
  position: relative;
  width: 50px;
  height: 24px;
  border-radius: 12px;
  background-color: var(--color-gray-300);
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.toggle-button.active {
  background-color: var(--color-primary);
}

.toggle-slider {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background-color: white;
  transition: transform 0.3s ease;
}

.toggle-button.active .toggle-slider {
  transform: translateX(26px);
}

@media (max-width: 576px) {
  .settings-panel {
    width: 100%;
  }
}
</style>