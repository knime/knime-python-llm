<script setup lang="ts">
import { defineProps, defineEmits } from 'vue';
import { type Model } from '../types';

const props = defineProps<{
  models: Model[];
  selectedModel: Model;
}>();

const emit = defineEmits<{
  (e: 'select', model: Model): void;
}>();

const selectModel = (model: Model) => {
  emit('select', model);
};
</script>

<template>
  <div class="model-selector">
    <h2 class="selector-title">Select AI Model</h2>
    <div class="models-list">
      <button
        v-for="model in models"
        :key="model.id"
        class="model-item"
        :class="{ active: model.id === selectedModel.id }"
        @click="selectModel(model)"
      >
        <div class="model-info">
          <div class="model-name">{{ model.name }}</div>
          <div class="model-description">{{ model.description }}</div>
        </div>
        <div class="selection-indicator" v-if="model.id === selectedModel.id">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
        </div>
      </button>
    </div>
  </div>
</template>

<style scoped>
.model-selector {
  padding: 16px 0;
}

.selector-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 16px 0;
  padding: 0 16px;
}

.models-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-radius: 8px;
  background-color: transparent;
  border: none;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.2s ease;
  color: var(--color-text-primary);
}

.model-item:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.dark-mode .model-item:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.model-item.active {
  background-color: rgba(59, 130, 246, 0.1);
}

.dark-mode .model-item.active {
  background-color: rgba(59, 130, 246, 0.2);
}

.model-info {
  display: flex;
  flex-direction: column;
}

.model-name {
  font-weight: 500;
  margin-bottom: 4px;
}

.model-description {
  font-size: 0.85rem;
  color: var(--color-text-secondary);
}

.selection-indicator {
  color: var(--color-primary);
}
</style>