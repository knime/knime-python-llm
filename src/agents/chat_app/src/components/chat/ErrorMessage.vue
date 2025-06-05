<script setup lang="ts">
defineProps<{
  content: string;
}>();
</script>

<template>
  <div class="error-message">
    {{ content }}
  </div>
</template>

<style scoped>
.error-message {
  text-align: center;
  font-size: 13px;
  color: var(--knime-white);
  margin: var(--knime-space-8) 0;
  padding: var(--knime-space-4) var(--knime-space-12);
  border-radius: var(--knime-space-16);
  background-color: var(--knime-dove-gray);
  align-self: center;
  animation: fade-in 0.3s ease;
}
</style>
