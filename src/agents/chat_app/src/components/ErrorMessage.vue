<script setup lang="ts">
import type { ErrorResponse } from "@/types";

defineProps<ErrorResponse>();
</script>

<template>
  <div class="error-message">
    {{ content }}
  </div>
</template>

<style scoped>
.error-message {
  text-align: center;
  font-size: 0.825rem;
  color: var(--knime-white);
  margin: 8px 0;
  padding: 4px 12px;
  border-radius: 16px;
  background-color: var(--knime-dove-gray);
  align-self: center;
  animation: fade-in 0.3s ease;
}
</style>
