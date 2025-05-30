<script setup lang="ts">
import { onMounted } from "vue";

import { JsonDataService } from "@knime/ui-extension-service";

import ChatInterface from "./components/ChatInterface.vue";

onMounted(async () => {
  // initializes the python backend (e.g. starting the python process)
  const jsonDataService = await JsonDataService.getInstance();
  jsonDataService.data({ method: "init" });
});
</script>

<template>
  <div class="app-container">
    <ChatInterface />
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--knime-white);
  color: var(--knime-masala);
}
</style>
