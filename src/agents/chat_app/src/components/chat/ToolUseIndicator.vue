<script setup lang="ts">
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import AnimatedEllipsis from "./AnimatedEllipsis.vue";
</script>

<template>
  <div class="tool-use-indicator-container">
    <div class="icon-wrapper">
      <WrenchIcon />
    </div>
    <span class="label">Using tools<AnimatedEllipsis /></span>
  </div>
</template>

<style scoped>
.tool-use-indicator-container {
  font-size: 12px;
  width: 50%;
  min-width: 320px;
  margin-bottom: var(--space-12);
  align-self: flex-start;
  display: flex;
  align-items: center;
  padding-top: 10px;
  padding-bottom: 10px;
}

.icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--knime-white);
  border: 2px solid var(--knime-porcelain);
  border-radius: 50%;
  padding: var(--space-4);
  margin-right: var(--space-12);
  flex-shrink: 0;
}

.icon-wrapper :deep(svg) {
  width: 16px;
  height: 16px;
  color: var(--knime-steel-gray);
  stroke-width: 1.5;
}

.label {
  color: var(--knime-steel-gray);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
