<script setup lang="ts">
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import AnimatedEllipsis from "./AnimatedEllipsis.vue";
</script>

<template>
  <div class="tool-use-indicator-container">
    <div class="icon">
      <WrenchIcon />
    </div>
    <span>Using tools<AnimatedEllipsis /></span>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.tool-use-indicator-container {
  font-size: 12px;
  margin-bottom: var(--space-8);
  align-self: flex-start;
  display: flex;
  align-items: center;
}

.icon {
  display: flex;
  background-color: var(--knime-white);
  border: 2px solid var(--knime-porcelain);
  border-radius: 100%;
  padding: var(--space-4);
  margin-right: var(--space-12);

  & :deep(svg) {
    @mixin svg-icon-size 16;

    stroke-width: 1.5;
  }
}
</style>
