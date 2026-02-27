<script setup lang="ts">
import { FunctionButton } from "@knime/components";
import AbortIcon from "@knime/styles/img/icons/close.svg";

import type { Warning } from "@/types";

defineProps<{ warning: Warning }>();

defineEmits<{ (event: "dismiss"): void }>();
</script>

<template>
  <div class="warning-banner">
    <span class="message">{{ warning.content }}</span>
    <FunctionButton
      class="dismiss-button"
      aria-label="Dismiss warning"
      @click="$emit('dismiss')"
    >
      <AbortIcon aria-hidden="true" focusable="false" />
    </FunctionButton>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.warning-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-12);
  margin: var(--space-8) 0 var(--space-16);
  padding: var(--space-12) var(--space-16);
  border-radius: var(--space-8);
  border: 1px solid var(--knime-yellow);
  background-color: var(--knime-yellow-ultra-light);
  color: var(--knime-masala);
  font-size: 13px;
}

.message {
  flex: 1;
}

.dismiss-button :deep(svg) {
  stroke: var(--knime-masala);

  @mixin svg-icon-size 16;
}
</style>
