<script setup lang="ts">
import { computed, defineProps, useSlots } from "vue";

withDefaults(defineProps<{ isUser?: boolean }>(), {
  isUser: false,
});

const slots = useSlots();
const hasNameSlot = computed(() => (slots.name?.().length ?? 0) > 0);
const hasIconSlot = computed(() => (slots.icon?.().length ?? 0) > 0);
</script>

<template>
  <div class="wrapper" :class="{ user: isUser, 'has-name': hasNameSlot }">
    <div class="container">
      <div class="message">
        <div v-if="hasIconSlot || hasNameSlot" class="header">
          <div class="icon">
            <slot name="icon" />
          </div>
          <div v-if="hasNameSlot" class="name">
            <slot name="name" />
          </div>
        </div>

        <div class="body">
          <slot />
        </div>
      </div>
    </div>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.wrapper {
  display: flex;
  margin-bottom: var(--space-8);

  &.user {
    justify-content: flex-end;
  }
}

.container {
  min-width: 20%;
  max-width: 80%;
}

.message {
  position: relative;
  width: 100%;
  font-size: 13px;
  font-weight: 400;
}

.header {
  position: absolute;
  left: 0;
  top: calc(var(--space-24) * -1);
  display: flex;
  align-items: center;
  background-color: var(--knime-white);
  border: 2px solid var(--knime-porcelain);
  border-radius: 100%;

  .has-name & {
    border-radius: 16px;
  }

  .user & {
    left: unset;
    right: 0;
  }
}

.icon {
  height: var(--space-24);
  width: var(--space-24);
  display: flex;
  justify-content: center;
  align-items: center;
}

.name {
  margin-right: var(--space-4);
}

.body {
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: 0 var(--space-4) var(--space-4);
  background-color: var(--knime-white);
  padding: var(--space-12) var(--space-8);

  .user & {
    border-radius: var(--space-4) var(--space-4) 0 var(--space-4);
  }
}
</style>
