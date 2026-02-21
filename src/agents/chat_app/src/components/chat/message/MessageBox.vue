<script setup lang="ts">
import { defineProps, useSlots } from "vue";

withDefaults(
  defineProps<{
    anchorId?: string;
    isUser?: boolean;
    isError?: boolean;
    isNodeView?: boolean;
  }>(),
  {
    anchorId: undefined,
    isUser: false,
    isError: false,
    isNodeView: false,
  },
);

const slots = useSlots();
const hasNameSlot = slots.name;
const hasIconSlot = slots.icon;
</script>

<template>
  <div
    :id="anchorId"
    class="wrapper"
    :class="{
      user: isUser,
      error: isError,
      'node-view': isNodeView,
      'with-name': hasNameSlot,
    }"
  >
    <div class="message-box">
      <div v-if="hasIconSlot || hasNameSlot" class="header">
        <div class="icon">
          <slot name="icon" />
        </div>
        <slot name="name" />
      </div>
      <div class="body">
        <slot />
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

.message-box {
  min-width: 50px;
  max-width: 60%;
  position: relative;
  font-size: 13px;
  font-weight: 400;

  .node-view & {
    min-width: 80%;
    aspect-ratio: 2 / 1;
  }
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
  padding: var(--space-4);

  .with-name & {
    border-radius: 16px;
    padding-right: var(--space-8);
  }

  .user & {
    left: unset;
    right: 0;
  }
}

.icon {
  height: var(--space-16);
  width: var(--space-16);

  .with-name & {
    margin-right: var(--space-4);
  }

  & :deep(svg) {
    @mixin svg-icon-size 16;
  }
}

.body {
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: 0 var(--space-4) var(--space-4);
  background-color: var(--knime-white);
  padding: var(--space-12) var(--space-8);
  height: 100%;

  .user & {
    border-radius: var(--space-4) 0 var(--space-4) var(--space-4);
  }

  .error & {
    background-color: var(--knime-coral-light);
    border-color: var(--knime-coral-light);
  }

  .node-view & {
    padding-bottom: 0;
    padding-top: 0;
  }
}
</style>
