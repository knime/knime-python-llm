<script setup lang="ts">
import { computed, defineProps } from "vue";

import AiIcon from "@knime/styles/img/icons/ai-general.svg";
import UserIcon from "@knime/styles/img/icons/user.svg";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { ToolCall, Type } from "../types";

import MarkdownRenderer from "./MarkdownRenderer.vue";
import MessagePlaceholder from "./MessagePlaceholder.vue";
import ToolCalls from "./ToolCalls.vue";

const props = defineProps<{
  name?: string | null;
  content: string;
  type: Type;
  toolCalls?: ToolCall[];
}>();

const isHuman = computed(() => props.type === "human");
</script>

<template>
  <div class="message">
    <!-- Message's sender icon -->
    <div class="header">
      <div class="icon" :class="{ human: isHuman }">
        <UserIcon v-if="isHuman" />
        <WrenchIcon v-else-if="type === 'tool'" />
        <AiIcon v-else />
      </div>
    </div>

    <!-- Message content -->
    <div class="body" :class="{ human: isHuman, error: false }">
      <div v-if="type === 'tool'">{{ name }}:</div>
      <ToolCalls v-if="toolCalls" :tool-calls="toolCalls" />
      <MarkdownRenderer v-if="content" :markdown="content" />
      <MessagePlaceholder v-if="!content && !toolCalls?.length" />
    </div>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.message {
  position: relative;
  width: 100%;
  font-size: 13px;
  font-weight: 400;

  & .header {
    position: absolute;
    left: 0;
    top: -21px;
    height: 21px;
    width: 100%;
    display: flex;
    justify-content: flex-end;
    align-items: center;

    & .icon {
      position: absolute;
      top: 0;
      left: 0;
      background-color: var(--knime-white);
      border: 2px solid var(--knime-porcelain);
      border-radius: 100%;
      height: var(--space-24);
      width: var(--space-24);
      display: flex;
      justify-content: center;
      align-items: center;

      & svg {
        margin-top: -2px;

        @mixin svg-icon-size 16;
      }
    }
  }

  & .body {
    border: 1px solid var(--knime-silver-sand);
    border-radius: 0 var(--space-4) var(--space-4);
    background-color: var(--knime-white);
    padding: var(--space-12) var(--space-8);

    &.human {
      border-radius: var(--space-4) var(--space-4) 0 var(--space-4);
    }

    &.error {
      background-color: var(--knime-coral-light);
    }
  }
}
</style>
