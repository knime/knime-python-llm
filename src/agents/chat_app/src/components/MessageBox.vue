<script setup lang="ts">
import { computed, defineProps } from "vue";

import KnimeIcon from "@knime/styles/img/KNIME_Triangle.svg";
import UserIcon from "@knime/styles/img/icons/user.svg";

import type { Type } from "../types";

import MarkdownRenderer from "./MarkdownRenderer.vue";
import MessagePlaceholder from "./MessagePlaceholder.vue";

const props = defineProps<{
  content?: string;
  type: Type;
}>();

const isHuman = computed(() => {
  return props.type === "human";
});
</script>

<template>
  <div class="message">
    <!-- Message's sender icon -->
    <div class="header">
      <div class="icon" :class="{ user: isHuman }">
        <UserIcon v-if="isHuman" />
        <KnimeIcon v-else />
      </div>
    </div>

    <!-- Message content -->
    <!-- TODO: Error logic -->
    <div class="body" :class="{ user: isHuman, error: false }">
      <MarkdownRenderer v-if="props.content" :markdown="props.content" />
      <MessagePlaceholder v-else />
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
      height: 26px;
      width: 26px;
      display: flex;
      justify-content: center;
      align-items: center;

      & svg {
        margin-top: -2px;

        @mixin svg-icon-size 16;
      }

      &.user svg.assistant {
        margin-top: -4px;
      }
    }
  }

  & .body {
    border: 1px solid var(--knime-silver-sand);
    border-radius: 0 5px 5px;
    background-color: var(--knime-white);
    padding: 10px 8px;

    &.user {
      border-radius: 5px 0 5px 5px;
    }

    &.error {
      background-color: var(--knime-coral-light);
    }
  }

  & .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 25px;

    & .footer-left {
      flex: 1;
      text-align: left;
    }

    & .footer-right {
      text-align: right;
    }

    & .show-full-content-button {
      all: unset;
      cursor: pointer;
      font-weight: 500;
      font-size: 11px;
      padding-top: 10px;
      color: var(--knime-dove-grey);
      margin-top: -5px;
      margin-left: 2px;
    }

    & .show-full-content-button:active,
    & .show-full-content-button:hover {
      text-decoration: underline;
    }
  }
}
</style>
