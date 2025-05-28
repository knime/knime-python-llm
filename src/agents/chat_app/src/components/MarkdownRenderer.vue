<script setup lang="ts">
import { computed } from "vue";

import { renderMarkdown } from "./markdown";

interface Props {
  markdown: string;
}
const props = defineProps<Props>();
const htmlContent = computed(() => renderMarkdown(props.markdown));
</script>

<template>
  <div class="content" v-html="htmlContent" />
</template>

<style lang="postcss" scoped>
.content :deep() {
  overflow-wrap: break-word;
  overflow-x: hidden;

  & > *:first-child {
    margin-top: 0;
  }

  & > *:last-child {
    margin-bottom: 0;
  }

  & h1 {
    font-size: 1.5em;
  }

  & h2,
  & h3,
  & h4,
  & h5,
  & h6 {
    font-size: 1em;
  }

  & pre {
    border: 1px solid var(--knime-silver-sand);
    padding: var(--space-4);
  }

  & code {
    white-space: pre-wrap;
    word-break: break-all;
  }

  & ol {
    counter-reset: list-counter;

    & li {
      counter-increment: list-counter;

      &::before {
        content: counter(list-counter) ".";
      }
    }
  }

  & ul,
  & ol {
    list-style: none;
    padding-left: 0;

    & li {
      margin-bottom: 0.4em;

      &::before {
        font-weight: bold;
        margin-right: 5px;
      }

      & p:first-child {
        display: inline;
      }
    }

    & ul,
    & ol {
      counter-reset: list-counter;
      padding-left: var(--space-16);
      margin-top: 0.4em;
    }
  }

  & ul {
    & li {
      &::before {
        content: "\2022";
      }
    }
  }
}
</style>
