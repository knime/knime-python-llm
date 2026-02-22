<script setup lang="ts">
import { computed } from "vue";

import { renderMarkdown } from "../../utils/markdown";

const props = defineProps<{
  backlinkCount?: number;
  referenceCount?: number;
  markdown: string;
  messageId?: string;
}>();

const emit = defineEmits<{
  toggleReferences: [messageId: string];
  toggleBacklinks: [messageId: string];
}>();

const htmlContent = computed(() => renderMarkdown(props.markdown, props.messageId));

const onToggleBacklinks = () => {
  if (props.messageId) {
    emit("toggleBacklinks", props.messageId);
  }
};

const onToggleReferences = () => {
  if (props.messageId) {
    emit("toggleReferences", props.messageId);
  }
};
</script>

<template>
  <!-- eslint-disable-next-line vue/no-v-html -->
  <div class="content" v-html="htmlContent" />
  <div v-if="messageId" class="citation-chips">
    <button
      class="chip"
      type="button"
      @click.stop="onToggleReferences"
    >
      References ({{ referenceCount ?? 0 }})
    </button>
    <button class="chip backlink-chip" type="button" @click.stop="onToggleBacklinks">
      Backlinks ({{ backlinkCount ?? 0 }})
    </button>
  </div>
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

.citation-chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-4);
  margin-top: var(--space-8);
}

.chip {
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: 999px;
  background: var(--knime-white);
  font-size: 11px;
  padding: 2px var(--space-8);
  cursor: pointer;
}

.backlink-chip {
  font-weight: 600;
}
</style>
