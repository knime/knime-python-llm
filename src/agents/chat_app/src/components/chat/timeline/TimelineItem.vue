<script setup lang="ts">
import { computed, ref } from "vue";

import { Pill } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";
import ArrowNextIcon from "@knime/styles/img/icons/arrow-next.svg";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { TimelineItem } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

const PREVIEW_LENGTH_FOR_OVERLAY = 80;
const CONTENT_LENGTH_FOR_PREVIEW = 150;

const props = defineProps<{
  item: TimelineItem;
}>();

const isExpanded = ref(false);

// determine what to render based on the item (ai reasoning vs tool call)
const iconComponent = computed(() =>
  props.item.type === "reasoning" ? AiIcon : WrenchIcon,
);
const name = computed(() =>
  props.item.type === "reasoning" ? "AI" : props.item.name,
);
const pillLabel = computed(() => {
  if (props.item.type === "reasoning") {
    return "Reasoning";
  }

  return props.item.status.charAt(0).toUpperCase() + props.item.status.slice(1);
});
const pillVariant = computed(() => {
  if (props.item.type === "reasoning") {
    return "info";
  }

  if (props.item.status === "completed") {
    return "success";
  }

  if (props.item.status === "failed") {
    return "error";
  }

  return "info";
});

// determine content
const hasArgs = computed(() => {
  return props.item.type === "tool_call" && Boolean(props.item.args);
});
const hasContent = computed(() => Boolean(props.item.content));
const isExpandable = computed(() => hasArgs.value || hasContent.value);

// determine preview (we only preview content, not args or anything else)
const hasPreview = computed(() =>
  Boolean(props.item.content && props.item.content?.trim().length > 0),
);
const previewNeedsOverlay = computed(() => {
  if (!props.item.content) {
    return false;
  }

  return props.item.content?.length > PREVIEW_LENGTH_FOR_OVERLAY;
});
const previewText = computed(() => {
  const contentText = props.item.content || "";
  return (
    contentText.substring(0, CONTENT_LENGTH_FOR_PREVIEW) +
    (contentText.length > CONTENT_LENGTH_FOR_PREVIEW ? "..." : "")
  );
});
</script>

<template>
  <div class="timeline-item-container">
    <!-- clickable header -->
    <div
      class="header"
      :class="{ clickable: isExpandable }"
      @click="isExpandable && (isExpanded = !isExpanded)"
    >
      <!-- icon -->
      <div class="icon">
        <component :is="iconComponent" />
      </div>

      <!-- label and status pill -->
      <div class="details">
        <span class="label">{{ name }}</span>
        <Pill :variant="pillVariant" class="status-pill">{{ pillLabel }}</Pill>
      </div>

      <!-- expansion chevron -->
      <div
        v-if="isExpandable"
        class="expansion-chevron"
        :class="{ 'is-expanded': isExpanded }"
      >
        <ArrowNextIcon />
      </div>
    </div>

    <!-- preview -->
    <div
      v-if="!isExpanded && isExpandable && hasPreview"
      class="preview-container"
      :class="{ 'with-overlay': previewNeedsOverlay }"
    >
      <div class="preview-text">{{ previewText }}</div>
      <div v-if="previewNeedsOverlay" class="preview-fade" />
    </div>

    <!-- full content -->
    <div v-if="isExpanded && isExpandable" class="expandable-content">
      <!-- tool call args -->
      <div
        v-if="item.type === 'tool_call' && item.args"
        class="content-section"
      >
        <pre>{{ item.args }}</pre>
      </div>

      <!-- text content -->
      <div v-if="hasContent" class="content-section">
        <MarkdownRenderer :markdown="item.content!" />
      </div>
    </div>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.timeline-item-container {
  position: relative;
  padding-left: var(--space-24);
}

.header {
  display: flex;
  align-items: flex-start;
  padding: 8px;
  border-radius: var(--border-radius-s);
  transition: background-color 0.15s ease;

  &.clickable {
    cursor: pointer;

    &:hover {
      background-color: var(--knime-gray-light-semi);
    }
  }

  /* Timeline dot */
  &::before {
    content: "";
    position: absolute;
    left: 12px;
    top: 14px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--knime-dove-gray);
  }

  .icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    margin-top: 2px;
    margin-right: var(--space-8);
    color: var(--knime-steel-gray);
    flex-shrink: 0;

    & :deep(svg) {
      @mixin svg-icon-size 16;
    }
  }

  .details {
    display: flex;
    align-items: baseline;
    gap: var(--space-8);
    flex: 1;

    & .status-pill {
      font-size: 11px;
    }

    .label {
      font-weight: 500;
      color: var(--knime-masala);
      word-break: break-all;
    }
  }

  .expansion-chevron {
    width: 16px;
    height: 16px;
    margin-right: var(--space-8);
    color: var(--knime-silver-sand);
    transition: transform 0.2s ease;
    flex-shrink: 0;

    &.is-expanded {
      transform: rotate(90deg);
    }

    & :deep(svg) {
      @mixin svg-icon-size 16;

      stroke-width: 1.5;
    }
  }
}

.preview-container {
  position: relative;
  padding-top: var(--space-8);
  padding-right: var(--space-24);
  padding-left: var(--space-8);

  &.with-overlay {
    max-height: 60px;
    overflow: hidden;
  }

  .preview-text {
    font-size: 12px;
    color: var(--knime-steel-gray);
  }

  .preview-fade {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 12px;
    background: linear-gradient(transparent, var(--knime-white));
    pointer-events: none;
  }
}

.expandable-content {
  padding-top: var(--space-8);
  padding-right: var(--space-24);
  padding-left: var(--space-8);
  display: flex;
  flex-direction: column;
  gap: var(--space-8);

  .content-section pre {
    background: var(--knime-gray-light-semi);
    border: 1px solid var(--knime-silver-sand);
    border-radius: var(--border-radius-s);
    font-family: "Roboto Mono", monospace;
    font-size: 12px;
    padding: var(--space-8) var(--space-12);
    white-space: pre-wrap;
    word-break: break-all;
  }
}
</style>
