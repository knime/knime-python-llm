<script setup lang="ts">
import { computed, ref } from "vue";

import { Pill } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";
import ArrowNextIcon from "@knime/styles/img/icons/arrow-next.svg";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { TimelineItem } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

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

// determine preview
const hasPreview = computed(() =>
  Boolean(props.item.content && props.item.content?.trim().length > 0),
);
const previewNeedsOverlay = computed(() => {
  if (!props.item.content) {
    return false;
  }

  return props.item.content?.length > 80;
});
const previewText = computed(() => {
  const contentText = props.item.content || "";
  return (
    contentText.substring(0, 150) + (contentText.length > 150 ? "..." : "")
  );
});
</script>

<template>
  <div class="timeline-item-wrapper">
    <div
      class="timeline-item"
      :class="{ clickable: isExpandable }"
      @click="isExpandable && (isExpanded = !isExpanded)"
    >
      <div class="icon-container">
        <component :is="iconComponent" />
      </div>

      <div class="details">
        <span class="name">{{ name }}</span>
        <Pill :variant="pillVariant" class="status-pill">{{ pillLabel }}</Pill>
      </div>

      <div
        v-if="isExpandable"
        class="chevron-container"
        :class="{ 'is-expanded': isExpanded }"
      >
        <ArrowNextIcon />
      </div>
    </div>

    <!-- Preview content when collapsed -->
    <div
      v-if="!isExpanded && isExpandable && hasPreview"
      class="preview-content"
      :class="{ 'with-overlay': previewNeedsOverlay }"
    >
      <div class="preview-text">{{ previewText }}</div>
      <div v-if="previewNeedsOverlay" class="preview-fade" />
    </div>

    <!-- Full content when expanded -->
    <div v-if="isExpanded && isExpandable" class="expandable-content">
      <div
        v-if="item.type === 'tool_call' && item.args"
        class="content-section"
      >
        <pre>{{ item.args }}</pre>
      </div>

      <div v-if="hasContent" class="content-section">
        <MarkdownRenderer :markdown="item.content!" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.timeline-item-wrapper {
  position: relative;
  padding-left: 24px;
}

.timeline-item {
  display: flex;
  align-items: flex-start;
  padding: 8px;
  border-radius: var(--border-radius-s);
  transition: background-color 0.15s ease;
}

.timeline-item.clickable {
  cursor: pointer;
}

.timeline-item.clickable:hover {
  background-color: var(--knime-gray-light-semi);
}

/* Timeline dot */
.timeline-item::before {
  content: "";
  position: absolute;
  left: 12px;
  top: 14px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--knime-dove-gray);
}

.icon-container {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  margin: 1px 8px 0 0;
  color: var(--knime-steel-gray);
  flex-shrink: 0;
}

.icon-container :deep(svg) {
  width: 100%;
  height: 100%;
}

.details {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex: 1;
}

.details :deep(.pill) {
  font-size: 11px;
}

.chevron-container {
  width: 16px;
  height: 16px;
  margin-right: 12px;
  color: var(--knime-silver-sand);
  transition: transform 0.2s ease;
  flex-shrink: 0;
}

.chevron-container.is-expanded {
  transform: rotate(90deg);
}

.chevron-container :deep(svg) {
  width: 100%;
  height: 100%;
  stroke-width: 1.5;
}

.expandable-content {
  padding: 8px 22px 0 8px;
  display: flex;
  flex-direction: column;
  gap: var(--space-8);
}

.preview-content {
  position: relative;
  padding: 8px 22px 0 8px;
}

/* Only limit height and hide overflow if we have a fade overlay */
.preview-content.with-overlay {
  max-height: 60px;
  overflow: hidden;
}

.preview-fade {
  position: absolute;
  bottom: 0;
  left: 8px;
  right: 22px;
  height: 20px;
  background: linear-gradient(transparent, var(--knime-white));
  pointer-events: none;
}

.name {
  font-weight: 500;
  color: var(--knime-masala);
  word-break: break-all;
}

.preview-text {
  font-size: 12px;
  color: var(--knime-steel-gray);
  white-space: pre-wrap;
  word-break: break-all;
}

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
</style>
