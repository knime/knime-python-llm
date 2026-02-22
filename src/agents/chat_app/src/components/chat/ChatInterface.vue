<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";
import { useChatStore } from "@/stores/chat";

import MessageInput from "./MessageInput.vue";
import StatusIndicator from "./StatusIndicator.vue";
import AiMessage from "./message/AiMessage.vue";
import ErrorMessage from "./message/ErrorMessage.vue";
import HumanMessage from "./message/HumanMessage.vue";
import MessageBox from "./message/MessageBox.vue";
import NodeViewMessage from "./message/NodeViewMessage.vue";
import ExpandableTimeline from "./timeline/ExpandableTimeline.vue";

const chatStore = useChatStore();

const scrollableContainer = ref<HTMLElement | null>(null);
const messagesList = ref<HTMLElement | null>(null);

const statusIndicatorLabel = computed(() =>
  chatStore.isInterrupted ? "Cancelling" : "Using tools",
);

const chatItemComponents = {
  ai: AiMessage,
  view: NodeViewMessage,
  human: HumanMessage,
  error: ErrorMessage,
  timeline: ExpandableTimeline,
};

useScrollToBottom(scrollableContainer, messagesList);

const HIGHLIGHT_CLASS = "link-target-highlight";
const HIGHLIGHT_DURATION_MS = 1800;
let highlightedElement: HTMLElement | null = null;
let highlightTimer: number | undefined;

const previewTargetId = ref<string | null>(null);
const previewText = ref<string>("");
const previewTop = ref(0);
const previewLeft = ref(0);
const PREVIEW_OFFSET_PX = 12;
const MAX_PREVIEW_CHARS = 220;

const clearHighlight = () => {
  if (highlightedElement) {
    highlightedElement.classList.remove(HIGHLIGHT_CLASS);
    highlightedElement = null;
  }
  if (highlightTimer !== undefined) {
    window.clearTimeout(highlightTimer);
    highlightTimer = undefined;
  }
};

const resolveFragmentTarget = (hash: string): HTMLElement | null => {
  if (!hash || !hash.startsWith("#")) {
    return null;
  }
  const targetId = decodeURIComponent(hash.slice(1));
  if (!targetId) {
    return null;
  }
  return document.getElementById(targetId);
};

const navigateToHash = (hash: string, updateLocation: boolean) => {
  const target = resolveFragmentTarget(hash);
  if (!target) {
    return;
  }
  if (updateLocation && window.location.hash !== hash) {
    window.location.hash = hash;
  }
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  clearHighlight();
  target.classList.add(HIGHLIGHT_CLASS);
  highlightedElement = target;
  highlightTimer = window.setTimeout(clearHighlight, HIGHLIGHT_DURATION_MS);
};

const onClickMessageList = (event: MouseEvent) => {
  const link = (event.target as HTMLElement | null)?.closest("a");
  if (!link) {
    return;
  }
  const href = link.getAttribute("href");
  if (!href || !href.startsWith("#")) {
    return;
  }
  event.preventDefault();
  navigateToHash(href, true);
};

const extractTargetMessageId = (href: string): string | null => {
  if (!href.startsWith("#")) {
    return null;
  }
  const hashTarget = decodeURIComponent(href.slice(1));
  if (!hashTarget) {
    return null;
  }
  return hashTarget.split("__")[0] || null;
};

const getMessagePreview = (messageId: string) => {
  const item = chatStore.chatItems.find(
    (chatItem) => chatItem.type !== "timeline" && chatItem.id === messageId,
  );
  if (!item || !("content" in item)) {
    return null;
  }
  const content = item.content?.trim();
  if (!content) {
    return null;
  }
  return (
    content.slice(0, MAX_PREVIEW_CHARS) +
    (content.length > MAX_PREVIEW_CHARS ? "..." : "")
  );
};

const hidePreview = () => {
  previewTargetId.value = null;
  previewText.value = "";
};

const updatePreviewPosition = (event: MouseEvent) => {
  previewLeft.value = event.clientX + PREVIEW_OFFSET_PX;
  previewTop.value = event.clientY + PREVIEW_OFFSET_PX;
};

const onMouseOverMessageList = (event: MouseEvent) => {
  const link = (event.target as HTMLElement | null)?.closest("a");
  if (!link) {
    hidePreview();
    return;
  }
  const href = link.getAttribute("href");
  if (!href) {
    hidePreview();
    return;
  }
  const targetMessageId = extractTargetMessageId(href);
  if (!targetMessageId) {
    hidePreview();
    return;
  }
  const preview = getMessagePreview(targetMessageId);
  if (!preview) {
    hidePreview();
    return;
  }

  previewTargetId.value = targetMessageId;
  previewText.value = preview;
  updatePreviewPosition(event);
};

const onMouseMoveMessageList = (event: MouseEvent) => {
  if (!previewTargetId.value) {
    return;
  }
  updatePreviewPosition(event);
};

const onMouseOutMessageList = (event: MouseEvent) => {
  const related = event.relatedTarget as HTMLElement | null;
  if (related?.closest("a")) {
    return;
  }
  hidePreview();
};

const onHashChange = () => {
  navigateToHash(window.location.hash, false);
};

const onEscapePreview = (event: KeyboardEvent) => {
  if (event.key === "Escape") {
    hidePreview();
  }
};

const onScrollHidePreview = () => {
  hidePreview();
};

onMounted(() => {
  window.addEventListener("hashchange", onHashChange);
  window.addEventListener("keydown", onEscapePreview);
  window.addEventListener("scroll", onScrollHidePreview, true);
  if (window.location.hash) {
    navigateToHash(window.location.hash, false);
  }
});

onUnmounted(() => {
  window.removeEventListener("hashchange", onHashChange);
  window.removeEventListener("keydown", onEscapePreview);
  window.removeEventListener("scroll", onScrollHidePreview, true);
  clearHighlight();
  hidePreview();
});
</script>

<template>
  <main class="chat-interface">
    <div ref="scrollableContainer" class="scrollable-container">
      <div
        ref="messagesList"
        class="message-list"
        @click="onClickMessageList"
        @mouseover="onMouseOverMessageList"
        @mousemove="onMouseMoveMessageList"
        @mouseout="onMouseOutMessageList"
      >
        <template v-for="item in chatStore.chatItems" :key="item.id">
          <component :is="chatItemComponents[item.type]" v-bind="item" />
        </template>

        <StatusIndicator
          v-if="chatStore.shouldShowStatusIndicator"
          :label="statusIndicatorLabel"
        />

        <MessageBox v-if="chatStore.shouldShowGenericLoadingIndicator">
          <SkeletonItem height="24px" width="200px" />
        </MessageBox>
      </div>
    </div>
    <div
      v-if="previewTargetId"
      class="reference-preview"
      :style="{ top: `${previewTop}px`, left: `${previewLeft}px` }"
      data-testid="reference-preview"
    >
      <div class="preview-title">{{ previewTargetId }}</div>
      <div class="preview-content">{{ previewText }}</div>
    </div>

    <MessageInput />
  </main>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.chat-interface {
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  position: relative;
  overflow-y: hidden;
  padding: var(--space-24);
  width: 100%;
}

.scrollable-container {
  flex: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
  scrollbar-gutter: stable;
}

.message-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-24);
  padding: var(--space-24) 0;

  :deep(.link-target-highlight) {
    outline: 2px solid var(--knime-masala);
    outline-offset: 4px;
    transition: outline-color 0.25s ease;
  }
}

.reference-preview {
  position: fixed;
  z-index: 1000;
  max-width: 320px;
  background: var(--knime-white);
  border: 1px solid var(--knime-silver-sand);
  border-radius: var(--space-4);
  padding: var(--space-8);
  box-shadow: var(--knime-shadow-level-2);
  pointer-events: none;
}

.preview-title {
  font-size: 12px;
  font-weight: 600;
  margin-bottom: var(--space-4);
}

.preview-content {
  font-size: 12px;
  line-height: 1.4;
  white-space: pre-wrap;
}
</style>
