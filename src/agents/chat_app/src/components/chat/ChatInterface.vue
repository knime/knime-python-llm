<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";
import { useChatStore } from "@/stores/chat";
import type { ChatItem } from "@/types";
import { renderMarkdown } from "@/utils/markdown";
import {
  extractReferenceTargets,
  extractReferencedMessageIds,
  normalizeMessageId,
} from "@/utils/messageReferences";

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
const previewHtml = computed(() => renderMarkdown(previewText.value));
const selectedMessageId = ref<string | null>(null);
const selectedPanel = ref<"backlinks" | "references" | null>(null);

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

type BacklinkEntry = {
  sourceId: string;
  previewMarkdown: string;
};
type ReferenceEntry = {
  targetHash: string;
  targetMessageId: string;
  previewMarkdown: string;
};
type PanelEntry = {
  id: string;
  previewMarkdown: string;
  type: "backlink" | "reference";
};

type NonTimelineChatItem = Exclude<ChatItem, { type: "timeline" }>;

const isMessageItem = (
  item: ChatItem,
): item is NonTimelineChatItem =>
  item.type !== "timeline";

const renderBacklinkPreview = (markdown: string): string =>
  renderMarkdown(markdown);

const backlinksByTarget = computed(() => {
  const byTarget = new Map<string, BacklinkEntry[]>();

  for (const item of chatStore.chatItems) {
    if (!isMessageItem(item)) {
      continue;
    }
    if (!("content" in item) || typeof item.content !== "string") {
      continue;
    }
    const refs = extractReferencedMessageIds(item.content);
    for (const targetId of refs) {
      const current = byTarget.get(targetId) ?? [];
      current.push({
        sourceId: item.id,
        previewMarkdown: item.content,
      });
      byTarget.set(targetId, current);
    }
  }

  return byTarget;
});

const backlinks = computed(() => {
  if (!selectedMessageId.value) {
    return [] as BacklinkEntry[];
  }
  return backlinksByTarget.value.get(selectedMessageId.value) ?? [];
});

const referencesBySource = computed(() => {
  const bySource = new Map<string, ReferenceEntry[]>();
  const messageById = new Map<string, NonTimelineChatItem>();
  for (const item of chatStore.chatItems) {
    if (isMessageItem(item)) {
      messageById.set(item.id, item);
    }
  }

  for (const item of chatStore.chatItems) {
    if (!isMessageItem(item)) {
      continue;
    }
    if (!("content" in item) || typeof item.content !== "string") {
      continue;
    }
    const targets = extractReferenceTargets(item.content);
    const entries: ReferenceEntry[] = targets.map((targetHash) => {
      const targetMessageId = normalizeMessageId(targetHash);
      const targetMessage = messageById.get(targetMessageId);
      return {
        targetHash,
        targetMessageId,
        previewMarkdown:
          targetMessage && "content" in targetMessage
            ? targetMessage.content
            : "Reference target not found.",
      };
    });
    bySource.set(item.id, entries);
  }

  return bySource;
});

const references = computed(() => {
  if (!selectedMessageId.value) {
    return [] as ReferenceEntry[];
  }
  return referencesBySource.value.get(selectedMessageId.value) ?? [];
});

const backlinkCountByMessageId = computed(() => {
  const counts = new Map<string, number>();
  for (const [targetId, entries] of backlinksByTarget.value) {
    counts.set(targetId, entries.length);
  }
  return counts;
});

const getBacklinkCount = (item: ChatItem): number => {
  if (!isMessageItem(item)) {
    return 0;
  }
  return backlinkCountByMessageId.value.get(item.id) ?? 0;
};

const referenceCountByMessageId = computed(() => {
  const counts = new Map<string, number>();
  for (const [sourceId, entries] of referencesBySource.value) {
    counts.set(sourceId, entries.length);
  }
  return counts;
});

const getReferenceCount = (item: ChatItem): number => {
  if (!isMessageItem(item)) {
    return 0;
  }
  return referenceCountByMessageId.value.get(item.id) ?? 0;
};

const panelTitle = computed(() => {
  if (!selectedMessageId.value) {
    return "";
  }
  if (selectedPanel.value === "references") {
    return `References in ${selectedMessageId.value}`;
  }
  return `Backlinks to ${selectedMessageId.value}`;
});

const panelEntries = computed((): PanelEntry[] => {
  if (selectedPanel.value === "references") {
    return references.value.map((entry) => ({
      id: entry.targetHash,
      previewMarkdown: entry.previewMarkdown,
      type: "reference",
    }));
  }
  return backlinks.value.map((entry) => ({
    id: entry.sourceId,
    previewMarkdown: entry.previewMarkdown,
    type: "backlink",
  }));
});

const panelEmptyText = computed(() =>
  selectedPanel.value === "backlinks"
    ? "No backlinks yet."
    : "No references in this message.",
);

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
  const normalized = normalizeMessageId(decodeURIComponent(hash.slice(1)));
  selectedMessageId.value = normalized || null;
  selectedPanel.value = "backlinks";
  if (updateLocation && window.location.hash !== hash) {
    window.location.hash = hash;
  }
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  clearHighlight();
  target.classList.add(HIGHLIGHT_CLASS);
  highlightedElement = target;
  highlightTimer = window.setTimeout(clearHighlight, HIGHLIGHT_DURATION_MS);
};

const selectMessageFromEvent = (event: MouseEvent) => {
  const clickedElement = event.target as HTMLElement | null;
  const idElement = clickedElement?.closest("[id]");
  const id = idElement?.getAttribute("id");
  if (!id) {
    return;
  }
  const normalized = normalizeMessageId(id);
  const exists = chatStore.chatItems.some(
    (item) => isMessageItem(item) && item.id === normalized,
  );
  if (exists) {
    selectedMessageId.value = normalized;
    selectedPanel.value = "backlinks";
  }
};

const onClickMessageList = (event: MouseEvent) => {
  const link = (event.target as HTMLElement | null)?.closest("a");
  if (!link) {
    selectMessageFromEvent(event);
    return;
  }
  const href = link.getAttribute("href");
  if (!href || !href.startsWith("#")) {
    selectMessageFromEvent(event);
    return;
  }
  event.preventDefault();
  navigateToHash(href, true);
};

const onClickBacklink = (sourceId: string) => {
  navigateToHash(`#${sourceId}`, true);
};

const onClickReference = (targetHash: string) => {
  navigateToHash(`#${targetHash}`, true);
};

const onClickPanelEntry = (entry: PanelEntry) => {
  if (entry.type === "backlink") {
    onClickBacklink(entry.id);
    return;
  }
  onClickReference(entry.id);
};

const onToggleBacklinks = (messageId: string) => {
  if (
    selectedPanel.value === "backlinks" &&
    selectedMessageId.value === messageId
  ) {
    selectedMessageId.value = null;
    selectedPanel.value = null;
    return;
  }
  selectedMessageId.value = messageId;
  selectedPanel.value = "backlinks";
};

const onToggleReferences = (messageId: string) => {
  if (
    selectedPanel.value === "references" &&
    selectedMessageId.value === messageId
  ) {
    selectedMessageId.value = null;
    selectedPanel.value = null;
    return;
  }
  selectedMessageId.value = messageId;
  selectedPanel.value = "references";
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
  return content;
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
          <component
            :is="chatItemComponents[item.type]"
            :backlink-count="getBacklinkCount(item)"
            :reference-count="getReferenceCount(item)"
            v-bind="item"
            @toggle-references="onToggleReferences"
            @toggle-backlinks="onToggleBacklinks"
          />
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

    <aside
      v-if="selectedMessageId && selectedPanel"
      class="backlink-panel"
      data-testid="backlink-panel"
    >
      <div class="backlink-title">{{ panelTitle }}</div>
      <div v-if="panelEntries.length === 0" class="backlink-empty">
        {{ panelEmptyText }}
      </div>
      <button
        v-for="entry in panelEntries"
        :key="entry.id"
        class="backlink-item"
        type="button"
        @click="onClickPanelEntry(entry)"
      >
        <div class="backlink-item-id">{{ entry.id }}</div>
        <!-- eslint-disable-next-line vue/no-v-html -->
        <div class="backlink-item-preview" v-html="renderBacklinkPreview(entry.previewMarkdown)" />
      </button>
    </aside>
    <div
      v-if="previewTargetId"
      class="reference-preview"
      :style="{ top: `${previewTop}px`, left: `${previewLeft}px` }"
      data-testid="reference-preview"
    >
      <div class="preview-title">{{ previewTargetId }}</div>
      <!-- eslint-disable-next-line vue/no-v-html -->
      <div class="preview-content" v-html="previewHtml" />
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
  max-height: 180px;
  overflow: hidden;
  white-space: normal;

  :deep() {
    & > *:first-child {
      margin-top: 0;
    }

    & > *:last-child {
      margin-bottom: 0;
    }
  }
}

.backlink-panel {
  position: absolute;
  top: var(--space-24);
  right: var(--space-24);
  width: 280px;
  max-height: 320px;
  overflow-y: auto;
  background: var(--knime-white);
  border: 1px solid var(--knime-silver-sand);
  border-radius: var(--space-4);
  box-shadow: var(--knime-shadow-level-2);
  padding: var(--space-8);
  z-index: 900;
}

.backlink-title {
  font-size: 12px;
  font-weight: 600;
  margin-bottom: var(--space-8);
}

.backlink-empty {
  font-size: 12px;
}

.backlink-item {
  width: 100%;
  text-align: left;
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: var(--space-4);
  background: var(--knime-white);
  margin-bottom: var(--space-8);
  padding: var(--space-8);
  cursor: pointer;
}

.backlink-item-id {
  font-size: 11px;
  font-weight: 600;
  margin-bottom: var(--space-4);
}

.backlink-item-preview {
  font-size: 12px;
  line-height: 1.3;
  max-height: 80px;
  overflow: hidden;

  :deep() {
    & > *:first-child {
      margin-top: 0;
    }

    & > *:last-child {
      margin-bottom: 0;
    }
  }
}
</style>
