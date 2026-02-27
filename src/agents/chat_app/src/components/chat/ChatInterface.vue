<script setup lang="ts">
import { computed, ref } from "vue";

import { SkeletonItem } from "@knime/components";

import { useScrollToBottom } from "@/composables/useScrollToBottom";
import { useChatStore } from "@/stores/chat";

import MessageInput from "./MessageInput.vue";
import StatusIndicator from "./StatusIndicator.vue";
import WarningBanner from "./WarningBanner.vue";
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
</script>

<template>
  <main class="chat-interface">
    <div ref="scrollableContainer" class="scrollable-container">
      <div ref="messagesList" class="message-list">
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

    <WarningBanner
      v-if="chatStore.warningMessage"
      :warning="chatStore.warningMessage"
      @dismiss="chatStore.dismissWarning"
    />

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
}
</style>
