<script setup lang="ts">
import { SkeletonItem } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";
import UserIcon from "@knime/styles/img/icons/user.svg";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { MessageResponse } from "@/types";

import ErrorMessage from "./ErrorMessage.vue";
import MarkdownRenderer from "./MarkdownRenderer.vue";
import MessageBox from "./MessageBox.vue";
import ToolCalls from "./ToolCalls.vue";

defineProps<{
  messages: MessageResponse[];
  isLoading: boolean;
}>();
</script>

<template>
  <div class="message-list">
    <template v-for="message in messages" :key="message.id">
      <ErrorMessage v-if="message.type === 'error'" v-bind="message" />

      <!-- AI Message -->
      <MessageBox v-if="message.type === 'ai'">
        <template #icon>
          <AiIcon />
        </template>
        <ToolCalls
          v-if="message.toolCalls?.length"
          :tool-calls="message.toolCalls"
        />
        <MarkdownRenderer v-if="message.content" :markdown="message.content" />
      </MessageBox>

      <!-- Tool Message -->
      <MessageBox v-if="message.type === 'tool'">
        <template #icon>
          <WrenchIcon />
        </template>
        <template #name>
          {{ message.name }}
        </template>
        <MarkdownRenderer v-if="message.content" :markdown="message.content" />
      </MessageBox>

      <!-- Human Message -->
      <MessageBox v-if="message.type === 'human'" :is-user="true">
        <template #icon>
          <UserIcon />
        </template>
        <MarkdownRenderer v-if="message.content" :markdown="message.content" />
      </MessageBox>
    </template>

    <!-- Loading new messages -->
    <MessageBox v-if="isLoading">
      <template #icon>
        <AiIcon />
      </template>
      <SkeletonItem height="24px" />
    </MessageBox>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.message-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-24);
  padding: 0 var(--space-16);

  & svg {
    margin-top: -2px;

    @mixin svg-icon-size 16;
  }
}
</style>
