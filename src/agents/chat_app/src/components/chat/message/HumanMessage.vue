<script setup lang="ts">
import { Tooltip } from "@knime/components";
import UserIcon from "@knime/styles/img/icons/user.svg";

import type { HumanMessage } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

import MessageBox from "./MessageBox.vue";

defineProps<HumanMessage & { backlinkCount?: number }>();

const emit = defineEmits<{
  navigateRef: [hash: string];
  toggleBacklinks: [messageId: string];
}>();
</script>

<template>
  <MessageBox :anchor-id="id" :is-user="true">
    <template #icon>
      <Tooltip text="User">
        <UserIcon />
      </Tooltip>
    </template>
    <MarkdownRenderer
      v-if="content"
      :backlink-count="backlinkCount"
      :markdown="content"
      :message-id="id"
      @navigate-ref="emit('navigateRef', $event)"
      @toggle-backlinks="emit('toggleBacklinks', $event)"
    />
  </MessageBox>
</template>
