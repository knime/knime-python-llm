<script setup lang="ts">
import { Tooltip } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";

import type { AiMessage } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

import MessageBox from "./MessageBox.vue";

defineProps<AiMessage & { backlinkCount?: number; referenceCount?: number }>();

const emit = defineEmits<{
  toggleReferences: [messageId: string];
  toggleBacklinks: [messageId: string];
}>();
</script>

<template>
  <MessageBox :anchor-id="id">
    <template #icon>
      <Tooltip text="AI"> <AiIcon /> </Tooltip>
    </template>
    <MarkdownRenderer
      v-if="content"
      :backlink-count="backlinkCount"
      :markdown="content"
      :message-id="id"
      :reference-count="referenceCount"
      @toggle-references="emit('toggleReferences', $event)"
      @toggle-backlinks="emit('toggleBacklinks', $event)"
    />
  </MessageBox>
</template>
