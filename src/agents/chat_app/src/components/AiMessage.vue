<script setup lang="ts">
import AiIcon from "@knime/styles/img/icons/ai-general.svg";

import type { AiResponse } from "@/types";

import ToolCalls from "./ToolCalls.vue";
import MarkdownRenderer from "./chat/MarkdownRenderer.vue";
import MessageBox from "./chat/MessageBox.vue";

defineProps<AiResponse>();
</script>

<template>
  <MessageBox>
    <template #icon>
      <AiIcon />
    </template>
    <ToolCalls v-if="toolCalls?.length" :tool-calls="toolCalls" />
    <MarkdownRenderer v-if="content" :markdown="content" />
  </MessageBox>
</template>
