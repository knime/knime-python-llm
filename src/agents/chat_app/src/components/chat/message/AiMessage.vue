<script setup lang="ts">
import { Tooltip } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";

import type { AiMessage } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

import MessageBox from "./MessageBox.vue";

defineProps<AiMessage>();
</script>

<template>
  <MessageBox>
    <template #icon>
      <Tooltip text="AI"> <AiIcon /> </Tooltip>
    </template>
    <MarkdownRenderer v-if="content" :markdown="content" />
  </MessageBox>
</template>
