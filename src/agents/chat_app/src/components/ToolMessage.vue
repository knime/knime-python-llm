<script setup lang="ts">
import { Tooltip } from "@knime/components";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { ToolResponse } from "@/types";

import MarkdownRenderer from "./chat/MarkdownRenderer.vue";
import MessageBox from "./chat/MessageBox.vue";

defineProps<ToolResponse>();
</script>

<template>
  <MessageBox>
    <template #icon>
      <Tooltip text="Tool">
        <WrenchIcon />
      </Tooltip>
    </template>
    <template #name>
      {{ name }}
    </template>
    <MarkdownRenderer v-if="content" :markdown="content" />
  </MessageBox>
</template>
