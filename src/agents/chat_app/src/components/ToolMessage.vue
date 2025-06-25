<script setup lang="ts">
import { Tooltip } from "@knime/components";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { ToolResponse } from "@/types";

import NodeView from "./NodeView.vue";
import MarkdownRenderer from "./chat/MarkdownRenderer.vue";
import MessageBox from "./chat/MessageBox.vue";

const props = defineProps<ToolResponse>();
const viewNodeIds = props.content.split("View node IDs")[1].split(",");
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
  <MessageBox v-if="viewNodeIds">
    <NodeView :view-node-ids="viewNodeIds" />
  </MessageBox>
</template>
