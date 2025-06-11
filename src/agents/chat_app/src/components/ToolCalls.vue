<script setup lang="ts">
import { computed, defineProps } from "vue";

import { Tree } from "@knime/virtual-tree";

import { toolCallToTreeNode } from "@/utils/toolCalls";
import type { ToolCall } from "../types";

const props = defineProps<{ toolCalls: ToolCall[] }>();

const treeSource = computed(() => props.toolCalls.map(toolCallToTreeNode));
</script>

<template>
  <div class="tool-calls">
    <Tree :source="treeSource" :selectable="false">
      <template #leaf="{ treeNode }">
        <template v-if="treeNode.name">
          <pre>{{ treeNode.name }}</pre>
        </template>
        <template v-else>No arguments</template>
      </template>
    </Tree>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.tool-calls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);

  & :deep(.vir-tree-node:has(.node-arrow:empty)) {
    height: auto;

    &:hover {
      background-color: var(--knime-white);
    }
  }

  & :deep(.tree-node.expandable) {
    font-weight: normal;
  }
}

.tool-calls pre {
  background: var(--knime-gray-light-semi);
  border: 1px solid var(--knime-silver-sand);
  font-family: "Roboto Mono", monospace;
  padding: var(--space-8) var(--space-12);
}
</style>
