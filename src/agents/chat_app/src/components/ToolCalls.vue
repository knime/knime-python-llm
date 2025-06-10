<script setup lang="ts">
import {
  type ComputedRef,
  computed,
  defineProps,
  markRaw,
  reactive,
} from "vue";

import WrenchIcon from "@knime/styles/img/icons/wrench.svg";
import { Tree, type TreeNodeOptions } from "@knime/virtual-tree";

import type { ToolCall } from "../types";

const props = defineProps<{
  toolCalls: ToolCall[];
}>();

const visibilityMap = reactive({});

const treeSource: ComputedRef<TreeNodeOptions[]> = computed(() =>
  props.toolCalls.map((toolCall) => ({
    nodeKey: toolCall.id,
    name: toolCall.name,
    icon: markRaw(WrenchIcon),
    hasChildren: true,
    children: [{ nodeKey: `${toolCall.id}-args`, name: toolCall.args ?? "" }],
  })),
);

props.toolCalls.forEach((message) => {
  visibilityMap[message.id] = false;
});
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
  padding: 8px 12px;
}
</style>
