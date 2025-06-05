<script setup lang="ts">
import { type ComputedRef, computed, defineProps, reactive } from "vue";
import VueJsonPretty from "vue-json-pretty";

import { Tree, type TreeNodeOptions } from "@knime/virtual-tree";
import "vue-json-pretty/lib/styles.css";

import type { ToolCall } from "../types";

import MarkdownRenderer from "./chat/MarkdownRenderer.vue";

const props = defineProps<{
  toolCalls: ToolCall[];
}>();

const visibilityMap = reactive({});

// TODO: Remove
const mockJsonString = '{"name":"John", "age":30, "car":null}';

const treeSource: ComputedRef<TreeNodeOptions[]> = computed(() =>
  props.toolCalls.map((toolCall) => ({
    nodeKey: toolCall.id,
    name: toolCall.name,
    hasChildren: true,
    children: [{ nodeKey: `${toolCall.id}-args`, name: toolCall.args ?? "" }],
  })),
);

props.toolCalls.forEach((message) => {
  visibilityMap[message.id] = false;
});

const getJson = (str: string): boolean => {
  try {
    const parsed = JSON.parse(str);
    return typeof parsed === "object" && parsed !== null && parsed;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (e) {
    return false;
  }
};
</script>

<template>
  <div class="tool-calls">
    <Tree :source="treeSource" :selectable="false">
      <template #leaf="{ treeNode }">
        <template v-if="treeNode.name">
          <VueJsonPretty
            v-if="getJson(mockJsonString)"
            :data="getJson(mockJsonString)"
            show-line-number
          />
          <MarkdownRenderer v-else :markdown="treeNode.name" />
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
}
</style>
