<script setup lang="ts">
import { defineProps, reactive } from "vue";
import VueJsonPretty from "vue-json-pretty";

import { Collapser } from "@knime/components";

import type { ToolCall } from "../types";

import MarkdownRenderer from "./MarkdownRenderer.vue";

const props = defineProps<{
  toolCalls: ToolCall[];
}>();

// Create a reactive object to store booleans keyed by ID
const visibilityMap = reactive({});

// Initialize visibility dynamically
props.toolCalls.forEach((message) => {
  visibilityMap[message.id] = false;
});

// Optional helper to toggle visibility
const toggleVisibility = (id: string, value: boolean) => {
  visibilityMap[id] = value;
};

const getJson = (str: string): boolean => {
  try {
    const parsed = JSON.parse(str);
    // Optional: Ensure the parsed result is an object or array
    return typeof parsed === "object" && parsed !== null && parsed;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (e) {
    return false;
  }
};
</script>

<template>
  <div class="tool-calls">
    <Collapser
      v-for="call in toolCalls"
      :key="call.id"
      v-model="visibilityMap[call.id]"
      class="collapser"
      @update:model-value="(value: boolean) => toggleVisibility(call.id, value)"
    >
      <template #title>
        <h5>Tool: {{ call.name }}</h5>
      </template>
      <div v-if="call.args" class="tool-call-content">
        Call arguments:
        <VueJsonPretty v-if="getJson(call.args)" :data="getJson(call.args)" />
        <MarkdownRenderer v-else :markdown="call.args" />
      </div>
      <div v-else>No arguments</div>
    </Collapser>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");
@import url("vue-json-pretty/lib/styles.css");

.tool-calls {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.collapser {
  background-color: var(--knime-porcelain);
  width: 350px;

  & :deep(h5) {
    padding: var(--space-12) var(--space-24);
    margin: 0;
    position: relative;
  }
}

.tool-call-content {
  padding: var(--space-12) var(--space-24);
}
</style>
