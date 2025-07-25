<script setup lang="ts">
import { ref } from "vue";

import ArrowNextIcon from "@knime/styles/img/icons/arrow-next.svg";
import WrenchIcon from "@knime/styles/img/icons/wrench.svg";

import type { Timeline } from "@/types";
import AnimatedEllipsis from "../AnimatedEllipsis.vue";

import TimelineItem from "./TimelineItem.vue";

defineProps<Timeline>();
const isExpanded = ref(false);
</script>

<template>
  <div class="timeline-container">
    <div class="summary" @click="isExpanded = !isExpanded">
      <div class="summary-content">
        <div class="icon-wrapper"><WrenchIcon /></div>
        <span class="summary-text"
          >{{ label }}<AnimatedEllipsis v-if="status === 'active'"
        /></span>
      </div>
      <div class="chevron-container" :class="{ 'is-expanded': isExpanded }">
        <ArrowNextIcon />
      </div>
    </div>
    <div v-if="isExpanded" class="timeline-body">
      <TimelineItem v-for="item in items" :key="item.id" :item="item" />
    </div>
  </div>
</template>

<style scoped>
.timeline-container {
  width: 50%;
  min-width: 320px;
  max-width: 1000px;
  margin-bottom: var(--space-12);
  font-size: 12px;
}

.summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  padding: 10px 20px 10px 0;
  transition: background-color 0.15s ease;
}

.summary:hover {
  background: var(--knime-gray-light-semi);
}

.summary-content {
  display: flex;
  align-items: center;
}

.icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--knime-white);
  border: 2px solid var(--knime-porcelain);
  border-radius: 50%;
  padding: var(--space-4);
  margin-right: var(--space-12);
}

.icon-wrapper :deep(svg) {
  width: 16px;
  height: 16px;
  color: var(--knime-steel-gray);
  stroke-width: 1.5;
}

.chevron-container {
  width: 16px;
  height: 16px;
  transition: transform 0.2s ease;
}

.chevron-container :deep(svg) {
  width: 100%;
  height: 100%;
  stroke-width: 1.5;
}

.chevron-container.is-expanded {
  transform: rotate(90deg);
}

.timeline-body {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.timeline-body::before {
  content: "";
  position: absolute;
  left: 13px;
  top: 0;
  bottom: 0;
  width: 2px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 5 5'%3E%3Ccircle cx='1' cy='1' r='.5' fill='%236E6E6E'/%3E%3C/svg%3E");
  background-size: 8px;
  background-repeat: repeat-y;
}
</style>
