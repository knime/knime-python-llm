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
    <!-- header -->
    <div class="timeline-header-container" @click="isExpanded = !isExpanded">
      <!-- icon and label -->
      <div class="timeline-header">
        <div class="icon"><WrenchIcon /></div>
        <span>{{ label }}<AnimatedEllipsis v-if="status === 'active'" /></span>
      </div>

      <!-- expansion chevron -->
      <div class="expansion-chevron" :class="{ 'is-expanded': isExpanded }">
        <ArrowNextIcon />
      </div>
    </div>

    <!-- body -->
    <div v-if="isExpanded" class="timeline-body">
      <TimelineItem v-for="item in items" :key="item.id" :item="item" />
    </div>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.timeline-container {
  width: 50%;
  min-width: 320px;
  max-width: 1000px;
  margin-bottom: var(--space-12);
  font-size: 12px;
}

.timeline-header-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;

  /* padding: 10px 20px 10px 0; */
  padding-top: var(--space-8);
  padding-bottom: var(--space-8);
  padding-right: var(--space-16);
  transition: background-color 0.15s ease;

  &:hover {
    background: var(--knime-gray-light-semi);
  }

  .timeline-header {
    display: flex;
    align-items: center;
  }

  .icon {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--knime-white);
    border: 2px solid var(--knime-porcelain);
    border-radius: 50%;
    padding: var(--space-4);
    margin-right: var(--space-12);

    & :deep(svg) {
      @mixin svg-icon-size 16;

      color: var(--knime-steel-gray);
      stroke-width: 1.5;
    }
  }

  .expansion-chevron {
    width: 16px;
    height: 16px;
    transition: transform 0.2s ease;

    & :deep(svg) {
      @mixin svg-icon-size 16;

      stroke-width: 1.5;
    }

    &.is-expanded {
      transform: rotate(90deg);
    }
  }
}

.timeline-body {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 10px;

  /* Dashed line representing the timeline */
  &::before {
    content: "";
    position: absolute;
    left: 13px;
    top: 0;
    bottom: 0;
    width: 2px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 5 5'%3E%3Ccircle cx='1' cy='1' r='.5' fill='%236E6E6E'/%3E%3C/svg%3E");
    background-size: 8px;
  }
}
</style>
