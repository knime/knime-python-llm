<script setup lang="ts">
import { Tooltip } from "@knime/components";
import UserIcon from "@knime/styles/img/icons/user.svg";

import type { HumanMessage } from "@/types";
import MarkdownRenderer from "../MarkdownRenderer.vue";

import MessageBox from "./MessageBox.vue";

defineProps<HumanMessage>();
</script>

<template>
  <MessageBox :is-user="true">
    <template #icon>
      <Tooltip text="User">
        <UserIcon />
      </Tooltip>
    </template>
    <MarkdownRenderer v-if="content" :markdown="content" />
  </MessageBox>
</template>
