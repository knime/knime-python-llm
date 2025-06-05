<script setup lang="ts">
import UserIcon from "@knime/styles/img/icons/user.svg";

import type { HumanResponse } from "@/types";

import MarkdownRenderer from "./chat/MarkdownRenderer.vue";
import MessageBox from "./chat/MessageBox.vue";

defineProps<HumanResponse>();
</script>

<template>
  <MessageBox :is-user="true">
    <template #icon>
      <UserIcon />
    </template>
    <MarkdownRenderer v-if="content" :markdown="content" />
  </MessageBox>
</template>
