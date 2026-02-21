<script setup lang="ts">
import { Tooltip } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";

import type { ErrorMessage } from "@/types";

import MessageBox from "./MessageBox.vue";

defineProps<ErrorMessage>();
</script>

<template>
  <MessageBox :anchor-id="id" is-error>
    <template #icon>
      <Tooltip text="AI">
        <AiIcon />
      </Tooltip>
    </template>
    {{ content }}
  </MessageBox>
</template>
