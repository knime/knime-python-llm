<script setup lang="ts">
import { Tooltip } from "@knime/components";
import AiIcon from "@knime/styles/img/icons/ai-general.svg";

import type { ErrorResponse } from "@/types";

import MessageBox from "./MessageBox.vue";

defineProps<ErrorResponse>();
</script>

<template>
  <MessageBox is-error>
    <template #icon>
      <Tooltip text="AI">
        <AiIcon />
      </Tooltip>
    </template>
    {{ content }}
  </MessageBox>
</template>
