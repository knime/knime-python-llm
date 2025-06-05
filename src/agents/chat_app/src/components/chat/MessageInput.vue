<script setup lang="ts">
import { computed, ref } from "vue";

import { FunctionButton, TextArea } from "@knime/components";
import SendIcon from "@knime/styles/img/icons/paper-flier.svg";

const props = defineProps<{ isLoading: boolean }>();
const emit = defineEmits<{ sendMessage: [message: string] }>();

const userInput = ref("");

const isInputValid = computed(() => userInput.value?.trim().length > 0);
const isDisabled = computed(() => !isInputValid.value || props.isLoading);

const handleSubmit = () => {
  emit("sendMessage", userInput.value);
  userInput.value = "";
};

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === "Enter" && !event.shiftKey && !isDisabled.value) {
    event.preventDefault();
    handleSubmit();
  }
};
</script>

<template>
  <div class="chat-controls">
    <TextArea
      v-model="userInput"
      class="textarea-wrapper"
      @keydown="handleKeydown"
    />
    <FunctionButton
      class="send-button"
      primary
      :disabled="isDisabled"
      @click="handleSubmit"
    >
      <SendIcon class="send-icon" aria-hidden="true" focusable="false" />
    </FunctionButton>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.chat-controls {
  position: relative;
  align-items: flex-end;
  height: 120px;
  background-color: var(--knime-white);
  overflow: hidden;
}

.textarea-wrapper {
  max-width: 100%;
  height: 100%;

  & :deep(textarea) {
    font-size: 13px;
    font-weight: 300;
    line-height: 1.5;
    padding: 10px 8px 0;
    width: 100%;
    height: 100%;
    resize: none;
  }
}

.send-button {
  position: absolute;
  right: 8px;
  bottom: 8px;
}
</style>
