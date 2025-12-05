<script setup lang="ts">
import { computed, ref } from "vue";
import { useTextareaAutosize } from "@vueuse/core";

import { FunctionButton } from "@knime/components";
import SendIcon from "@knime/styles/img/icons/paper-flier.svg";
import AbortIcon from "@knime/styles/img/icons/close.svg";

import { useChatStore } from "@/stores/chat";

const chatStore = useChatStore();
const { textarea, input } = useTextareaAutosize();

const characterLimit = 5000;

const isInputValid = computed(() => input.value?.trim().length > 0);
const isDisabled = computed(() => (!isInputValid.value && !chatStore.isLoading) || chatStore.isInterrupted);
const isProcessing = computed(() => chatStore.isLoading);


const handleClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget) {
    textarea.value?.focus();
  }
};

const handleSubmit = () => {
  if (isProcessing.value) {
    chatStore.cancelAgent();
  } else {
    chatStore.sendUserMessage(input.value);
    input.value = "";
  }
};

const handleKeyDown = (event: KeyboardEvent) => {
  // enter: send message
  if (event.key === "Enter" && !isProcessing.value && !event.shiftKey && !isDisabled.value) {
    event.preventDefault();
    handleSubmit();
  }

  // arrow up: recall last user message
  if (
    event.key === "ArrowUp" &&
    input.value === "" &&
    chatStore.lastUserMessage
  ) {
    input.value = chatStore.lastUserMessage;
  }
};
</script>

<template>
  <div class="chat-controls" @click="handleClick">
    <textarea
      ref="textarea"
      v-model="input"
      class="textarea"
      aria-label="Type your message"
      :maxlength="characterLimit"
      @keydown="handleKeyDown"
    />
    <FunctionButton
      class="send-button"
      primary
      :disabled="isDisabled"
      @click="handleSubmit"
    >
      <AbortIcon
        v-if="isProcessing"
        class="abort-icon"
        aria-hidden="true"
        focusable="false"
      />
      <SendIcon
        v-else
        class="send-icon"
        aria-hidden="true"
        focusable="false"
      />
    </FunctionButton>
  </div>
</template>

<style lang="postcss" scoped>
@import url("@knime/styles/css/mixins");

.chat-controls {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  max-height: 200px;
  min-height: 50px;
  background-color: white;
  border: 1px solid var(--knime-stone-gray);
  overflow: hidden;
  cursor: text;

  & .textarea {
    font-size: 13px;
    font-weight: 300;
    line-height: 150%;
    padding: 10px 8px 0;
    flex-grow: 1;
    width: 100%;
    resize: none;
    border: none;

    &:focus {
      outline: none;
    }
  }

  & .send-button {
    align-self: flex-end;
    margin-right: 8px;
    margin-bottom: 8px;

    & svg {
      stroke: var(--knime-dove-gray);

      & .send-icon .abort-icon {
        margin-left: -1px;
      }
    }
  }
}
</style>
