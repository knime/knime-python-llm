<script setup lang="ts">
import { computed } from "vue";
import { useTextareaAutosize } from "@vueuse/core";

import { FunctionButton } from "@knime/components";
import SendIcon from "@knime/styles/img/icons/paper-flier.svg";

const characterLimit = 5000;

const props = defineProps<{ isLoading: boolean }>();
const emit = defineEmits<{ sendMessage: [message: string] }>();

const { textarea, input } = useTextareaAutosize();

const isInputValid = computed(() => input.value?.trim().length > 0);
const isDisabled = computed(() => !isInputValid.value || props.isLoading);

const handleClick = (event: MouseEvent) => {
  if (event.target === event.currentTarget) {
    textarea.value?.focus();
  }
};

const handleSubmit = () => {
  emit("sendMessage", input.value);
  input.value = "";
};

const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === "Enter" && !event.shiftKey && !isDisabled.value) {
    event.preventDefault();
    handleSubmit();
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
      <SendIcon class="send-icon" aria-hidden="true" focusable="false" />
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

      &.send-icon {
        margin-left: -1px;
      }
    }
  }
}
</style>
