<script setup lang="ts">
import { computed, nextTick, ref } from "vue";

import { FunctionButton } from "@knime/components";
import SendIcon from "@knime/styles/img/icons/paper-flier.svg";

const characterLimit = 300;

const props = defineProps<{
  isLoading: boolean;
  sendMessage: (message: string) => void;
}>();

const userInput = ref("");
const inputElement = ref<HTMLTextAreaElement | null>(null);

const isInputValid = computed(
  () => userInput.value && userInput.value.trim().length > 0,
);
const disabled = computed(() => !isInputValid.value || props.isLoading);

const handleSubmit = () => {
  // TODO: Handle canceling?
  if (props.isLoading || !userInput.value.trim()) {
    return;
  }

  props.sendMessage(userInput.value);

  userInput.value = "";

  // Reset height after sending
  nextTick(() => {
    if (inputElement.value) {
      inputElement.value.style.height = "auto";
    }
  });
};

const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === "Enter" && !event.shiftKey && !disabled.value) {
    event.preventDefault();
    handleSubmit();
  }
};
</script>

<template>
  <div class="chat-controls">
    <textarea
      v-model="userInput"
      class="textarea"
      aria-label="Type your message"
      :maxlength="characterLimit"
      placeholder=""
      @keydown="handleKeydown"
    />
    <FunctionButton
      class="send-button"
      primary
      :disabled="disabled"
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
  min-height: 120px;
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
    box-sizing: border-box;

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
