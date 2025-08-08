import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import { useChatStore } from "@/stores/chat";
import MessageInput from "../MessageInput.vue";

describe("MessageInput", () => {
  beforeEach(() => {
    setActivePinia(
      createTestingPinia({
        createSpy: vi.fn,
        stubActions: false,
      }),
    );
  });

  const createWrapper = () => {
    return mount(MessageInput, {
      attachTo: document.body,
    });
  };

  it("doesn't focus textarea on mount", () => {
    const wrapper = createWrapper();
    const textarea = wrapper.find("textarea");

    expect(textarea.element).not.toBe(document.activeElement);
  });

  it("focuses textarea when clicking on the container but not on textarea", async () => {
    const wrapper = createWrapper();
    const textarea = wrapper.find("textarea");

    await wrapper.find(".chat-controls").trigger("click");

    expect(textarea.element).toBe(document.activeElement);
  });

  it("disables send button when empty or loading", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const sendButton = wrapper.find(".send-button");

    // Initially empty -> disabled
    expect(sendButton.attributes("disabled")).toBeDefined();

    // Set input value to non-empty
    await wrapper.find("textarea").setValue("Hello");
    await wrapper.vm.$nextTick();
    expect(sendButton.attributes("disabled")).toBeUndefined();

    // Set store loading to true should disable button
    chatStore.isLoading = true;
    await wrapper.vm.$nextTick();
    expect(sendButton.attributes("disabled")).toBeDefined();
  });

  it("calls store sendUserMessage and clears input on send button click", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const sendUserMessageSpy = vi.spyOn(chatStore, "sendUserMessage").mockResolvedValue();
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Test message");
    await wrapper.find(".send-button").trigger("click");

    expect(sendUserMessageSpy).toHaveBeenCalledWith("Test message");
    expect(textarea.element.value).toBe("");
  });

  it("does not call sendUserMessage when button is disabled due to loading", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const sendUserMessageSpy = vi.spyOn(chatStore, "sendUserMessage").mockResolvedValue();
    
    chatStore.isLoading = true;
    await wrapper.vm.$nextTick();
    
    const textarea = wrapper.find("textarea");
    await textarea.setValue("Message");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });

    expect(sendUserMessageSpy).not.toHaveBeenCalled();
    expect(textarea.element.value).toContain("Message");
  });

  it("calls store sendUserMessage and clears input on Enter key press without Shift", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const sendUserMessageSpy = vi.spyOn(chatStore, "sendUserMessage").mockResolvedValue();
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Enter message");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });

    expect(sendUserMessageSpy).toHaveBeenCalledWith("Enter message");
    expect(textarea.element.value).toBe("");
  });

  it("does not send message on Enter if shift key is pressed", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const sendUserMessageSpy = vi.spyOn(chatStore, "sendUserMessage").mockResolvedValue();
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Shift+Enter message");

    const preventDefault = vi.fn();
    await textarea.trigger("keydown", {
      key: "Enter",
      shiftKey: true,
      preventDefault,
    });

    expect(sendUserMessageSpy).not.toHaveBeenCalled();
    expect(preventDefault).not.toHaveBeenCalled();
    expect(textarea.element.value).toContain("Shift+Enter message");
  });

  it("recalls last user message on ArrowUp when input is empty", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const textarea = wrapper.find("textarea");

    chatStore.lastUserMessage = "Previous message";
    
    // Ensure input is empty first
    await textarea.setValue("");
    
    await textarea.trigger("keydown", { key: "ArrowUp" });

    expect(textarea.element.value).toBe("Previous message");
  });

  it("does not recall last message on ArrowUp when input is not empty", async () => {
    const wrapper = createWrapper();
    const chatStore = useChatStore();
    const textarea = wrapper.find("textarea");

    chatStore.lastUserMessage = "Previous message";
    
    await textarea.setValue("Current message");
    await textarea.trigger("keydown", { key: "ArrowUp" });

    expect(textarea.element.value).toBe("Current message");
  });
});
