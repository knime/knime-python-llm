import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";

import MessageInput from "../MessageInput.vue";

describe("MessageInput", () => {
  it("focuses textarea on mount", () => {
    const wrapper = mount(MessageInput, {
      props: { isLoading: false },
      attachTo: document.body,
    });
    const textarea = wrapper.find("textarea");

    expect(textarea.element).toBe(document.activeElement);
  });

  it("focuses textarea when clicking on the container but not on textarea", async () => {
    const wrapper = mount(MessageInput, {
      props: { isLoading: false },
      attachTo: document.body,
    });
    const textarea = wrapper.find("textarea");

    await wrapper.find(".chat-controls").trigger("click");

    expect(textarea.element).toBe(document.activeElement);
  });

  it("disables send button when empty or loading", async () => {
    const wrapper = mount(MessageInput, { props: { isLoading: false } });
    const sendButton = wrapper.find(".send-button");

    // Initially empty -> disabled
    expect(sendButton.attributes("disabled")).toBeDefined();

    // Set input value to non-empty
    await wrapper.find("textarea").setValue("Hello");
    expect(sendButton.attributes("disabled")).toBeUndefined();

    // Set isLoading to true disables button
    await wrapper.setProps({ isLoading: true });
    expect(sendButton.attributes("disabled")).toBeDefined();
  });

  it("emits sendMessage and clears input on send button click", async () => {
    const wrapper = mount(MessageInput, { props: { isLoading: false } });
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Test message");
    await wrapper.find(".send-button").trigger("click");

    expect(wrapper.emitted("sendMessage")).toBeTruthy();
    expect(wrapper.emitted("sendMessage")![0]).toEqual(["Test message"]);
    expect(textarea.element.value).toBe("");
  });

  it("does not emit sendMessage on click if send button is disabled", async () => {
    const wrapper = mount(MessageInput, { props: { isLoading: true } });
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Message");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });

    expect(wrapper.emitted("sendMessage")).toBeFalsy();
    expect(textarea.element.value).toContain("Message");
  });

  it("emits sendMessage and clears input on Enter key press without Shift", async () => {
    const wrapper = mount(MessageInput, { props: { isLoading: false } });
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Enter message");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });

    expect(wrapper.emitted("sendMessage")).toBeTruthy();
    expect(wrapper.emitted("sendMessage")![0]).toEqual(["Enter message"]);
    expect(textarea.element.value).toBe("");
  });

  it("does not emit sendMessage on Enter if shift key is pressed", async () => {
    const wrapper = mount(MessageInput, { props: { isLoading: false } });
    const textarea = wrapper.find("textarea");

    await textarea.setValue("Shift+Enter message");

    const preventDefault = vi.fn();
    await textarea.trigger("keydown", {
      key: "Enter",
      shiftKey: true,
      preventDefault,
    });

    expect(wrapper.emitted("sendMessage")).toBeFalsy();
    expect(preventDefault).not.toHaveBeenCalled();
    expect(textarea.element.value).toContain("Shift+Enter message");
  });
});
