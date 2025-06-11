import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import ChatInterface from "../ChatInterface.vue";

describe("ChatInterface", () => {
  it("renders slot content inside message list", () => {
    const slotContent = "<p>Test</p>";

    const wrapper = mount(ChatInterface, {
      slots: { default: slotContent },
      props: { isLoading: false },
    });

    expect(wrapper.find(".message-list").html()).toContain(slotContent);
  });

  it("renders loading skeleton inside MessageBox when isLoading is true", () => {
    const wrapper = mount(ChatInterface, { props: { isLoading: true } });
    const messageBox = wrapper.find(".message-box");

    expect(messageBox.exists()).toBe(true);
    expect(messageBox.find(".skeleton-item").exists()).toBe(true);
  });

  it("does not render loading skeleton when isLoading is false", () => {
    const wrapper = mount(ChatInterface, { props: { isLoading: false } });

    expect(wrapper.find(".skeleton-item").exists()).toBe(false);
  });

  it("emits sendMessage event when MessageInput emits send-message", async () => {
    const MessageInputStub = {
      template: "<input @send-message=\"$emit('send-message', $event)\" />",
      name: "MessageInput",
    };
    const wrapper = mount(ChatInterface, {
      global: { stubs: { MessageInput: MessageInputStub } },
      props: { isLoading: false },
    });

    const input = wrapper.findComponent(MessageInputStub);
    await input.vm.$emit("send-message", "Message");

    expect(wrapper.emitted("sendMessage")).toBeTruthy();
    expect(wrapper.emitted("sendMessage")![0]).toEqual(["Message"]);
  });
});
