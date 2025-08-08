import { describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";

import type { ViewMessage } from "@/types";
import MessageBox from "../chat/message/MessageBox.vue";
import NodeViewMessage from "../chat/message/NodeViewMessage.vue";

const mockCallKnimeUiApi = vi.fn(() =>
  Promise.resolve({
    isSome: true,
    result: { resourceInfo: { path: "resource" } },
  }),
);

vi.mock("@knime/ui-extension-service", () => ({
  AlertingService: {
    getInstance: vi.fn(() =>
      Promise.resolve({ baseService: { sendAlert: vi.fn() } }),
    ),
  },
  JsonDataService: {
    getInstance: vi.fn(() =>
      Promise.resolve({ baseService: { callKnimeUiApi: mockCallKnimeUiApi } }),
    ),
  },
  ResourceService: {
    getInstance: vi.fn(() =>
      Promise.resolve({
        getResourceUrl: vi.fn((path) => {
          return Promise.resolve(`/base/${path}`);
        }),
      }),
    ),
  },
}));

describe("NodeViewMessage", () => {
  const doMount = (props: Partial<ViewMessage> = {}) => {
    const defaultProps = {
      id: "1",
      content: "projectId#workflowId:nodeId",
      type: "view" as const,
      name: "Test View",
    };

    mockCallKnimeUiApi.mockClear();
    mockCallKnimeUiApi.mockImplementation(() =>
      Promise.resolve({
        isSome: true,
        result: { resourceInfo: { path: "resource" } },
      }),
    );

    const wrapper = mount(NodeViewMessage, {
      props: { ...defaultProps, ...props },
      global: {
        stubs: {
          UIExtension: {
            name: "UIExtension",
            template: "<div class='ui-extension-mock'></div>",
            props: [
              "apiLayer",
              "extensionConfig",
              "resourceLocation",
              "shadowAppStyle",
            ],
          },
        },
      },
    });

    return { wrapper };
  };

  it("renders UIExtension when data is available", async () => {
    const { wrapper } = doMount();

    await flushPromises();

    expect(wrapper.findComponent({ name: "UIExtension" }).exists()).toBe(true);
    expect(wrapper.find(".ui-extension-mock").exists()).toBe(true);
    expect(wrapper.findComponent(MessageBox).exists()).toBe(true);
  });

  it("passes correct props to UIExtension", async () => {
    const { wrapper } = doMount();

    await flushPromises();

    const uiExtension = wrapper.findComponent({ name: "UIExtension" });

    expect(typeof uiExtension.props("apiLayer")).toBe("object");
    expect(uiExtension.props("extensionConfig")).not.toBeNull();
    expect(uiExtension.props("resourceLocation")).toContain("/base/resource");
    expect(uiExtension.props("shadowAppStyle")).toEqual({
      height: "100%",
      width: "100%",
      overflowX: "scroll",
    });
  });

  it("shows error message if renderError is set", async () => {
    const { wrapper } = doMount({
      content: "not-a-virtual-project#brokenNode",
    });

    await flushPromises();

    expect(wrapper.text()).toContain(
      "View for node brokenNode can't be rendered",
    );
    expect(wrapper.findComponent({ name: "UIExtension" }).exists()).toBe(false);
  });

  it("does not render UIExtension if data is not available", async () => {
    mockCallKnimeUiApi.mockImplementationOnce(() =>
      Promise.reject(new Error("fail")),
    );

    const { wrapper } = doMount();

    await flushPromises();

    expect(wrapper.findComponent({ name: "UIExtension" }).exists()).toBe(false);
  });
});
