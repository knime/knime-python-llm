import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

import type { ViewMessage } from "@/types";
import MessageBox from "../chat/message/MessageBox.vue";
import NodeViewMessage from "../chat/message/NodeViewMessage.vue";
import { useChatStore } from "@/stores/chat";
import { version } from "os";

const mockCallKnimeUiApi = vi.fn(() =>
  Promise.resolve({
    isSome: true,
    result: { resourceInfo: { path: "resource" } },
  }),
);

const mockJsonDataService = {
  data: vi.fn(),
  applyData: vi.fn(),
  initialData: vi.fn(),
  baseService: { callKnimeUiApi: mockCallKnimeUiApi },
};

vi.mock("@knime/ui-extension-service", () => ({
  AlertingService: {
    getInstance: vi.fn(() =>
      Promise.resolve({ baseService: { sendAlert: vi.fn() } }),
    ),
  },
  JsonDataService: {
    getInstance: vi.fn(() => Promise.resolve(mockJsonDataService)),
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
  setActivePinia(
    createTestingPinia({
      createSpy: vi.fn,
      stubActions: false,
    }),
  );

  const store = useChatStore();
  mockJsonDataService.data.mockImplementation(() =>
    Promise.resolve({
      project_id: "test_project_id",
      workflow_id: "test_workflow_id",
    }),
  );
  store.jsonDataService = mockJsonDataService;

  const doMount = (props: Partial<ViewMessage> = {}) => {
    const defaultProps = {
      id: "1",
      content: "test_node_id",
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

  it("does not render UIExtension if data is not available", async () => {
    mockCallKnimeUiApi.mockImplementationOnce(() =>
      Promise.reject(new Error("fail")),
    );

    const { wrapper } = doMount();

    await flushPromises();

    expect(wrapper.findComponent({ name: "UIExtension" }).exists()).toBe(false);
  });

  it("resolves node ID correctly", async () => {
    doMount();

    await flushPromises();

    expect(mockJsonDataService.data).toHaveBeenCalledWith({
      method: "get_combined_tools_workflow_info",
    });
    expect(mockCallKnimeUiApi).toHaveBeenCalledWith("NodeService.getNodeView", {
      nodeId: "test_workflow_id:test_node_id",
      projectId: "test_project_id",
      workflowId: "test_workflow_id",
      versionId: "current-state",
    });
  });

  it("shows error when resolving node ID fails", async () => {
    store.jsonDataService = mockJsonDataService;
    mockJsonDataService.data.mockImplementationOnce(() =>
      Promise.resolve(undefined),
    );
    const { wrapper } = doMount();

    await flushPromises();

    expect(mockJsonDataService.data).toHaveBeenCalledWith({
      method: "get_combined_tools_workflow_info",
    });
    expect(wrapper.text()).toContain(
      "View can't be rendered. Error resolving node ID. Debug workflow deleted?",
    );
  });
});
