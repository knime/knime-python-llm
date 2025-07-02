import { describe, expect, it } from "vitest";
import { nextTick } from "vue";
import { mount } from "@vue/test-utils";

import type { ToolCall } from "@/types";
import ToolCalls from "../ToolCalls.vue";

const sampleToolCalls = [
  { id: "1", name: "Calculator", args: "1 + 1" },
  { id: "2", name: "Calculator", args: "" },
  { id: "3", name: "Calculator" },
];

describe("ToolCalls", () => {
  const doMount = async (toolCalls: ToolCall[]) => {
    const wrapper = mount(ToolCalls, { props: { toolCalls } });
    await nextTick();
    return wrapper;
  };

  it("renders correctly when arguments are provided", async () => {
    const wrapper = await doMount(sampleToolCalls);

    // Open a tool call with args
    await wrapper.findAll(".vir-tree-node")[0].trigger("click");

    // Args rendered in a <pre> element
    expect(wrapper.find("pre").text()).toContain(sampleToolCalls[0].args);
    expect(wrapper.text()).not.toContain("No arguments");
  });

  it("renders correctly when arguments are an empty string", async () => {
    const wrapper = await doMount(sampleToolCalls);

    // Open a tool call with an empty string args prop
    await wrapper.findAll(".vir-tree-node")[1].trigger("click");

    expect(wrapper.text()).toContain("No arguments");
  });

  it("renders correctly when arguments are undefined", async () => {
    const wrapper = await doMount(sampleToolCalls);

    // Open a tool call with an undefined args prop
    await wrapper.findAll(".vir-tree-node")[2].trigger("click");

    expect(wrapper.text()).toContain("No arguments");
  });
});
