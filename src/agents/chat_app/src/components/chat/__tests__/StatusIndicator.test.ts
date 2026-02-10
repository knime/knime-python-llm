import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import StatusIndicator from "../StatusIndicator.vue";

describe("StatusIndicator", () => {
  it("renders the label prop correctly", () => {
    const wrapper = mount(StatusIndicator, {
      props: {
        label: "Test Label",
      },
      global: {
        stubs: {
          WrenchIcon: true,
          AnimatedEllipsis: true,
        },
      },
    });

    expect(wrapper.text()).toContain("Test Label");
  });
});
