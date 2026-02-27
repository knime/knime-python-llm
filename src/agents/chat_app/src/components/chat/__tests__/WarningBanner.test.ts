import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";

import WarningBanner from "../WarningBanner.vue";

const createWrapper = (content = "Warning text") =>
  mount(WarningBanner, {
    props: {
      warning: {
        id: "warning-1",
        type: "warning",
        content,
      },
    },
    global: {
      stubs: {
        AbortIcon: { template: "<svg />" },
      },
    },
  });

describe("WarningBanner", () => {
  it("renders warning content", () => {
    const wrapper = createWrapper("Heads up");

    expect(wrapper.text()).toContain("Heads up");
  });

  it("emits dismiss when dismiss button is clicked", async () => {
    const wrapper = createWrapper();

    await wrapper.find(".dismiss-button").trigger("click");

    expect(wrapper.emitted("dismiss")).toHaveLength(1);
  });
});
