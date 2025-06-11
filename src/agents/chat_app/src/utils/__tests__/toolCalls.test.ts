import { describe, expect, it } from "vitest";

import { toolCallToTreeNode } from "../toolCalls";

describe("toolCallToTreeNode", () => {
  it("should convert a ToolCall to a TreeNodeOptions object", () => {
    const input = { id: "1", name: "Tool", args: "Args" };

    const result = toolCallToTreeNode(input);

    expect(result).toEqual({
      nodeKey: "1",
      name: "Tool",
      icon: expect.objectContaining({ render: expect.any(Function) }),
      hasChildren: true,
      children: [{ nodeKey: "1-args", name: "Args" }],
    });
  });

  it("should handle undefined args in ToolCall", () => {
    const input = { id: "2", name: "Tool" };

    const result = toolCallToTreeNode(input);

    expect(result).toEqual({
      nodeKey: "2",
      name: "Tool",
      icon: expect.objectContaining({ render: expect.any(Function) }),
      hasChildren: true,
      children: [{ nodeKey: "2-args", name: "" }],
    });
  });
});
