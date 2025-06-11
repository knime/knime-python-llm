import { markRaw } from "vue";

import WrenchIcon from "@knime/styles/img/icons/wrench.svg";
import type { TreeNodeOptions } from "@knime/virtual-tree";

import type { ToolCall } from "@/types";

export const toolCallToTreeNode = (toolCall: ToolCall): TreeNodeOptions => ({
  nodeKey: toolCall.id,
  name: toolCall.name,
  icon: markRaw(WrenchIcon),
  hasChildren: true,
  children: [{ nodeKey: `${toolCall.id}-args`, name: toolCall.args ?? "" }],
});
