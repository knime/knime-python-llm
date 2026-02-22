<script setup lang="ts">
import { computed, ref } from "vue";

import { useChatStore } from "@/stores/chat";
import type { ChatItem } from "@/types";
import { renderMarkdown } from "@/utils/markdown";
import { extractReferencedMessageIds } from "@/utils/messageReferences";

type NonTimelineChatItem = Exclude<ChatItem, { type: "timeline" }>;

type GraphNode = {
  id: string;
  displayId: string;
  fullLabel: string;
  label: string;
  type: NonTimelineChatItem["type"];
  column: number;
  order: number;
  x: number;
  y: number;
};

type GraphEdge = {
  id: string;
  sourceId: string;
  targetId: string;
  path: string;
  isPrimary: boolean;
};

const emit = defineEmits<{
  openMessage: [messageId: string];
}>();

const chatStore = useChatStore();
const selectedNodeId = ref<string | null>(null);
const layoutMode = ref<"horizontal" | "vertical" | "dag">("dag");

const isMessageItem = (item: ChatItem): item is NonTimelineChatItem =>
  item.type !== "timeline";

const laneYByType: Record<NonTimelineChatItem["type"], number> = {
  human: 110,
  ai: 220,
  tool: 330,
  view: 440,
  error: 550,
};

const laneLabels: Array<{ type: NonTimelineChatItem["type"]; label: string }> = [
  { type: "human", label: "Human" },
  { type: "ai", label: "AI" },
  { type: "tool", label: "Tool" },
  { type: "view", label: "View" },
  { type: "error", label: "Error" },
];

const laneOffsetPattern = [0, -28, 28, -56, 56, -84, 84];
const DAG_COLUMN_START = 70;
const DAG_COLUMN_GAP = 64;
const DAG_ROW_START = 68;
const DAG_ROW_GAP = 64;
const DAG_NODE_RADIUS = 11;
const DAG_LABEL_GAP = 84;

const offsetForLaneIndex = (laneIndex: number) =>
  laneOffsetPattern[laneIndex % laneOffsetPattern.length];

const truncateForNode = (value: string, maxLength: number) =>
  value.length > maxLength ? `${value.slice(0, maxLength - 3)}...` : value;

const graphMessages = computed(() =>
  chatStore.chatItems.filter(isMessageItem).filter((item) => Boolean(item.id)),
);

const messageOrderById = computed(
  () => new Map(graphMessages.value.map((item, index) => [item.id, index])),
);

const referenceIdsBySource = computed(() => {
  const map = new Map<string, string[]>();
  for (const source of graphMessages.value) {
    if (!("content" in source) || typeof source.content !== "string") {
      continue;
    }
    map.set(source.id, extractReferencedMessageIds(source.content));
  }
  return map;
});

const primaryParentById = computed(() => {
  const map = new Map<string, string>();
  graphMessages.value.forEach((source, sourceIndex) => {
    const refs = referenceIdsBySource.value.get(source.id) ?? [];
    const olderRefs = refs
      .map((id) => ({ id, order: messageOrderById.value.get(id) ?? -1 }))
      .filter((entry) => entry.order >= 0 && entry.order < sourceIndex)
      .sort((a, b) => b.order - a.order);
    if (olderRefs.length > 0) {
      map.set(source.id, olderRefs[0].id);
    }
  });
  return map;
});

const childrenByPrimaryParent = computed(() => {
  const map = new Map<string, string[]>();
  for (const item of graphMessages.value) {
    const parentId = primaryParentById.value.get(item.id);
    if (!parentId) {
      continue;
    }
    const children = map.get(parentId) ?? [];
    children.push(item.id);
    map.set(parentId, children);
  }
  return map;
});

const dagColumnById = computed(() => {
  const map = new Map<string, number>();
  let nextColumn = 0;

  for (const item of graphMessages.value) {
    const parentId = primaryParentById.value.get(item.id);
    if (!parentId) {
      map.set(item.id, nextColumn);
      nextColumn += 1;
      continue;
    }

    const parentColumn = map.get(parentId);
    const siblings = childrenByPrimaryParent.value.get(parentId) ?? [];
    const isFirstChild = siblings[0] === item.id;

    if (typeof parentColumn === "number" && isFirstChild) {
      map.set(item.id, parentColumn);
      continue;
    }

    map.set(item.id, nextColumn);
    nextColumn += 1;
  }

  return map;
});

const dagColumnCount = computed(() =>
  Math.max(1, ...Array.from(dagColumnById.value.values()).map((value) => value + 1)),
);

const dagColumnX = (column: number) => DAG_COLUMN_START + column * DAG_COLUMN_GAP;
const dagLabelX = computed(
  () => dagColumnX(Math.max(0, dagColumnCount.value - 1)) + DAG_LABEL_GAP,
);

const dagPalette = [
  "#b85652",
  "#b7b84a",
  "#65b85b",
  "#6fbac7",
  "#5a5ec7",
  "#cc7a45",
  "#7d66c7",
];

const colorForColumn = (column: number) => dagPalette[column % dagPalette.length];

const nodes = computed<GraphNode[]>(() => {
  const laneCounts: Record<NonTimelineChatItem["type"], number> = {
    human: 0,
    ai: 0,
    tool: 0,
    view: 0,
    error: 0,
  };

  return graphMessages.value.map((item, index) => {
    const laneIndex = laneCounts[item.type];
    laneCounts[item.type] += 1;
    const column = dagColumnById.value.get(item.id) ?? 0;

    const content = "content" in item && typeof item.content === "string" ? item.content : "";
    const compact = content.replace(/\s+/g, " ").trim();
    const fullLabel = compact || item.id;
    return {
      id: item.id,
      displayId: truncateForNode(item.id, 22),
      fullLabel,
      label: truncateForNode(fullLabel, 24),
      type: item.type,
      column,
      order: index,
      x:
        layoutMode.value === "horizontal"
          ? 180 + index * 220
          : layoutMode.value === "vertical"
            ? laneYByType[item.type] + offsetForLaneIndex(laneIndex)
            : dagColumnX(column),
      y:
        layoutMode.value === "horizontal"
          ? laneYByType[item.type] + offsetForLaneIndex(laneIndex)
          : layoutMode.value === "vertical"
            ? 120 + index * 140
            : DAG_ROW_START + index * DAG_ROW_GAP,
    };
  });
});

const nodeById = computed(() => new Map(nodes.value.map((node) => [node.id, node])));

const edges = computed<GraphEdge[]>(() => {
  const result: GraphEdge[] = [];
  for (const source of graphMessages.value) {
    const sourceNode = nodeById.value.get(source.id);
    if (!sourceNode || !("content" in source) || typeof source.content !== "string") {
      continue;
    }
    const refs = extractReferencedMessageIds(source.content);
    for (const targetId of refs) {
      const targetNode = nodeById.value.get(targetId);
      if (!targetNode) {
        continue;
      }
      const x1 = sourceNode.x;
      const y1 = sourceNode.y;
      const x2 = targetNode.x;
      const y2 = targetNode.y;

      let path = "";
      if (layoutMode.value === "dag") {
        const sx = sourceNode.x;
        const sy = sourceNode.y;
        const tx = targetNode.x;
        const ty = targetNode.y;
        if (sx === tx) {
          path = `M ${sx} ${sy} L ${tx} ${ty}`;
        } else {
          const direction = ty >= sy ? 1 : -1;
          const bendY = sy + direction * Math.max(14, Math.abs(ty - sy) * 0.3);
          const curveY = bendY + direction * 12;
          path = `M ${sx} ${sy} L ${sx} ${bendY} C ${sx} ${curveY}, ${tx} ${curveY}, ${tx} ${bendY} L ${tx} ${ty}`;
        }
      } else {
        const curvature = Math.max(60, Math.abs(x2 - x1) * 0.25);
        const c1x = x1 + (x2 >= x1 ? curvature : -curvature);
        const c2x = x2 - (x2 >= x1 ? curvature : -curvature);
        path = `M ${x1} ${y1} C ${c1x} ${y1}, ${c2x} ${y2}, ${x2} ${y2}`;
      }

      result.push({
        id: `${source.id}->${targetId}`,
        sourceId: source.id,
        targetId,
        path,
        isPrimary: primaryParentById.value.get(source.id) === targetId,
      });
    }
  }
  return result;
});

const inboundById = computed(() => {
  const map = new Map<string, number>();
  for (const edge of edges.value) {
    map.set(edge.targetId, (map.get(edge.targetId) ?? 0) + 1);
  }
  return map;
});

const outboundById = computed(() => {
  const map = new Map<string, number>();
  for (const edge of edges.value) {
    map.set(edge.sourceId, (map.get(edge.sourceId) ?? 0) + 1);
  }
  return map;
});

const selectedNode = computed(() =>
  selectedNodeId.value ? nodeById.value.get(selectedNodeId.value) ?? null : null,
);

const selectedMessage = computed(() => {
  if (!selectedNodeId.value) {
    return null;
  }
  return graphMessages.value.find((item) => item.id === selectedNodeId.value) ?? null;
});

const selectedMessageHtml = computed(() => {
  if (!selectedMessage.value || typeof selectedMessage.value.content !== "string") {
    return "";
  }
  return renderMarkdown(selectedMessage.value.content, selectedMessage.value.id);
});

const selectedSummary = computed(() => {
  if (!selectedNode.value) {
    return null;
  }
  return {
    inbound: inboundById.value.get(selectedNode.value.id) ?? 0,
    outbound: outboundById.value.get(selectedNode.value.id) ?? 0,
  };
});

const svgWidth = computed(() =>
  layoutMode.value === "horizontal"
    ? Math.max(1200, nodes.value.length * 220 + 220)
    : layoutMode.value === "vertical"
      ? 700
      : Math.max(680, dagLabelX.value + 180),
);
const svgHeight = computed(() =>
  layoutMode.value === "horizontal"
    ? 620
      : layoutMode.value === "vertical"
      ? Math.max(720, nodes.value.length * 140 + 180)
      : Math.max(420, DAG_ROW_START + nodes.value.length * DAG_ROW_GAP + 80),
);

const edgeStyle = (edge: GraphEdge) => {
  if (layoutMode.value !== "dag") {
    return undefined;
  }
  const sourceNode = nodeById.value.get(edge.sourceId);
  if (!sourceNode) {
    return undefined;
  }
  return { stroke: colorForColumn(sourceNode.column) };
};

const selectNode = (nodeId: string) => {
  selectedNodeId.value = nodeId;
};

const openSelectedMessage = () => {
  if (selectedNode.value) {
    emit("openMessage", selectedNode.value.id);
  }
};
</script>

<template>
  <main class="graph-view">
    <div class="graph-header">
      <div class="title">Reference Graph</div>
      <div class="summary">
        {{ nodes.length }} messages · {{ edges.length }} references
      </div>
      <div class="layout-toggle">
        <button
          class="layout-button"
          :class="{ active: layoutMode === 'horizontal' }"
          type="button"
          @click="layoutMode = 'horizontal'"
        >
          Horizontal
        </button>
        <button
          class="layout-button"
          :class="{ active: layoutMode === 'vertical' }"
          type="button"
          @click="layoutMode = 'vertical'"
        >
          Vertical
        </button>
        <button
          class="layout-button"
          :class="{ active: layoutMode === 'dag' }"
          type="button"
          @click="layoutMode = 'dag'"
        >
          DAG
        </button>
      </div>
    </div>

    <div v-if="nodes.length === 0" class="empty">
      No messages available to render as a graph.
    </div>

    <div v-else class="graph-layout">
      <div class="graph-scroll">
        <svg :height="svgHeight" :width="svgWidth" class="graph-svg">
          <g v-if="layoutMode !== 'dag'" class="lanes">
            <template v-for="lane in laneLabels" :key="lane.type">
              <line
                :x1="layoutMode === 'horizontal' ? 60 : laneYByType[lane.type]"
                :x2="layoutMode === 'horizontal' ? svgWidth - 60 : laneYByType[lane.type]"
                :y1="layoutMode === 'horizontal' ? laneYByType[lane.type] : 40"
                :y2="layoutMode === 'horizontal' ? laneYByType[lane.type] : svgHeight - 40"
                class="lane-line"
              />
              <text
                :x="layoutMode === 'horizontal' ? 20 : laneYByType[lane.type] - 18"
                :y="layoutMode === 'horizontal' ? laneYByType[lane.type] + 4 : 24"
                class="lane-label"
              >
                {{ lane.label }}
              </text>
            </template>
          </g>
          <g v-else class="dag-guides">
            <line
              v-for="column in dagColumnCount"
              :key="`column-${column}`"
              :x1="dagColumnX(column - 1)"
              :x2="dagColumnX(column - 1)"
              :y1="42"
              :y2="svgHeight - 40"
              class="dag-column-line"
            />
          </g>

          <g class="edges">
            <path
              v-for="edge in edges"
              :key="edge.id"
              :class="{
                selected: selectedNodeId === edge.sourceId || selectedNodeId === edge.targetId,
                secondary: layoutMode === 'dag' && !edge.isPrimary,
              }"
              :d="edge.path"
              class="edge-path"
              :style="edgeStyle(edge)"
            />
          </g>

          <g v-if="layoutMode === 'dag'" class="dag-row-guides">
            <line
              v-for="node in nodes"
              :key="`guide-${node.id}`"
              :x1="node.x + DAG_NODE_RADIUS + 6"
              :x2="dagLabelX - 10"
              :y1="node.y"
              :y2="node.y"
              class="dag-label-guide"
            />
            <text
              v-for="node in nodes"
              :key="`label-${node.id}`"
              :x="dagLabelX"
              :y="node.y + 5"
              class="dag-node-text"
            >
              {{ node.displayId }}
            </text>
          </g>

          <g class="nodes">
            <g
              v-for="node in nodes"
              :key="node.id"
              :transform="
                layoutMode === 'dag'
                  ? `translate(${node.x}, ${node.y})`
                  : `translate(${node.x - 70}, ${node.y - 22})`
              "
              class="node"
              :class="[node.type, { selected: selectedNodeId === node.id }]"
              :style="layoutMode === 'dag' ? { '--dag-node-color': colorForColumn(node.column) } : undefined"
              @click="selectNode(node.id)"
            >
              <title>{{ node.id }}: {{ node.fullLabel }}</title>
              <template v-if="layoutMode === 'dag'">
                <circle :r="DAG_NODE_RADIUS" class="dag-node-circle" />
              </template>
              <template v-else>
                <rect height="44" rx="8" ry="8" width="140" />
                <text x="8" y="16" class="node-id">{{ node.displayId }}</text>
                <text x="8" y="32" class="node-label">{{ node.label }}</text>
              </template>
            </g>
          </g>
        </svg>
      </div>

      <aside class="node-panel">
        <div v-if="!selectedNode" class="node-empty">
          Select a node to inspect references.
        </div>
        <div v-else class="node-details">
          <div class="node-title">{{ selectedNode.id }}</div>
          <div class="node-meta">Type: {{ selectedNode.type }}</div>
          <div class="node-meta">Inbound: {{ selectedSummary?.inbound ?? 0 }}</div>
          <div class="node-meta">Outbound: {{ selectedSummary?.outbound ?? 0 }}</div>
          <div
            v-if="selectedMessageHtml"
            class="node-message"
            v-html="selectedMessageHtml"
          />
          <button class="open-button" type="button" @click="openSelectedMessage">
            Open in chat
          </button>
        </div>
      </aside>
    </div>
  </main>
</template>

<style lang="postcss" scoped>
.graph-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  padding: var(--space-24);
}

.graph-header {
  display: flex;
  justify-content: flex-start;
  align-items: center;
  gap: var(--space-12);
  margin-bottom: var(--space-12);
}

.title {
  font-size: 16px;
  font-weight: 700;
}

.summary {
  font-size: 12px;
  margin-right: auto;
}

.layout-toggle {
  display: flex;
  gap: var(--space-4);
}

.layout-button {
  border: 1px solid var(--knime-silver-sand);
  border-radius: var(--space-4);
  background: var(--knime-white);
  padding: var(--space-4) var(--space-8);
  font-size: 11px;
  cursor: pointer;
}

.layout-button.active {
  border-color: var(--knime-masala);
  color: var(--knime-masala);
  font-weight: 700;
}

.empty {
  font-size: 13px;
}

.graph-layout {
  display: flex;
  gap: var(--space-16);
  min-height: 0;
  height: 100%;
}

.graph-scroll {
  flex: 1;
  overflow: auto;
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: var(--space-4);
  background: var(--knime-white);
}

.graph-svg {
  display: block;
}

.lane-line {
  stroke: var(--knime-silver-sand-semi);
  stroke-width: 1;
}

.lane-label {
  font-size: 11px;
  fill: var(--knime-dove-gray);
}

.dag-column-line {
  stroke: var(--knime-silver-sand-semi);
  stroke-width: 1;
}

.dag-label-guide {
  stroke: var(--knime-silver-sand-semi);
  stroke-width: 1;
  stroke-dasharray: 4 4;
}

.dag-node-text {
  font-size: 13px;
  fill: var(--knime-masala);
}

.edge-path {
  fill: none;
  stroke: var(--knime-dove-gray);
  stroke-width: 1.5;
  opacity: 0.7;
}

.edge-path.secondary {
  stroke-dasharray: 5 4;
  opacity: 0.5;
}

.edge-path.selected {
  stroke: var(--knime-masala);
  opacity: 1;
  stroke-width: 2;
}

.node {
  cursor: pointer;
}

.node rect {
  fill: var(--knime-white);
  stroke: var(--knime-silver-sand);
  stroke-width: 1.2;
}

.node .dag-node-circle {
  fill: var(--dag-node-color, var(--knime-masala));
  stroke: var(--dag-node-color, var(--knime-masala));
  stroke-width: 2;
}

.node.selected rect {
  stroke: var(--knime-masala);
  stroke-width: 2;
}

.node.selected .dag-node-circle {
  stroke: var(--knime-masala);
  stroke-width: 2.5;
}

.node.human rect {
  fill: var(--knime-porcelain);
}

.node.ai rect {
  fill: var(--knime-white);
}

.node.error rect {
  fill: var(--knime-coral-light);
}

.node.tool rect {
  fill: var(--knime-aquamarine-semi);
}

.node.view rect {
  fill: var(--knime-azure);
}

.node-id {
  font-size: 10px;
  font-weight: 700;
  fill: var(--knime-masala);
}

.node-label {
  font-size: 10px;
  fill: var(--knime-dove-gray);
}

.node-panel {
  width: 240px;
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: var(--space-4);
  background: var(--knime-white);
  padding: var(--space-12);
  max-height: 100%;
  overflow: auto;
}

.node-empty {
  font-size: 12px;
}

.node-title {
  font-size: 13px;
  font-weight: 700;
  margin-bottom: var(--space-8);
}

.node-meta {
  font-size: 12px;
  margin-bottom: var(--space-4);
}

.node-message {
  margin-top: var(--space-8);
  border: 1px solid var(--knime-silver-sand-semi);
  border-radius: var(--space-4);
  padding: var(--space-8);
  font-size: 12px;
  line-height: 1.4;
  max-height: 260px;
  overflow: auto;
}

.node-message :deep() {
  overflow-wrap: break-word;
  overflow-x: hidden;

  & > *:first-child {
    margin-top: 0;
  }

  & > *:last-child {
    margin-bottom: 0;
  }

  & pre {
    border: 1px solid var(--knime-silver-sand);
    padding: var(--space-4);
    overflow: auto;
  }
}

.open-button {
  margin-top: var(--space-8);
  width: 100%;
  border: 1px solid var(--knime-silver-sand);
  border-radius: var(--space-4);
  background: var(--knime-white);
  padding: var(--space-6) var(--space-8);
  cursor: pointer;
}
</style>
