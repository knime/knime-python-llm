import { expect } from "vitest";
import type {
  AiMessage,
  Config,
  ToolMessage,
  ViewMessage,
  Timeline,
  ToolCallTimelineItem,
  reexecution_trigger,
  HumanMessage,
} from "@/types";

export const createErrorMessage = (content: string) => ({
  id: expect.any(String),
  type: "error",
  content,
});

export const createAiMessage = (
  content: string,
  toolCalls?: AiMessage["toolCalls"],
): AiMessage => ({
  id: expect.any(String),
  type: "ai",
  content,
  ...(toolCalls && { toolCalls }),
});

export const createUserMessage = (content: string): HumanMessage => ({
  id: expect.any(String),
  type: "human",
  content,
});

export const createToolMessage = (
  content: string,
  toolCallId: string,
): ToolMessage => ({
  id: expect.any(String),
  type: "tool",
  content,
  toolCallId,
});

export const createViewMessage = (
  content: string,
  name: string,
): ViewMessage => ({
  id: expect.any(String),
  type: "view",
  content,
  name,
});

export const createConfig = (
  showToolCalls = true,
  reexecution_trigger: reexecution_trigger = "NONE",
): Config => ({
  show_tool_calls_and_results: showToolCalls,
  reexecution_trigger,
});

export const createTimeline = (
  label = "Using tools",
  status: Timeline["status"] = "active",
  items: Timeline["items"] = [],
): Timeline => ({
  id: expect.any(String),
  type: "timeline",
  label,
  status,
  items,
});

export const createToolCall = (
  name = "search_tool",
  args = '{"query": "test"}',
) => ({
  id: expect.any(String),
  name,
  args,
});

export const createToolCallTimelineItem = (
  name = "search",
  status: ToolCallTimelineItem["status"] = "running",
  content?: string,
): ToolCallTimelineItem => ({
  id: expect.any(String),
  type: "tool_call",
  name,
  status,
  ...(content && { content }),
});
