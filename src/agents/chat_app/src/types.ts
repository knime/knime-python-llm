// Message-related entities
type MessageType = "human" | "ai" | "tool" | "view" | "error" | "warning";

interface BaseMessage {
  id: string;
  type: MessageType;
}

export interface ToolCall {
  id: string;
  name: string;
  args?: string;
}

export interface AiMessage extends BaseMessage {
  type: "ai";
  name?: string | null;
  content: string;
  toolCalls?: ToolCall[];
}

export interface ViewMessage extends BaseMessage {
  content: string;
  type: "view";
  name: string;
}

export interface ErrorMessage extends BaseMessage {
  content: string;
  type: "error";
}

export interface WarningMessage extends BaseMessage {
  content: string;
  type: "warning";
}

export interface HumanMessage extends BaseMessage {
  content: string;
  type: "human";
}

export interface ToolMessage extends BaseMessage {
  content: string;
  type: "tool";
  toolCallId: string;
}

export type Message =
  | AiMessage
  | ViewMessage
  | ErrorMessage
  | WarningMessage
  | HumanMessage
  | ToolMessage;

// Timeline-related entities
export type TimelineItemType = "reasoning" | "tool_call";

export interface BaseTimelineItem {
  id: string;
  type: TimelineItemType;
}

export interface ReasoningTimelineItem extends BaseTimelineItem {
  type: "reasoning";
  content?: string;
}

export interface ToolCallTimelineItem extends BaseTimelineItem {
  type: "tool_call";
  name: string;
  status: "running" | "completed" | "failed";
  args?: string;
  content?: string;
}

export type TimelineItem = ReasoningTimelineItem | ToolCallTimelineItem;

export interface Timeline {
  id: string;
  items: TimelineItem[];
  label: string;
  status: "active" | "completed";
  type: "timeline";
}

export type ChatItem = Message | Timeline;

// misc types
export interface Config {
  show_tool_calls_and_results: boolean;
  reexecution_trigger: ReexecutionTrigger;
}

export type ReexecutionTrigger = "NONE" | "INTERACTION";

export type InitializationState = "idle" | "ready" | "error";

export interface ViewData {
  conversation: Message[];
  config: Config;
}

export interface WorkflowInfo {
  projectId: string;
  workflowId: string;
}
