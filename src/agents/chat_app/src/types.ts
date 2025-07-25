// Message-related entities
type MessageType = "human" | "ai" | "tool" | "view" | "error";

interface BaseMessageResponse {
  id: string;
  type: MessageType;
}

export interface ToolCall {
  id: string;
  name: string;
  args?: string;
}

export interface AiResponse extends BaseMessageResponse {
  type: "ai";
  name?: string | null;
  content: string;
  toolCalls?: ToolCall[];
}

export interface ViewResponse extends BaseMessageResponse {
  content: string;
  type: "view";
  name: string;
}

export interface ErrorResponse extends BaseMessageResponse {
  content: string;
  type: "error";
}

export interface HumanResponse extends BaseMessageResponse {
  content: string;
  type: "human";
}

export interface ToolResponse extends BaseMessageResponse {
  content: string;
  type: "tool";
  toolCallId: string;
}

export type MessageResponse =
  | AiResponse
  | ViewResponse
  | ErrorResponse
  | HumanResponse
  | ToolResponse;

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

export type ChatItem = MessageResponse | Timeline;

// misc types
export interface Config {
  show_tool_calls_and_results: boolean;
}
