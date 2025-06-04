export type Type = "human" | "ai" | "tool" | "error";

export interface ToolCall {
  id: string;
  name: string;
  args?: string;
}

export interface AiResponse {
  id: string;
  type: "ai";
  name: string | null;
  content: string;
  toolCalls: ToolCall[];
}

export interface ToolResponse {
  id: string;
  toolCallId: string;
  type: "tool";
  name: string;
  content: string;
}

export interface ErrorResponse {
  id: string;
  content: string;
  type: "error";
}

export interface UserResponse {
  id: string;
  content: string;
  type: "human";
}

export type MessageResponse =
  | AiResponse
  | ToolResponse
  | ErrorResponse
  | UserResponse;
