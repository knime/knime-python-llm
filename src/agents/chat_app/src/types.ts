export type Type = "human" | "ai" | "error";

export interface ToolCall  {
  id: string;
  name: string;
}

export interface MessageResponse {
  type: Type;
  content?: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
}

export interface Message extends MessageResponse {
  id: string;
}
