import type { DefineComponent } from "vue";

type Type = "human" | "ai" | "tool" | "view" | "error";

export interface ToolCall {
  id: string;
  name: string;
  args?: string;
}

interface BaseMessageResponse {
  id: string;
  type: Type;
}

export interface AiResponse extends BaseMessageResponse {
  type: "ai";
  name?: string | null;
  content: string;
  toolCalls?: ToolCall[];
}

export interface ToolResponse extends BaseMessageResponse {
  toolCallId: string;
  type: "tool";
  name: string;
  content: string;
}

export interface ViewResponse extends BaseMessageResponse {
  content: string;
  type: "view";
}

export interface ErrorResponse extends BaseMessageResponse {
  content: string;
  type: "error";
}

export interface HumanResponse extends BaseMessageResponse {
  content: string;
  type: "human";
}

export type MessageResponse =
  | AiResponse
  | ToolResponse
  | ViewResponse
  | ErrorResponse
  | HumanResponse;

export type MessageComponentMap = {
  [K in MessageResponse["type"]]: DefineComponent<
    Extract<MessageResponse, { type: K }>,
    {},
    any
  >;
};
