export type Role = "user" | "assistant" | "system";

export interface MessageResponse {
  content: string;
  role: Role;
  toolCalls?: any[];
}

export interface Message extends MessageResponse {
  id: string;
}
