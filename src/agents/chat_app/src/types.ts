export type Role = "user" | "assistant" | "system";

export interface Message {
  id: string;
  content: string;
  role: Role;
  timestamp: Date;
}
