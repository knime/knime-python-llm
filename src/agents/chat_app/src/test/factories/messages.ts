import { expect } from "vitest";

export const createErrorMessage = (content: string) => ({
  id: expect.any(String),
  type: "error",
  content,
});

export const createAiMessage = (content: string) => ({
  id: expect.any(String),
  type: "ai",
  content,
});

export const createUserMessage = (content: string) => ({
  id: expect.any(String),
  type: "human",
  content,
});
