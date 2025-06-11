import { describe, expect, it, vi } from "vitest";
import { flushPromises } from "@vue/test-utils";

import { JsonDataService } from "@knime/ui-extension-service";

import {
  createAiMessage,
  createErrorMessage,
  createUserMessage,
} from "@/test/factories/messages";
import { withSetup } from "@/test/utils/withSetup";
import { initError, processingError, sendingError, useChat } from "../useChat";

describe("useChat", () => {
  const createChat = async () => {
    const [chat] = withSetup(() => useChat());
    chat.resetChat();
    await flushPromises();
    return chat;
  };
  const userMessage = createUserMessage("New message!");
  const aiMessage = createAiMessage("AI Response!");

  it("inits with error message if JsonDataService initialization fails", async () => {
    JsonDataService.getInstance = vi
      .fn()
      .mockRejectedValueOnce(new Error("Initialization error"));
    const chat = await createChat();

    expect(chat.messages.value).toEqual([createErrorMessage(initError)]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("inits with empty messages if no initial data is provided", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockResolvedValueOnce(undefined),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    expect(chat.messages.value).toEqual([]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("inits with initial message if initial message is provided", async () => {
    const initialMessage = createAiMessage("Welcome!");
    const jsonDataServiceMock = {
      data: vi.fn().mockResolvedValueOnce(initialMessage),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    expect(chat.messages.value).toEqual([initialMessage]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("inits with empty messages if fetching initial message fails", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockRejectedValueOnce(new Error("Fetch error")),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    expect(chat.messages.value).toEqual([]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("sends a message and update messages", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockImplementation((request) => {
        return request.method === "get_last_messages" ? [aiMessage] : undefined;
      }),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    await chat.sendMessage(userMessage.content);

    expect(chat.messages.value).toEqual([userMessage, aiMessage]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("keeps polling if no AI responses are received", async () => {
    let counter = 0;
    const jsonDataServiceMock = {
      data: vi.fn().mockImplementation((request) => {
        if (request.method === "get_last_messages" && counter < 5) {
          counter++;
          return [];
        }
        return request.method === "get_last_messages" ? [aiMessage] : undefined;
      }),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    await chat.sendMessage(userMessage.content);

    expect(chat.messages.value).toEqual([userMessage, aiMessage]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("does not send an empty message", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockImplementation((request) => {
        return request.method === "get_last_messages" ? [aiMessage] : undefined;
      }),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    await chat.sendMessage("");

    expect(chat.messages.value).toEqual([]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("shows an error message if sending user message fails", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockImplementation((request) => {
        if (request.method === "post_user_message") {
          throw new Error("Fetch error");
        }
        return request.method === "get_last_messages" ? [aiMessage] : undefined;
      }),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    await chat.sendMessage(userMessage.content);

    expect(chat.messages.value).toEqual([
      userMessage,
      createErrorMessage(sendingError),
    ]);
    expect(chat.isLoading.value).toBe(false);
  });

  it("shows an error message if fetching new messages fails", async () => {
    const jsonDataServiceMock = {
      data: vi.fn().mockImplementation((request) => {
        if (request.method === "get_last_messages") {
          throw new Error("Fetch error");
        }
        return undefined;
      }),
    };
    JsonDataService.getInstance = vi
      .fn()
      .mockResolvedValue(jsonDataServiceMock);
    const chat = await createChat();

    await chat.sendMessage(userMessage.content);

    expect(chat.messages.value).toEqual([
      userMessage,
      createErrorMessage(processingError),
    ]);
    expect(chat.isLoading.value).toBe(false);
  });
});
