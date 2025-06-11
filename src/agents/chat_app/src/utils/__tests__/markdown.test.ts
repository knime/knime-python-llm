import { describe, expect, it } from "vitest";

import { renderMarkdown } from "../markdown";

// Utility to check presence of tags
const includesTag = (html: string, tag: string) =>
  new RegExp(`<${tag}[\\s>]`, "i").test(html);

describe("renderMarkdown", () => {
  it("renders basic Markdown to HTML", () => {
    const input = "# Hello World";

    const result = renderMarkdown(input);

    expect(result).toContain("<h1>");
    expect(result).toContain("Hello World");
  });

  it("sanitizes disallowed tags", () => {
    const dirty = '<script>alert("XSS")</script><p>Text</p>';

    const result = renderMarkdown(dirty);

    expect(result).not.toContain("<script>");
    expect(result).toContain("<p>Text</p>");
  });

  it("retains allowed tags", () => {
    const input = "**bold**\n\n- item 1\n- item 2";

    const result = renderMarkdown(input);

    expect(includesTag(result, "strong")).toBe(true);
    expect(includesTag(result, "ul")).toBe(true);
    expect(includesTag(result, "li")).toBe(true);
  });

  it("removes disallowed attributes", () => {
    const input = '<p onclick="evil()">Click me</p>';

    const result = renderMarkdown(input);

    expect(result).not.toContain("onclick=");
    expect(result).toContain("<p>Click me</p>");
  });

  it("retains allowed attributes", () => {
    const input = '<p style="color:red;">Red text</p>';

    const result = renderMarkdown(input);

    expect(result).toContain('style="color:red;"');
  });

  it("escapes inline code", () => {
    const input = "`<script>`";

    const result = renderMarkdown(input);

    expect(result).toContain("&lt;script&gt;");
    expect(includesTag(result, "code")).toBe(true);
  });
});
