import DOMPurify from "dompurify";
import markdownit from "markdown-it";
import markdownItTable from "markdown-it-multimd-table";

const md = markdownit({
  html: true,
  linkify: true,
  typographer: true,
  breaks: true,
}).use(markdownItTable);

const sanitizationConfig = {
  ALLOWED_TAGS: [
    "a",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "strong",
    "b",
    "i",
    "em",
    "blockquote",
    "ul",
    "ol",
    "li",
    "code",
    "pre",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
  ],
  ALLOWED_ATTR: ["style", "href"],
};

const HEADING_SELECTOR = "h1, h2, h3, h4, h5, h6";
const SECTION_ID_FALLBACK = "section";

const slugify = (text: string): string => {
  const slug = text
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
  return slug || SECTION_ID_FALLBACK;
};

const headingId = (
  messageId: string,
  headingText: string,
  occurrenceBySlug: Map<string, number>,
): string => {
  const slug = slugify(headingText);
  const occurrence = (occurrenceBySlug.get(slug) ?? 0) + 1;
  occurrenceBySlug.set(slug, occurrence);
  const suffix = occurrence > 1 ? `-${occurrence}` : "";
  return `${messageId}__${slug}${suffix}`;
};

const addSectionIds = (container: HTMLElement, messageId: string): void => {
  const occurrenceBySlug = new Map<string, number>();
  container.querySelectorAll(HEADING_SELECTOR).forEach((heading) => {
    heading.id = headingId(
      messageId,
      heading.textContent ?? SECTION_ID_FALLBACK,
      occurrenceBySlug,
    );
  });
};

export const renderMarkdown = (src: string, messageId?: string) => {
  const sanitizedHtml = DOMPurify.sanitize(md.render(src), sanitizationConfig);
  if (!messageId) {
    return sanitizedHtml;
  }

  const parser = new DOMParser();
  const document = parser.parseFromString(
    `<div id="__root">${sanitizedHtml}</div>`,
    "text/html",
  );
  const root = document.getElementById("__root");
  if (!root) {
    return sanitizedHtml;
  }

  addSectionIds(root, messageId);
  return root.innerHTML;
};
