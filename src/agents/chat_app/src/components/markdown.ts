import DOMPurify from "dompurify";
import MarkdownIt from "markdown-it";

const md = new MarkdownIt();

const sanitizationConfig = {
  ALLOWED_TAGS: [
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
  ],
  ALLOWED_ATTR: ["style"],
};

export const renderMarkdown = (src: string) => {
  return DOMPurify.sanitize(md.render(src), sanitizationConfig);
};
