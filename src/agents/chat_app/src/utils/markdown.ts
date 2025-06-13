import DOMPurify from "dompurify";
import markdownit from "markdown-it";
import markdownItTable from "markdown-it-multimd-table";

const md = markdownit({ html: true, linkify: true, typographer: true, breaks: true }).use(markdownItTable);

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
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
  ],
  ALLOWED_ATTR: ["style"],
};

export const renderMarkdown = (src: string) =>
  DOMPurify.sanitize(md.render(src), sanitizationConfig);
