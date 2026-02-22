const MESSAGE_REF_PATTERN =
  /#([A-Za-z0-9][A-Za-z0-9_-]*(?:__[A-Za-z0-9][A-Za-z0-9_-]*)?)/g;

export const normalizeMessageId = (id: string): string => id.split("__")[0];

export const extractReferenceTargets = (content: string): string[] => {
  const targets = new Set<string>();
  for (const match of content.matchAll(MESSAGE_REF_PATTERN)) {
    if (match[1]) {
      targets.add(match[1]);
    }
  }
  return Array.from(targets);
};

export const extractReferencedMessageIds = (content: string): string[] =>
  extractReferenceTargets(content).map(normalizeMessageId);
