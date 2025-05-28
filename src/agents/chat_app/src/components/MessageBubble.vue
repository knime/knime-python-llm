<script setup lang="ts">
import { defineProps, computed } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

const props = defineProps<{
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
}>();

const htmlContent = computed(() => {
  if (props.role !== 'assistant') {
    return props.content;
  }

  const rawHtml = marked.parse(props.content);
  return DOMPurify.sanitize(rawHtml);
});
</script>

<template>
  <div class="message-bubble" :class="role">
    <div 
      v-if="role === 'assistant'" 
      class="message-content markdown-content" 
      v-html="htmlContent"
    ></div>
    <div v-else class="message-content">{{ content }}</div>
    <div class="message-time">{{ timestamp }}</div>
  </div>
</template>

<style scoped>
.message-bubble {
  padding: 12px 16px;
  border-radius: 18px;
  position: relative;
  overflow-wrap: break-word;
  word-wrap: break-word;
  word-break: break-word;
}

.message-bubble.user {
  background-color: var(--user-bubble-bg);
  color: var(--user-bubble-text);
  border-bottom-right-radius: 4px;
}

.message-bubble.assistant {
  background-color: var(--ai-bubble-bg);
  color: var(--ai-bubble-text);
  border-bottom-left-radius: 4px;
}

.message-content {
  font-size: 0.95rem;
  line-height: 1.5;
}

.message-time {
  font-size: 0.7rem;
  opacity: 0.8;
  margin-top: 6px;
  text-align: right;
}

/* Override scoped style for v-html content */
:deep(.markdown-content *) {
  color: inherit;
}

:deep(.markdown-content pre) {
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 6px;
  overflow-x: auto;
  padding: 12px;
}

:deep(.markdown-content code) {
  background-color: rgba(0, 0, 0, 0.1);
  padding: 2px 4px;
  border-radius: 4px;
  font-family: 'SF Mono', 'Monaco', 'Menlo', 'Courier New', monospace;
  font-size: 0.85em;
}

:deep(.markdown-content a) {
  color: var(--color-primary-light);
  text-decoration: none;
}

:deep(.markdown-content a:hover) {
  text-decoration: underline;
}
</style>