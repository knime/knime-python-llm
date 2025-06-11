import { type Ref, onBeforeUnmount, onMounted } from "vue";

export const useScrollToBottom = (container: Ref) => {
  const scrollToBottom = () => {
    if (container.value) {
      container.value.scrollTop = container.value.scrollHeight;
    }
  };

  let observer: MutationObserver;

  onMounted(() => {
    observer = new MutationObserver(scrollToBottom);

    if (container.value) {
      observer.observe(container.value, { childList: true });
    }
  });

  onBeforeUnmount(() => {
    observer?.disconnect();
  });
};
