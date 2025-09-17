import { type Ref, onBeforeUnmount, onMounted } from "vue";

export const useScrollToBottom = (
  container: Ref<HTMLElement | null>,
  list: Ref<HTMLElement | null>,
) => {
  const scrollToBottom = () => {
    if (container.value) {
      container.value.scrollTop = container.value.scrollHeight;
    }
  };

  let observer: MutationObserver;

  onMounted(() => {
    observer = new MutationObserver(scrollToBottom);

    if (container.value && list.value) {
      observer.observe(list.value, { childList: true });
    }
  });

  onBeforeUnmount(() => {
    observer?.disconnect();
  });
};
